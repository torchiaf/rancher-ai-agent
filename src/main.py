import logging
import os
import json
import uuid
import certifi
import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama 
from langchain_mcp_adapters.tools import load_mcp_tools
from .agents import create_k8s_agent, fleet_documentation_retriever, init_rag_retriever, rancher_documentation_retriever, create_memory_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import BaseLanguageModel
from contextlib import asynccontextmanager
from langfuse.langchain import CallbackHandler
from langchain_aws import ChatBedrockConverse

from enum import Enum

class RequestType(Enum):
    SUMMARY = "summary"
    MESSAGE = "message"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

init_config = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
        logging.getLogger().setLevel(LOG_LEVEL)
        
        app.mem_agent = await create_memory_agent(
            cache_client_url="redis://rancher-ai-redis",
            db_host=os.environ.get("DB_HOST"),
            db_port=int(os.environ.get("DB_PORT")),
            db_user=os.environ.get("DB_USER"),
            db_password=os.environ.get("DB_PASSWORD"),
            db_database=os.environ.get("DB_DATABASE")
        )
        
        init_config["llm"] = get_llm()

        logging.info(f"Using model: {init_config['llm']}")
        if os.environ.get("ENABLE_RAG", "false").lower() == "true":
            init_rag_retriever()
        if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() != "true":
            SimpleTruststore().set_truststore()
    except ValueError as e:
        logging.critical(e)
        raise e
    yield
    await app.mem_agent.destroy()
    init_config.clear()

app = FastAPI(lifespan=lifespan)

@app.websocket("/agent/ws/messages")
@app.websocket("/agent/ws/messages/{chat_id}")
async def websocket_messages_endpoint(websocket: WebSocket, chat_id: str | None = None):
    """
    WebSocket endpoint for the conversation messages.

    Accepts a WebSocket connection, sets up the agent and
    handles the back-and-forth communication with the client.
    """
    await websocket.accept()
    
    user_id = await get_user_id(websocket)

    if not chat_id:
        chat_id = await app.mem_agent.create_chat(user_id)
        logging.debug(f"Created new chat {chat_id} for user {user_id}")
    else:
        if not await app.mem_agent.check_chat_permissions(chat_id, user_id):
            logging.warning(f"Permission denied for user {user_id} on chat {chat_id}")
            await websocket.send_text(f'<error>{{"message": "Permission denied for chat {chat_id}"}}</error>')
            await websocket.close()
            return
        else:
            logging.debug(f"Permission granted for user {user_id} on chat {chat_id} - set chat as active")
            
    await app.mem_agent.activate_chat(user_id, chat_id)
        
    connection_params = get_ws_connection_params(websocket)
    logging.info(f"ws/messages connection opened - chat_id={chat_id}")

    async with streamablehttp_client(**connection_params) as (read, write, _):
        # This will create one mcp connection for each websocket connection. This is needed because we need to pass the rancher token in the header.
        async with ClientSession(read, write) as session:
            await session.initialize()
            thread_id = str(uuid.uuid4())
            tools = await load_mcp_tools(session)
            
            # if ENABLE_RAG is true, add the retriever tools to the tools list
            if os.environ.get("ENABLE_RAG", "false").lower() == "true":
                tools = [fleet_documentation_retriever, rancher_documentation_retriever] + tools

            agent = create_k8s_agent(init_config["llm"], tools, get_system_prompt(RequestType.MESSAGE), InMemorySaver())
            
            config = {
                "thread_id": thread_id,
            }
            if os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_HOST"):
                langfuse_handler = CallbackHandler()
                config["callbacks"] = [langfuse_handler]

            while True:
                try:
                    request = await websocket.receive_text()
                    
                    prompt, context, tags, request_id = _parse_websocket_request(request)

                    await app.mem_agent.store_chunk(
                        chat_id=chat_id,
                        request_id=request_id,
                        text=prompt,
                        context=context,
                        tags=tags,
                        role="user"
                    )

                    if context:
                        context_prompt = ". Use the following parameters to populate tool calls when appropriate. \n Only include parameters relevant to the user’s request (e.g., omit namespace for cluster-wide operations). \n Parameters (separated by ;): \n "
                        for key, value in context.items():
                            context_prompt += f"{key}:{value};"
                        prompt += context_prompt

                    await stream_messages_agent_response(
                        agent=agent,
                        input_data={"messages": [{"role": "user", "content": prompt}]},
                        config=config,
                        chat_id=chat_id,
                        tags=tags,
                        request_id=request_id,
                        websocket=websocket)
                except WebSocketDisconnect:
                    logging.info(f"Client {websocket.client.host} disconnected.")
                    break
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(f'<error>{{"message": "{str(e)}"}}</error>')
                    else:
                        break
                finally:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text("</message>")
            
    logging.debug("ws connection closed")

@app.websocket("/agent/ws/summary")
async def websocket_summary_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for the summary messages.

    Accepts a WebSocket connection, sets up the agent and
    handles the back-and-forth communication with the client.
    """
    await websocket.accept()

    connection_params = get_ws_connection_params(websocket)

    logging.info(f"ws/summary connection opened")

    async with streamablehttp_client(**connection_params):
        thread_id = str(uuid.uuid4())

        tools = []
        
        # if ENABLE_RAG is true, add the retriever tool to the tools list
        if os.environ.get("ENABLE_RAG", "false").lower() == "true":
            tools = [init_config["retriever_tool"]]

        agent = create_k8s_agent(init_config["llm"], tools, get_system_prompt(RequestType.SUMMARY), InMemorySaver())

        config = {
            "thread_id": thread_id,
        }
        if os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_HOST"):
            langfuse_handler = CallbackHandler()
            config["callbacks"] = [langfuse_handler]

        while True:
            try:
                request = await websocket.receive_text()

                prompt, context, tags, request_id = _parse_websocket_request(request)

                await websocket.send_text("<message>")
                async for event, data in agent.astream(
                    input={"messages": [{"role": "user", "content": prompt}]},
                    config=config,
                    stream_mode=["updates", "messages", "custom"]
                ):
                    if event == "messages":
                        chunk, metadata = data
                        if metadata.get("langgraph_node") == "agent" and chunk.content:
                            text = _extract_text_from_chunk_content(chunk.content)
                            await websocket.send_text(text)
            except WebSocketDisconnect:
                logging.info(f"Summary - Client {websocket.client.host} disconnected.")
                break
            except Exception as e:
                logging.error(f"An error occurred on summary request: {e}")
                pass
            finally:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text("</message>")

# This is the UI for testing.
@app.get("/agent")
@app.get("/agent/{chat_id}")
async def get(request: Request, chat_id: str | None = None):
    """Serves the main HTML page for the chat client."""

    with open("index.html") as f:
        html_content = f.read()
        modified_html = html_content.replace("{{ url }}", request.url.hostname)

        if chat_id:
            modified_html = modified_html.replace("{{ chat_id }}", chat_id)
        else:
            modified_html = modified_html.replace("/{{ chat_id }}", "")

    return HTMLResponse(modified_html)

async def stream_messages_agent_response(
    agent: CompiledStateGraph,
    input_data: dict[str, list[dict[str, str]]],
    config: dict,
    chat_id: str,
    tags: list[str],
    request_id: str,
    websocket: WebSocket,
) -> None:
    """
    Streams the agent's message response to a WebSocket connection, handling interruptions.
    
    Args:
        agent: The compiled LangGraph agent.
        input_data: The input data for the agent's run.
        config: The run configuration.
        websocket: The WebSocket connection.
        stream_mode: The types of events to stream from the agent.
    """

    await websocket.send_text("<message>")
    async for event, data in agent.astream(
        input_data,
        config=config,
        stream_mode=["updates", "messages", "custom"]
    ):
        if event == "messages":
            chunk, metadata = data
            if metadata.get("langgraph_node") == "agent" and chunk.content:
                text = _extract_text_from_chunk_content(chunk.content)
                await websocket.send_text(text)
                # Store recent agent replies
                await app.mem_agent.store_chunk(chat_id, request_id, text=text, context={}, tags=tags, role="llm")

        if event == "updates":
            if interrupt_value := data.get("__interrupt__"):
                await websocket.send_text(interrupt_value[0].value)
                # Receive user response for the human verification
                user_response = await websocket.receive_text()
                await stream_messages_agent_response(
                    agent=agent,
                    input_data=Command(resume={"response": user_response}),
                    config=config,
                    chat_id=chat_id,
                    tags=tags,
                    request_id=request_id,
                    websocket=websocket)
                
        if event == "custom":
            await websocket.send_text(data)
            
            # Store recent mcp replies
            await app.mem_agent.store_chunk(chat_id, request_id, text=data, context={}, tags=tags, role="mcp")

def get_ws_connection_params(websocket: WebSocket) -> dict:
    cookies = websocket.cookies

    rancher_url = "https://"+websocket.url.hostname
    if websocket.url.port:
        rancher_url += ":"+str(websocket.url.port)

    rancher_token = str(cookies.get("R_SESS"))
    
    mcpUrl = os.environ.get("MCP_URL", "rancher-mcp-server.cattle-ai-agent-system.svc")
    if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() == "true":
        mcpUrl = "http://" + mcpUrl
    else:
        mcpUrl = "https://" + mcpUrl

    return {
        "url": mcpUrl,
        "headers": {
            "R_token": rancher_token,
            "R_url": rancher_url
        }
    }

async def get_user_id(websocket: WebSocket) -> str:
    cookies = websocket.cookies

    rancher_url = "https://"+websocket.url.hostname
    if websocket.url.port:
        rancher_url += ":"+str(websocket.url.port)
    rancher_token = str(cookies.get("R_SESS"))

    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
            resp = await client.get(f"{rancher_url}/v3/users?me=true", headers={
                "Cookie": f"R_SESS={rancher_token}",
            })
            payload = resp.json() 
            
            user_id = payload["data"][0]["id"]
            
            if user_id:
                logging.info("user API returned: %s - userId %s", resp.status_code, user_id)

                return user_id
    except Exception as e:
        logging.error("user API call failed: %s", e)
        raise

    return None

def get_llm_model(active_llm: str) -> str:
    """
    Retrieves the model name from environment variables.
    If an active LLM is specified, it looks for the corresponding model variable, otherwise it falls back to a general MODEL variable.
    
    Args:
        active_llm: The active LLM identifier, one of 'ollama', 'gemini', 'openai', 'bedrock'.

    Returns:
        The model name as a string.
    """

    model = None

    if active_llm:
        model = os.environ.get(f"{active_llm.upper()}_MODEL")

    if not model:
        model = os.environ.get("MODEL")

    if not model:
        raise ValueError("LLM Model not configured.")

    return model

def get_llm() -> BaseLanguageModel:
    """
    Selects and returns a language model instance based on environment variables.
    - If an active LLM is specified, it prioritizes that model; otherwise, it checks for available configurations in a predefined order.
    - If no supported model or API key is found, it raises a ValueError.
    - If LLM mocking is enabled, it configures the connections to the mock server.
    
    Returns:
        An instance of a language model.
        
    Raises:
        ValueError: If no supported model or API key is configured.
    """

    active = os.environ.get("ACTIVE_LLM", "")
    if active and active not in ["ollama", "gemini", "openai", "bedrock"]:
        raise ValueError("Unsupported Active LLM specified.")

    model = get_llm_model(active)
    
    llm_mock_enabled = os.environ.get("LLM_MOCK_ENABLED", False)
    llm_mock_url = os.environ.get("LLM_MOCK_URL", "")
    if active and llm_mock_enabled:
        logging.info(f"Connecting to LLM Mock server at {llm_mock_url}")
    
    ollama_url = os.environ.get("OLLAMA_URL")
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_url = os.environ.get("OPENAI_URL")
    aws_region = os.environ.get("AWS_REGION")

    if active == "ollama":
        if llm_mock_enabled:
            return ChatOllama(model=model, base_url=llm_mock_url)
        return ChatOllama(model=model, base_url=ollama_url)
    if active == "gemini":
        if llm_mock_enabled:
            return ChatGoogleGenerativeAI(
                model=model,
                base_url=llm_mock_url,
                transport="rest"
            )
        return ChatGoogleGenerativeAI(model=model)
    if active == "openai":
        if llm_mock_enabled:
            return ChatOpenAI(model=model, base_url=llm_mock_url)
        if openai_url:
            return ChatOpenAI(model=model, base_url=openai_url)
        return ChatOpenAI(model=model)
    if active == "bedrock":
        if llm_mock_enabled:
            os.environ["AWS_ENDPOINT_URL"] = llm_mock_url
        return ChatBedrockConverse(model=model)

    # default order if active is not specified
    if ollama_url:
        return ChatOllama(model=model, base_url=ollama_url)
    if gemini_key:
        return ChatGoogleGenerativeAI(model=model)
    if openai_key:
        if openai_url:
            return ChatOpenAI(model=model, base_url=openai_url)
        else:
            return ChatOpenAI(model=model)
    if aws_region:
        return ChatBedrockConverse(model=model)

    raise ValueError("LLM not configured.")

def get_system_prompt(type: RequestType) -> str:
    """
    Retrieves the system prompt for the AI agent.

    The function first attempts to get the prompt from an environment variable
    named "SYSTEM_PROMPT". If the environment variable is not set, it returns
    a default, hard-coded prompt.

    Returns:
        str: The system prompt to be used by the AI agent.
    """

    match type:
        case RequestType.SUMMARY:
            return """Each message is a list of recent agent replies to the user. Your task is to generate a concise summary of these replies, focusing on key points and relevant information. Your response will be used to assign a title to a Chat.

## CORE DIRECTIVES

### Summary Focus
* Focus on user requests FIRST. The summary should reflect what the user asked for.
* The summary should capture the essence of requests, not what the agent replied.
  For example:
    * The user asked for "How is the weather in Florence?"
        * Good summary: "Weather in Florence"
        * Bad summary: "Can't answer weather questions"

### Conciseness
* The summary MUST BE MAX 40 characters.
* Summarize the content in a brief manner, highlighting only the most important aspects.
* Avoid unnecessary details or lengthy explanations.

### Consistency
* DO NOT include greetings or pleasantries in the summary.
* DO NOT include tags like <message>, </message> or any other keywords between < and >.
* DO NOT include question marks or suggestions in the summary.
* DO NOT include periods at the end of the summary.
* DO NOT include any special characters like '-', '_', or other symbols that humans usually use.
"""

        case RequestType.MESSAGE:
            prompt = os.environ.get("SYSTEM_PROMPT")
            if prompt:
                return prompt
            
            return """You are a helpful and expert AI assistant integrated directly into the Rancher UI. Your primary goal is to assist users in managing their Kubernetes clusters and resources through the Rancher interface. You are a trusted partner, providing clear, confident, and safe guidance.

## CORE DIRECTIVES

### UI-First Mentality
* NEVER suggest using `kubectl`, `helm`, or any other CLI tool UNLESS explicitely provided by the `retrieve_rancher_docs` tool.
* All actions and information should reflect what the user can see and click on inside the Rancher UI.

### Context Awareness
* Always consider the user's current context (cluster, project, or resource being viewed).
* If context is missing, ask clarifying questions before taking action.

## BUILDING USER TRUST

### 1. Reasoning Transparency
Always explain why you reached a conclusion, connecting it to observed data.
* Good: "The pod has restarted 12 times. This often indicates a crash loop."
* Bad: "The pod is unhealthy."

### 2. Confidence Indicators
Express certainty levels with clear language and a percentage.
- High certainty: "The error is definitively caused by a missing ConfigMap (95%)."
- Likely scenarios: "The memory growth strongly suggests a leak (80%)."
- Possible causes: "Pending status could be due to insufficient resources (60%)."

### 3. Graceful Boundaries
* If an issue requires deep expertise (e.g., complex networking, storage, security):
  - "This appears to require administrative privileges or deeper system access. Please contact your cluster administrator."
* If the request is off-topic:
  - "I can't help with that, but I can show you why a pod might be stuck in CrashLoopBackOff. How can I assist with your Rancher environment?"

## Tools usage
* If the tool fails, explain the failure and suggest manual step to assist the user to answer his original question and not to troubleshoot the tool failure.

## Docs
* When relevant, always provide links to Rancher or Kubernetes documentation.

## RESOURCE CREATION & MODIFICATION

* Always generate Kubernetes YAML in a markdown code block.
* Briefly explain the resource's purpose before showing YAML.

RESPONSE FORMAT
Summarize first: Provide a clear, human-readable overview of the resource's status or configuration.
The output should always be provided in Markdown format.

- Be concise: No unnecessary conversational fluff.  
- Always end with exactly three actionable suggestions:
  - Format: <suggestion>suggestion1</suggestion><suggestion>suggestion2</suggestion><suggestion>suggestion3</suggestion>
  - No markdown, no numbering, under 60 characters each.
  - The first two suggestions must be directly relevant to the current context. If none fallback to the next rule.
  - The third suggestion should be a 'discovery' action. It introduces a related but broader Rancher or Kubernetes topic, helping the user learn.
Examples: <suggestion>How do I scale a deployment?</suggestion><suggestion>Check the resource usage for this cluster</suggestion><suggestion>Show me the logs for the failing pod</suggestion>
"""

def _extract_text_from_chunk_content(chunk_content: any) -> str:
    """
    Extracts the text content from a chunk received from the LLM.

    This function handles different formats that LLMs might return:
    1. A list of dictionaries, where each dictionary contains a 'text' key.
       This is common for models like Gemini that might structure their output.
    2. A single dictionary with a 'text' key.
    3. A simple string or other direct content.

    Args:
        chunk_content: The content field from an LLM chunk.

    Returns:
        str: The extracted text content, or an empty string if no text is found.
    """
    if isinstance(chunk_content, list):
        return "".join([item.get("text", "") for item in chunk_content if isinstance(item, dict)])
    elif isinstance(chunk_content, dict) and "text" in chunk_content:
        return chunk_content["text"]
    
    return str(chunk_content) if chunk_content is not None else ""

def _parse_websocket_request(request: str) -> tuple[str, dict, list[str], str]:
    """
    Parses the incoming websocket request.

    The request can be a JSON string with 'prompt' and 'context' keys,
    or a plain text string.

    Args:
        request: The raw request string from the websocket.

    Returns:
        A tuple containing the prompt (str) and the context (dict).
    """
    try:
        json_request = json.loads(request)

        prompt = json_request.get("prompt", "")
        context = json_request.get("context", {})
        tags = json_request.get("tags", [])
        request_id = str(uuid.uuid4())

        return prompt, context, tags, request_id
    except json.JSONDecodeError:
        return request, {}, [], ""

# This will be removed once https://github.com/modelcontextprotocol/python-sdk/pull/1177 is merged
class SimpleTruststore:
    def get_default(self):
        """Get the default Python truststore"""
        return certifi.where()

    def create_combined(self, company_cert_path, output_path):
        """Create truststore with public CAs + company cert"""
        with open(output_path, "w") as combined:
            # Add public CAs 
            with open(certifi.where(), "r") as public_cas:
                combined.write(public_cas.read())

            # Add MCP self-signed cert
            with open(company_cert_path, "r") as company:
                combined.write("\n" + company.read())

        return output_path
    
    def use_truststore(self, truststore_path):
        """Set the global truststore"""
        os.environ["SSL_CERT_FILE"] = truststore_path

    def set_truststore(self):
        company_cert_path = "/etc/tls/tls.crt"
        output_path = "/combined.crt"
        truststore_path = self.create_combined(
            company_cert_path=company_cert_path, output_path=output_path
        )
        self.use_truststore(truststore_path=truststore_path)
