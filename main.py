import logging
import asyncio
import os
import json
import uuid
from enum import Enum

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama 
from langchain_mcp_adapters.tools import load_mcp_tools
from agents import create_k8s_agent, init_rag_rancher, create_memory_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langchain_openai import OpenAI
from langchain_core.language_models.llms import BaseLanguageModel
from contextlib import asynccontextmanager
from langchain_core.tools import create_retriever_tool
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langfuse.langchain import CallbackHandler

class RequestType(Enum):
    AUTOCOMPLETE = "autocomplete"
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
        # if ENABLE_RAG flag is set, initialize the RAG retriever tool
        if os.environ.get("ENABLE_RAG", "false").lower() == "true":
            retriever = init_rag_rancher(get_llm_embeddings())
            init_config["retriever_tool"] = create_retriever_tool(
                retriever,
                "retrieve_rancher_docs",
                "Search and return relevant passages from local Rancher/SUSE documentation. Always use the retrieve_rancher_docs tool when relevant to fetch up-to-date Rancher documentation.",
            )

        """
        Maps user IDs to their active autocomplete tasks.
        
        active_autocomplete_tasks: dict[str, asyncio.Task]
        """
        app.active_autocomplete_tasks = {}

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
    elif not await app.mem_agent.check_chat_permissions(chat_id, user_id):
        logging.warning(f"Permission denied for user {user_id} on chat {chat_id}")
        await websocket.send_text(f'<error>{{"message": "Permission denied for chat {chat_id}"}}</error>')
        await websocket.close()
        return
        
    connection_params = get_ws_connection_params(websocket)

    logging.info(f"ws/messages connection opened - chat_id={chat_id}")

    async with streamablehttp_client(**connection_params) as (read, write, _):
        # This will create one mcp connection for each websocket connection. This is needed because we need to pass the rancher token in the header.
        async with ClientSession(read, write) as session:
            await session.initialize()
            thread_id = str(uuid.uuid4())
            tools = await load_mcp_tools(session)
            
            # if ENABLE_RAG is true, add the retriever tool to the tools list
            if os.environ.get("ENABLE_RAG", "false").lower() == "true":
                tools = [init_config["retriever_tool"]] + tools

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

                    prompt, context, chat_payload, wildcard, request_id = _parse_websocket_request(request)

                    await app.mem_agent.store_chunk(chat_id=chat_id, request_id=request_id, text=prompt, role="user")

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

@app.websocket("/agent/ws/autocomplete")
async def websocket_autocomplete_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for the autocomplete messages.
    
    Accepts a WebSocket connection, sets up the agent and
    handles the back-and-forth communication with the client.
    """
    await websocket.accept()
    
    user_id = await get_user_id(websocket)

    connection_params = get_ws_connection_params(websocket)

    logging.info(f"ws/autocomplete connection opened")

    async with streamablehttp_client(**connection_params):
        thread_id = str(uuid.uuid4())

        tools = []
        
        # if ENABLE_RAG is true, add the retriever tool to the tools list
        if os.environ.get("ENABLE_RAG", "false").lower() == "true":
            tools = [init_config["retriever_tool"]]

        agent = create_k8s_agent(init_config["llm"], tools, get_system_prompt(RequestType.AUTOCOMPLETE), InMemorySaver())
        
        config = {
            "thread_id": thread_id,
        }
        if os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_HOST"):
            langfuse_handler = CallbackHandler()
            config["callbacks"] = [langfuse_handler]

        while True:
            try:
                request = await websocket.receive_text()

                prompt, context, chat_payload, wildcard, request_id = _parse_websocket_request(request)

                prompt = f"User unfinished input: '{prompt}'"

                if context:
                    context_prompt = "\n\nContext parameters: use the following parameters as source to create completions. Parameters (separated by ;): \n "
                    for key, value in context.items():
                        context_prompt += f"{key}:{value};"
                    prompt += context_prompt

                if len(chat_payload) > 0:
                    latestMessages = ""
                    for msg in chat_payload:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        latestMessages += f"\n [{role}]: '{content}'"

                    prompt += f"\n\nRecent messages: use the following recent messages as source to create completions: \n---------- {latestMessages} \n----------\n"
                
                prompt += f"\n\nWildcard: {wildcard}\n"
                
                logging.debug(f"Autocomplete prompt: {prompt}")

                # Cancel any previous running autocomplete task for this user
                prev_entry = app.active_autocomplete_tasks.get(user_id)
                # Normalize prev task and finished_event if stored as dict or bare task
                prev_task = None
                prev_finished = None
                if isinstance(prev_entry, dict):
                    prev_task = prev_entry.get("task")
                    prev_finished = prev_entry.get("finished_event")
                    prev_renew = prev_entry.get("renew")
                else:
                    prev_task = prev_entry
                    prev_renew = None

                if prev_task and not prev_task.done():
                    prev_task.cancel()
                if prev_renew and not prev_renew.done():
                    prev_renew.cancel()

                # If a previous finished_event exists, wait for it so ordering is preserved
                if prev_finished:
                    try:
                        await prev_finished.wait()
                    except Exception:
                        pass

                # Create a finished_event for this stream so subsequent requests
                # can wait until this stream emits its closing tag.
                finished_event = asyncio.Event()

                # Send the opening tag from the request handler to preserve ordering
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text("<message>")
                except Exception:
                    pass

                # Start the stream; the stream will NOT send the opening tag itself
                task = asyncio.create_task(stream_autocomplete_agent_response(
                    agent=agent,
                    input_data={"messages": [{"role": "user", "content": prompt}]},
                    config=config,
                    websocket=websocket,
                    finished_event=finished_event,
                    send_opening_tag=False,
                ))

                app.active_autocomplete_tasks[user_id] = {"task": task, "renew": None, "finished_event": finished_event}
            except WebSocketDisconnect:
                logging.info(f"Client {websocket.client.host} disconnected.")
                break
            except Exception as e:
                logging.error(f"An error occurred on autocomplete request: {e}")
                pass

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

                prompt, context, chat_payload, wildcard, request_id = _parse_websocket_request(request)

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
                # store recent agent replies
                await app.mem_agent.store_chunk(chat_id, request_id, text=text, role="agent")

        if event == "updates":
            if interrupt_value := data.get("__interrupt__"):
                await websocket.send_text(interrupt_value[0].value)
                # Receive user response for the human verification
                user_response = await websocket.receive_text()
                await stream_messages_agent_response(
                    agent=agent,
                    input_data=Command(resume={"response": user_response}),
                    config=config,
                    websocket=websocket)
                
        if event == "custom":
            await websocket.send_text(data)
            # store recent mcp replies
            await app.mem_agent.store_chunk(chat_id, request_id, text=data, role="mcp")

async def stream_autocomplete_agent_response(
    agent: CompiledStateGraph,
    input_data: dict[str, list[dict[str, str]]],
    config: dict,
    websocket: WebSocket,
    finished_event: asyncio.Event | None = None,
    send_opening_tag: bool = True,
) -> None:
    """
    Streams the agent's autocomplete response to a WebSocket connection, handling cancellations.
    
    Args:
        agent: The compiled LangGraph agent.
        input_data: The input data for the agent's run.
        config: The run configuration.
        websocket: The WebSocket connection.
        stream_mode: The types of events to stream from the agent.
    """

    # Optionally send opening tag; caller may send it to preserve ordering.
    if send_opening_tag:
        await websocket.send_text("<message>")
    try:
        async for event, data in agent.astream(
            input_data,
            config=config,
            stream_mode=["messages"],
        ):
            if event == "messages":
                chunk, metadata = data
                if metadata.get("langgraph_node") == "agent" and chunk.content:
                    text = _extract_text_from_chunk_content(chunk.content)
                    await websocket.send_text(text)

    except asyncio.CancelledError:
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text("</message>")
        except Exception:
            pass
        raise
    finally:
        # Ensure closing tag
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text("</message>")
        except Exception:
            pass
        # Notify waiter that stream finished
        try:
            if finished_event:
                finished_event.set()
        except Exception:
            pass

def get_ws_connection_params(websocket: WebSocket) -> dict:
    cookies = websocket.cookies

    rancher_url = "https://"+websocket.url.hostname
    if websocket.url.port:
        rancher_url += ":"+str(websocket.url.port)

    rancher_token = str(cookies.get("R_SESS"))

    return {
        "url": "http://rancher-mcp-server",
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

def get_llm() -> BaseLanguageModel:
    """
    Selects and returns a language model instance based on environment variables.
    
    Returns:
        An instance of a language model.
        
    Raises:
        ValueError: If no supported model or API key is configured.
    """

    model = os.environ.get("MODEL")
    if not model:
        raise ValueError("LLM Model not configured.")
    
    active = os.environ.get("ACTIVE_CHATBOT", "")
    ollama_url = os.environ.get("OLLAMA_URL")
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if active == "ollama":
        return ChatOllama(model=model, base_url=ollama_url)
    if active == "gemini":
        return ChatGoogleGenerativeAI(model=model)
    if active == "openai":
        return OpenAI(model=model)

    # default order if active is not specified
    if ollama_url:
        return ChatOllama(model=model, base_url=ollama_url)
    if gemini_key:
        return ChatGoogleGenerativeAI(model=model)
    if openai_key:
        return OpenAI(model=model)

    raise ValueError("LLM not configured.")

def get_llm_embeddings() -> Embeddings:
    """
    Selects and returns an embedding model instance based on environment variables.

    Returns:
        An instance of a LangChain embedding model that implements the Embeddings interface.

    Raises:
        ValueError: If a required environment variable (like EMBEDDING_MODEL for Ollama) is missing,
                    or if no supported embedding provider is configured at all.
    """

     # Provider 1: Ollama
    ollama_url = os.environ.get("OLLAMA_URL")
    embedding_model_name = os.environ.get("EMBEDDINGS_MODEL")
    if not embedding_model_name:
            raise ValueError("EMBEDDINGS_MODEL must be set.")
    if ollama_url:
        return OllamaEmbeddings(model=embedding_model_name, base_url=ollama_url)

    # Provider 2: Google Gemini
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        return GoogleGenerativeAIEmbeddings(model=embedding_model_name)

    # Provider 3: OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
            return OpenAIEmbeddings(model=embedding_model_name)

    raise ValueError("No embedding provider configured. Set OLLAMA_URL, GOOGLE_API_KEY, or OPENAI_API_KEY.")

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
        case RequestType.AUTOCOMPLETE:
            return """
The autocomplete request contains 4 parts:
    1. User unfinished input: The partial input provided by the user that needs to be completed.
    2. Context parameters: Key-value pairs that provide context about the current environment (e.g., cluster, namespace, resources).
    3. Recent messages: A list of recent messages exchanged in the conversation
    4. Wildcard: A special character or string that may influence the completion behavior.

IF (the Wildcard field is equal to '@') THEN
    Output ONLY a list of 1 up to 20 <item></item> tags containing resource names or identifiers, relevant to the User unfinished input, that are included in the context or recent messages (cluster, namespace, resources). Your response will be used to populate an autocompletion dropdown in the Chat prompt.

    ## CORE DIRECTIVES

    ### Context Awareness
    * Always consider the user's current context (User unfinished input, Context parameters, Recent messages) when defined (clusters, namespaces, or resources) to build the list.
      For example:
        * User unfinished input: "Give me the logs of rancher-", Recent messages or Context parameters contain pod "rancher-xyz"
            * Good item: <item>{"name": "rancher-xyz", "type": "pod"}</item>
            * Bad item: <item>{"name": "rancher-abc", "type": "pod"}</item> - missing "rancher-abc" from Recent messages or Context parameters.
            * Bad item: <item>{"name": "pod-123", "type": "pod"}</item> - missing "rancher-" prefix from User unfinished input.
    
    ### User perspective
    * Remember to keep the user's intent in mind when generating the list. DO NOT return suggestions that are from the Agent side of context.
    * DO NOT return your thinking or any other text outside the <item> tags.

    ### <item></item> tags
    * Each <item> tag must contain a single resource name or identifier that are included in autocompletion request (Context parameters or Recent messages).
    * The <item> payload should be:
    {
        "name": "resource-name",
        "type": "resource-type" // e.g., pod, deployment, service, namespace, cluster
    }
    The list of items can be, for instance:
        <item>{"name": "pod-1", "type": "pod"}</item><item>{"name": "pod-2", "type": "pod"}</item><item>{"name": "namespace-1", "type": "namespace"}</item>

ELSE
    Complete the User unfinished input, naturally and concisely. Do not add any introductory text, explanations, wildcards like '@' or '#', or formatting. Only output the direct continuation of the User's text. Your response will be concatenated with the User unfinished input in the Chat prompt, so that it forms a complete request to be sent by the user to the AI agent later.

    ## CORE DIRECTIVES

    ### Syntax
    * Keep in mind space and punctuation from the User unfinished input to ensure a seamless continuation.
    For example:
        * User unfinished input: "Give me the logs of"
            * Good completion: " pod-{some-example-id} in namespace {some-example-namespace}"
            * Bad completion: "pod-{some-example-id} in namespace {some-example-namespace}" - missing space at the beginning.

    ### Context Awareness
    * Always consider the user's current context (User unfinished input, Context parameters) when defined (clusters, namespaces, or resources) to build completions.
    * Use also the provided Recent messages from the conversation to build your completions. First messages are most relevant.
    * Always consider the User unfinished input to build completions.
        * Good: If the User unfinished input mentioned "pod-{some-example-id}", use that in the completion and build around it, for example like " in namespace {some-example-namespace}".
        * Bad: Ignore the User unfinished input and start the completion unrelated to them, for example if the User unfinished input mentioned "pod-{some-example-id}" but you respond with "What can I do for you today?".

    ### User perspective
    * Remember to keep the user's intent in mind when generating completions. DO NOT return suggestions that are from the Agent side of context.
    For example:
        * User unfinished input: "Give me the logs for the failing p"
            * Good completion: "od-{some-example-id} in local cluster" - this is an acceptable completion because it addresses the User's intent.
            * Bad completion: "od. Sure, I can help with that." - this is not an acceptable completion because it does not continue the User unfinished input. It contains instead a response from the Agent side of context.
        * User unfinished input: "How can I"
            * Good completion: " check the pod-{some-example-id} status?" - this is an acceptable completion because it continues the User unfinished input.
            * Bad completion: " help you?" - this is not an acceptable completion because it's a question that is from Agent side of context.
        * User unfinished input: "What type of resou"
            * Good completion: "rce are is {some-example-id}?" - this is an acceptable completion because it continues the User unfinished input.
            * Bad completion: "rces are you interested in?" - this is not an acceptable completion because it's a question that is from Agent side of context.

    ### Consistency
    * If the User unfinished input is already a complete phrase or question, do not provide a suggestion, return empty string.
    For example:
        * User input: "Show me the logs for pod-{some-example-id}"
            * Good completion: ""
            * Bad completion: ". What else can I do for you? This is not a completion, it's an answer to a user's question that is completely out of context.
        * User input: "How do I create a deployment?"
            * Good completion: ""
            * Bad completion: "How do I create a deployment?" - Would you like to know more about deployments?
            * Bad completion: "You can create a deployment by..." - this is not a completion, it's an answer to the user's question.
    * If the User unfinished input needs a completion in the middle of a sentence, provide the completion only for the missing part.
    For example:
        * User unfinished input: "Show me the logs for p in namespace fleet-local"
            * Good completion: "od-{some-example-id}"
            * Bad completion: "pod-{some-example-id} in namespace fleet-local" - this repeats the prompt already provided by the user.

    ### Natural language Mentality
    * The completions should be in natural language, as the user would express it.
"""
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

def _parse_websocket_request(request: str) -> tuple[str, dict, list[str], str, str]:
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
        chat_payload = json_request.get("chatPayload", [])
        wildcard = json_request.get("wildcard", "")
        request_id = str(uuid.uuid4())

        return prompt, context, chat_payload, wildcard, request_id
    except json.JSONDecodeError:
        return request, {}, [], "", None
