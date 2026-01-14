import logging
from fastapi import APIRouter, Request
from typing import List, Dict, Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from ..services.agent.agent import create_rest_api_agent

router = APIRouter(prefix="/api", tags=["chats"])

@router.get("/chats")
async def get_chats(request: Request) -> List[Dict[str, Any]]:
    """
    Get all threads that have at least one user message and return them as chats list.

    Returns:
        A list of chat objects.
    """

    async with AsyncPostgresSaver.from_conn_string(request.app.db_manager.db_url) as checkpointer:
        threads = []
        async for checkpoint in checkpointer.alist(config=None, filter={"user_id": "admin"}):
            logging.debug(f"Found chat thread: {checkpoint.config["configurable"]["thread_id"]}")
            
            threads.append({
                "thread_id": checkpoint.config["configurable"]["thread_id"],
                "user_id": checkpoint.metadata["user_id"],
            })

        return threads
    
@router.get("/chats/{chat_id}/messages")
async def get_chat_messages(request: Request, chat_id: str) -> List[Dict[str, Any]]:
    """
    Get messages for a specific thread.

    Args:
        chat_id: The ID of the thread.
    Returns:
        A list of message objects.
    """
    
    async with AsyncPostgresSaver.from_conn_string(request.app.db_manager.db_url) as checkpointer:
        rows = []

        # Create agent with the checkpointer to access state
        agent = create_rest_api_agent(checkpointer)
        
        # Filter by chat_id
        config = {"configurable": {"thread_id": chat_id}}
        
        # Collect states grouped by request_id in reverse order
        states_list = []
        async for state in agent.aget_state_history(config):
            states_list.append(state)

        # Group states by request_id
        states_dict = {}
        for state in reversed(states_list):
            if state and state.values and state.metadata:
                state_request_id = state.metadata.get("request_id")
                if state_request_id:
                    if state_request_id not in states_dict:
                        states_dict[state_request_id] = []
                    states_dict[state_request_id].append(state)
        
        # Process states for each request_id
        processed_message_ids = []
        for request_id, states in states_dict.items():
            
            logging.debug(f"Processing state for chat_id: {chat_id}, request_id: {request_id}")
            
            user_row = None
            agent_row = None
            
            mcp_str = ""
            llm_str = ""

            for state in states:
                agent_metadata = state.values.get("agent_metadata", {})
                context = agent_metadata.get("context", {})
                tags = agent_metadata.get("tags", [])
                mcp_responses = agent_metadata.get("mcp_responses", [])
                mcp_resp_str = "".join(mcp_responses) if mcp_responses else ""

                # Filter out already processed messages
                messages = [m for m in state.values.get("messages", []) if hasattr(m, "id") and m.id not in processed_message_ids]

                for msg in messages:
                    if msg.type == 'human':
                        if user_row is None:
                            text = agent_metadata.get("prompt", "")
                            user_row = {
                                "chatId": chat_id,
                                "requestId": request_id,
                                "role": "user",
                                "message": text if text else "",
                                "context": context,
                                "tags": tags,
                                "createdAt": msg.additional_kwargs.get("created_at"),
                            }

                    if msg.type == 'ai':
                        # Always concatenate MCP responses to agent message
                        if mcp_str == "":
                            mcp_str = mcp_resp_str
                        if llm_str == "":
                            llm_str = msg.content if msg.content else ""
                            
                        text = (mcp_str + llm_str) if (mcp_str or llm_str) else agent_row["message"] if agent_row else ""
                        if text:
                            agent_row = {
                                "chatId": chat_id,
                                "requestId": request_id,
                                "role": "agent",
                                "message": text,
                                "context": None,
                                "tags": tags,
                                "createdAt": msg.additional_kwargs.get("created_at"), # Always the date from latest Agent node
                            }
                    processed_message_ids.append(msg.id)
            if user_row:
                rows.append(user_row)
            if agent_row:
                rows.append(agent_row)

    return rows
