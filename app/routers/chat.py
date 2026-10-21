import logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, JSONResponse

from ..services.auth import get_user_id_from_token

# Token pricing (in USD per 1K tokens) - adjust as needed
TOKEN_PRICING = {
    "input": 0.0005,
    "output": 0.0015,
    "cache_read": 0.00005,  # Usually 90% discount
    "cache_creation": 0.0015,
}

router = APIRouter(prefix="/v1/api", tags=["chats"])

@router.get("/chats")
async def get_chats(request: Request, sort: str = "createdAt:desc"):
    """
    TODO: add filtering by tags from query parameters.
    Get all threads that have at least one user message and return them as chats list.

    Returns:
        A list of chat objects.
    """

    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        chats = await request.app.memory_manager.fetch_chats(
            user_id=user_id
        )

        chats = sorted(chats, key=lambda c: c.get("createdAt") or "", reverse=(sort != "createdAt:asc"))

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=chats
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching chats for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
        
@router.delete("/chats")
async def delete_chats(request: Request) -> JSONResponse:
    """
    Delete all chats for the user.

    Returns:
        A success message and an HTTP status code.
    """

    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        await request.app.memory_manager.delete_chats(
            user_id=user_id
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting chats for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

@router.get("/chats/{chat_id}")
async def get_chat(request: Request, chat_id: str) -> JSONResponse:
    """
    Get a chat by ID.

    Args:
        chat_id: The ID of the thread.

    Returns:
        A chat object and an HTTP status code.
    """

    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        if not chat_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id is required")

        chat = await request.app.memory_manager.fetch_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        if not chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=chat
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching chat for chat_id {chat_id} and user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
        
@router.put("/chats/{chat_id}")
async def update_chat(request: Request, chat_id: str, chat_data: dict) -> JSONResponse:
    """
    Update a chat.
    Args:
        chat_id: The ID of the thread.
        chat_data: The chat data to update.
    Returns:
        The updated chat object and an HTTP status code.
    """

    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        if not chat_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id is required")
        
        if not chat_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_data is required")

        chat = await request.app.memory_manager.fetch_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        if not chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

        updated_chat = await request.app.memory_manager.update_chat(
            chat_id=chat_id,
            user_id=user_id,
            chat_data=chat_data
        )
        
        if not updated_chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Failed to update chat, chat not found")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=updated_chat
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating chat for chat_id {chat_id} and user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

@router.delete("/chats/{chat_id}")
async def delete_chat(request: Request, chat_id: str) -> JSONResponse:
    """
    Delete a specific chat by ID.

    Args:
        chat_id: The ID of the thread.

    Returns:
        A success message and an HTTP status code.
    """

    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        if not chat_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id is required")

        await request.app.memory_manager.delete_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting chat for chat_id {chat_id} and user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

@router.get("/chats/{chat_id}/messages")
async def get_chat_messages(request: Request, chat_id: str) -> JSONResponse:
    """
    TODO: add filtering by tags from query parameters.
    Get messages for a specific thread.

    Args:
        chat_id: The ID of the thread.
    Returns:
        A list of message objects and an HTTP status code.
    """

    try:        
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        if not chat_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id is required")

        chat = await request.app.memory_manager.fetch_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        if not chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

        messages = await request.app.memory_manager.fetch_messages(
            chat_id=chat_id,
            user_id=user_id
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=messages
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching messages for chat_id {chat_id} and user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


# ============= ANALYTICS ENDPOINTS =============

def _calculate_chat_cost(chat: dict) -> float:
    """Calculate cost for a single chat based on token usage."""
    usage = chat.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    cache_read_tokens = usage.get("cacheReadTokens", 0)
    cache_creation_tokens = usage.get("cacheCreationTokens", 0)
    
    cost = (
        (input_tokens / 1000) * TOKEN_PRICING["input"] +
        (output_tokens / 1000) * TOKEN_PRICING["output"] +
        (cache_read_tokens / 1000) * TOKEN_PRICING["cache_read"] +
        (cache_creation_tokens / 1000) * TOKEN_PRICING["cache_creation"]
    )
    return round(cost, 6)


def _parse_period(period_str: str) -> tuple[datetime, datetime]:
    """Parse period string (e.g. '7d', '30d', '1m') and return (start, end) dates."""
    now = datetime.now(timezone.utc)
    
    if period_str.endswith("d"):
        days = int(period_str[:-1])
        start = now - timedelta(days=days)
    elif period_str.endswith("m"):
        months = int(period_str[:-1])
        start = now - timedelta(days=months*30)
    else:
        # Default to 7 days
        start = now - timedelta(days=7)
    
    return start, now


def _chat_in_period(chat: dict, start: datetime, end: datetime) -> bool:
    """Check if chat was created within the period."""
    try:
        created_at = chat.get("createdAt")
        if isinstance(created_at, str):
            # Handle ISO format with or without Z
            created_at_clean = created_at.replace("Z", "+00:00")
            chat_time = datetime.fromisoformat(created_at_clean)
            # Ensure timezone aware
            if chat_time.tzinfo is None:
                chat_time = chat_time.replace(tzinfo=timezone.utc)
        else:
            return False
        return start <= chat_time <= end
    except Exception:
        return False


@router.get("/chats/stats/daily")
async def get_daily_stats(request: Request, days: int = 7) -> JSONResponse:
    """
    Get daily token consumption statistics for the past N days.
    
    Args:
        days: Number of days to look back (default: 7)
    
    Returns:
        Daily breakdown of token usage and costs.
    """
    try:
        user_id = await get_user_id_from_token(request.cookies)
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        chats = await request.app.memory_manager.fetch_chats(user_id=user_id)
        
        daily_stats = defaultdict(lambda: {
            "date": "",
            "totalTokens": 0,
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheReadTokens": 0,
            "cacheCreationTokens": 0,
            "totalCost": 0.0,
            "chatCount": 0,
        })
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        for chat in chats:
            try:
                created_at = chat.get("createdAt")
                if isinstance(created_at, str):
                    created_at_clean = created_at.replace("Z", "+00:00")
                    chat_time = datetime.fromisoformat(created_at_clean)
                    if chat_time.tzinfo is None:
                        chat_time = chat_time.replace(tzinfo=timezone.utc)
                else:
                    continue
                
                if chat_time < cutoff:
                    continue
                
                date_key = chat_time.strftime("%Y-%m-%d")
                usage = chat.get("usage", {})
                cost = _calculate_chat_cost(chat)
                
                daily_stats[date_key]["date"] = date_key
                daily_stats[date_key]["totalTokens"] += usage.get("totalTokens", 0)
                daily_stats[date_key]["inputTokens"] += usage.get("inputTokens", 0)
                daily_stats[date_key]["outputTokens"] += usage.get("outputTokens", 0)
                daily_stats[date_key]["cacheReadTokens"] += usage.get("cacheReadTokens", 0)
                daily_stats[date_key]["cacheCreationTokens"] += usage.get("cacheCreationTokens", 0)
                daily_stats[date_key]["totalCost"] += cost
                daily_stats[date_key]["chatCount"] += 1
            except Exception:
                continue
        
        # Sort by date
        sorted_stats = sorted(daily_stats.values(), key=lambda x: x["date"])
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "period": f"Last {days} days",
                "data": sorted_stats,
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching daily stats for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


@router.get("/chats/stats/agents")
async def get_agent_stats(request: Request) -> JSONResponse:
    """
    Get token consumption statistics aggregated by agent.
    
    Returns:
        Per-agent breakdown of total tokens, input, output, and costs.
    """
    try:
        user_id = await get_user_id_from_token(request.cookies)
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        chats = await request.app.memory_manager.fetch_chats(user_id=user_id)
        
        agent_stats = defaultdict(lambda: {
            "agent": "",
            "totalTokens": 0,
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheReadTokens": 0,
            "cacheCreationTokens": 0,
            "totalCost": 0.0,
            "chatCount": 0,
            "avgTokensPerChat": 0,
        })
        
        for chat in chats:
            usage = chat.get("usage", {})
            by_agent = usage.get("byAgent", {})
            cost = _calculate_chat_cost(chat)
            
            for agent_name, agent_usage in by_agent.items():
                agent_stats[agent_name]["agent"] = agent_name
                agent_stats[agent_name]["totalTokens"] += agent_usage.get("totalTokens", 0)
                agent_stats[agent_name]["inputTokens"] += agent_usage.get("inputTokens", 0)
                agent_stats[agent_name]["outputTokens"] += agent_usage.get("outputTokens", 0)
                agent_stats[agent_name]["cacheReadTokens"] += agent_usage.get("cacheReadTokens", 0)
                agent_stats[agent_name]["cacheCreationTokens"] += agent_usage.get("cacheCreationTokens", 0)
                agent_stats[agent_name]["chatCount"] += 1
                # Distribute cost proportionally
                if usage.get("totalTokens", 0) > 0:
                    agent_cost = cost * (agent_usage.get("totalTokens", 0) / usage.get("totalTokens", 0))
                    agent_stats[agent_name]["totalCost"] += agent_cost
        
        # Calculate averages
        for agent_name in agent_stats:
            if agent_stats[agent_name]["chatCount"] > 0:
                agent_stats[agent_name]["avgTokensPerChat"] = round(
                    agent_stats[agent_name]["totalTokens"] / agent_stats[agent_name]["chatCount"], 2
                )
        
        # Sort by total tokens (descending)
        sorted_stats = sorted(agent_stats.values(), key=lambda x: x["totalTokens"], reverse=True)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=sorted_stats
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching agent stats for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


@router.get("/chats/analytics/costs")
async def get_cost_analytics(request: Request) -> JSONResponse:
    """
    Get cost breakdown by token type (input, output, cache).
    
    Returns:
        Cost analysis with input, output, and cache costs.
    """
    try:
        user_id = await get_user_id_from_token(request.cookies)
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        chats = await request.app.memory_manager.fetch_chats(user_id=user_id)
        
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_tokens = 0
        total_cache_creation_tokens = 0
        
        for chat in chats:
            usage = chat.get("usage", {})
            total_input_tokens += usage.get("inputTokens", 0)
            total_output_tokens += usage.get("outputTokens", 0)
            total_cache_read_tokens += usage.get("cacheReadTokens", 0)
            total_cache_creation_tokens += usage.get("cacheCreationTokens", 0)
        
        input_cost = (total_input_tokens / 1000) * TOKEN_PRICING["input"]
        output_cost = (total_output_tokens / 1000) * TOKEN_PRICING["output"]
        cache_read_cost = (total_cache_read_tokens / 1000) * TOKEN_PRICING["cache_read"]
        cache_creation_cost = (total_cache_creation_tokens / 1000) * TOKEN_PRICING["cache_creation"]
        total_cost = input_cost + output_cost + cache_read_cost + cache_creation_cost
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "breakdown": {
                    "input": {
                        "tokens": total_input_tokens,
                        "cost": round(input_cost, 6),
                        "percentage": round((input_cost / total_cost * 100) if total_cost > 0 else 0, 2),
                    },
                    "output": {
                        "tokens": total_output_tokens,
                        "cost": round(output_cost, 6),
                        "percentage": round((output_cost / total_cost * 100) if total_cost > 0 else 0, 2),
                    },
                    "cacheRead": {
                        "tokens": total_cache_read_tokens,
                        "cost": round(cache_read_cost, 6),
                        "percentage": round((cache_read_cost / total_cost * 100) if total_cost > 0 else 0, 2),
                    },
                    "cacheCreation": {
                        "tokens": total_cache_creation_tokens,
                        "cost": round(cache_creation_cost, 6),
                        "percentage": round((cache_creation_cost / total_cost * 100) if total_cost > 0 else 0, 2),
                    },
                },
                "summary": {
                    "totalTokens": total_input_tokens + total_output_tokens + total_cache_read_tokens + total_cache_creation_tokens,
                    "totalCost": round(total_cost, 6),
                    "chatCount": len(chats),
                    "avgCostPerChat": round(total_cost / len(chats), 6) if chats else 0,
                }
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching cost analytics for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


@router.get("/chats/top/expensive")
async def get_expensive_chats(request: Request, limit: int = 10) -> JSONResponse:
    """
    Get the most expensive chats (by token cost).
    
    Args:
        limit: Number of top chats to return (default: 10)
    
    Returns:
        List of most expensive chats with their costs and token counts.
    """
    try:
        user_id = await get_user_id_from_token(request.cookies)
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        chats = await request.app.memory_manager.fetch_chats(user_id=user_id)
        
        chats_with_costs = []
        for chat in chats:
            cost = _calculate_chat_cost(chat)
            usage = chat.get("usage", {})
            chats_with_costs.append({
                "id": chat.get("id"),
                "createdAt": chat.get("createdAt"),
                "totalTokens": usage.get("totalTokens", 0),
                "inputTokens": usage.get("inputTokens", 0),
                "outputTokens": usage.get("outputTokens", 0),
                "cacheReadTokens": usage.get("cacheReadTokens", 0),
                "cacheCreationTokens": usage.get("cacheCreationTokens", 0),
                "cost": cost,
            })
        
        # Sort by cost (descending)
        sorted_chats = sorted(chats_with_costs, key=lambda x: x["cost"], reverse=True)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=sorted_chats[:limit]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching expensive chats for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


@router.get("/chats/analytics")
async def get_analytics_by_period(request: Request, period: str = "7d") -> JSONResponse:
    """
    Get aggregated analytics for a specific period.
    
    Args:
        period: Time period (e.g. '7d', '30d', '1m')
    
    Returns:
        Aggregated token usage and cost statistics for the period.
    """
    try:
        user_id = await get_user_id_from_token(request.cookies)
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        chats = await request.app.memory_manager.fetch_chats(user_id=user_id)
        start, end = _parse_period(period)
        
        period_chats = [c for c in chats if _chat_in_period(c, start, end)]
        
        total_tokens = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_tokens = 0
        total_cache_creation_tokens = 0
        total_cost = 0.0
        agent_breakdown = defaultdict(lambda: {"totalTokens": 0, "chatCount": 0})
        
        for chat in period_chats:
            usage = chat.get("usage", {})
            cost = _calculate_chat_cost(chat)
            
            total_tokens += usage.get("totalTokens", 0)
            total_input_tokens += usage.get("inputTokens", 0)
            total_output_tokens += usage.get("outputTokens", 0)
            total_cache_read_tokens += usage.get("cacheReadTokens", 0)
            total_cache_creation_tokens += usage.get("cacheCreationTokens", 0)
            total_cost += cost
            
            # Aggregate by agent
            for agent_name, agent_usage in usage.get("byAgent", {}).items():
                agent_breakdown[agent_name]["totalTokens"] += agent_usage.get("totalTokens", 0)
                agent_breakdown[agent_name]["chatCount"] += 1
        
        avg_tokens_per_chat = round(total_tokens / len(period_chats), 2) if period_chats else 0
        cache_hit_rate = round(
            (total_cache_read_tokens / (total_input_tokens + total_cache_read_tokens) * 100) 
            if (total_input_tokens + total_cache_read_tokens) > 0 else 0, 2
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "period": {
                    "name": period,
                    "startDate": start.isoformat(),
                    "endDate": end.isoformat(),
                },
                "summary": {
                    "chatCount": len(period_chats),
                    "totalTokens": total_tokens,
                    "inputTokens": total_input_tokens,
                    "outputTokens": total_output_tokens,
                    "cacheReadTokens": total_cache_read_tokens,
                    "cacheCreationTokens": total_cache_creation_tokens,
                    "totalCost": round(total_cost, 6),
                    "avgTokensPerChat": avg_tokens_per_chat,
                    "cacheHitRate": f"{cache_hit_rate}%",
                },
                "agentBreakdown": dict(agent_breakdown),
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching analytics for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


@router.get("/chats/cost")
async def get_cost_summary(request: Request) -> JSONResponse:
    """
    Get total cost summary for all user chats.
    
    Returns:
        Total cost breakdown and statistics.
    """
    try:
        user_id = await get_user_id_from_token(request.cookies)
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        chats = await request.app.memory_manager.fetch_chats(user_id=user_id)
        
        total_cost = 0.0
        total_tokens = 0
        
        for chat in chats:
            cost = _calculate_chat_cost(chat)
            usage = chat.get("usage", {})
            total_cost += cost
            total_tokens += usage.get("totalTokens", 0)
        
        avg_cost_per_chat = round(total_cost / len(chats), 6) if chats else 0
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "totalCost": round(total_cost, 6),
                "totalTokens": total_tokens,
                "chatCount": len(chats),
                "avgCostPerChat": avg_cost_per_chat,
                "currency": "USD",
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching cost summary for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
