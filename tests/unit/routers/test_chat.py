import pytest
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import status, HTTPException
from app.routers import chat as chat_router

@pytest.fixture
def mock_request():
    req = MagicMock()
    req.app.memory_manager = AsyncMock()
    req.cookies = {"R_SESS": "token"}
    req.headers = {"Host": "localhost"}
    return req

@pytest.mark.asyncio
async def test_get_chats_success(mock_request):
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {"id": "1", "userId": "u", "createdAt": 2},
        {"id": "2", "userId": "u", "createdAt": 1},
    ]
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        resp = await chat_router.get_chats(mock_request)
        assert resp.status_code == status.HTTP_200_OK
        assert resp.body

@pytest.mark.asyncio
async def test_get_chats_sort_asc(mock_request):
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {"id": "1", "userId": "u", "createdAt": 2},
        {"id": "2", "userId": "u", "createdAt": 1},
    ]
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        resp = await chat_router.get_chats(mock_request, sort="createdAt:asc")
        result = json.loads(resp.body)
        assert result[0]["createdAt"] == 1
        assert result[1]["createdAt"] == 2

@pytest.mark.asyncio
async def test_get_chat_not_found(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        with pytest.raises(HTTPException) as exc:
            await chat_router.get_chat(mock_request, chat_id="notfound")
        assert exc.value.status_code == status.HTTP_404_NOT_FOUND

@pytest.mark.asyncio
async def test_update_chat_success(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = {"id": "1", "userId": "u"}
    mock_request.app.memory_manager.update_chat.return_value = {"id": "1", "userId": "u", "name": "updated"}
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        resp = await chat_router.update_chat(mock_request, chat_id="1", chat_data={"name": "updated"})
        assert resp.status_code == status.HTTP_200_OK
        assert b"updated" in resp.body


@pytest.mark.asyncio
async def test_update_chat_not_found(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        with pytest.raises(HTTPException) as exc:
            await chat_router.update_chat(mock_request, chat_id="notfound", chat_data={"name": "updated"})
        assert exc.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_update_chat_update_failure(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = {"id": "1", "userId": "u"}
    mock_request.app.memory_manager.update_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        with pytest.raises(HTTPException) as exc:
            await chat_router.update_chat(mock_request, chat_id="1", chat_data={"name": "fail"})
        assert exc.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_update_chat_invalid_data(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = {"id": "1", "userId": "u"}
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        with pytest.raises(Exception):
            await chat_router.update_chat(mock_request, chat_id="1", chat_data=None)

@pytest.mark.asyncio
async def test_delete_chat_success(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = {"id": "1", "userId": "u"}
    mock_request.app.memory_manager.delete_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        resp = await chat_router.delete_chat(mock_request, chat_id="1")
        assert resp.status_code == status.HTTP_204_NO_CONTENT

@pytest.mark.asyncio
async def test_get_chat_messages_not_found(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="u")):
        with pytest.raises(HTTPException) as exc:
            await chat_router.get_chat_messages(mock_request, chat_id="notfound")
        assert exc.value.status_code == status.HTTP_404_NOT_FOUND


# Token usage and counter tests
@pytest.mark.asyncio
async def test_get_chat_includes_token_usage(mock_request):
    """Verify that get_chat endpoint returns token usage information."""
    mock_request.app.memory_manager.fetch_chat.return_value = {
        "id": "chat1",
        "userId": "user1",
        "createdAt": "2024-01-01T00:00:00Z",
        "usage": {
            "totalTokens": 150,
            "inputTokens": 100,
            "outputTokens": 50,
            "cacheReadTokens": 0,
            "cacheCreationTokens": 0,
            "byAgent": {
                "supervisor": {
                    "totalTokens": 100,
                    "inputTokens": 70,
                    "outputTokens": 30,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                },
                "kubernetes": {
                    "totalTokens": 50,
                    "inputTokens": 30,
                    "outputTokens": 20,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                },
            },
        },
    }
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="user1")):
        resp = await chat_router.get_chat(mock_request, chat_id="chat1")
        assert resp.status_code == status.HTTP_200_OK
        result = json.loads(resp.body)
        assert "usage" in result
        assert result["usage"]["totalTokens"] == 150
        assert result["usage"]["inputTokens"] == 100
        assert result["usage"]["outputTokens"] == 50


@pytest.mark.asyncio
async def test_get_chat_token_usage_structure(mock_request):
    """Verify that token usage has the correct structure."""
    mock_request.app.memory_manager.fetch_chat.return_value = {
        "id": "chat1",
        "userId": "user1",
        "usage": {
            "totalTokens": 100,
            "inputTokens": 60,
            "outputTokens": 40,
            "cacheReadTokens": 5,
            "cacheCreationTokens": 10,
            "byAgent": {
                "agent1": {
                    "totalTokens": 100,
                    "inputTokens": 60,
                    "outputTokens": 40,
                    "cacheReadTokens": 5,
                    "cacheCreationTokens": 10,
                },
            },
        },
    }
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="user1")):
        resp = await chat_router.get_chat(mock_request, chat_id="chat1")
        result = json.loads(resp.body)
        usage = result["usage"]
        
        # Verify top-level fields
        assert usage["totalTokens"] == 100
        assert usage["inputTokens"] == 60
        assert usage["outputTokens"] == 40
        assert usage["cacheReadTokens"] == 5
        assert usage["cacheCreationTokens"] == 10
        
        # Verify byAgent breakdown
        assert "byAgent" in usage
        assert "agent1" in usage["byAgent"]
        agent_usage = usage["byAgent"]["agent1"]
        assert agent_usage["totalTokens"] == 100
        assert agent_usage["inputTokens"] == 60
        assert agent_usage["outputTokens"] == 40


@pytest.mark.asyncio
async def test_get_chat_zero_token_usage(mock_request):
    """Verify that zero token usage is returned when no tokens are tracked."""
    mock_request.app.memory_manager.fetch_chat.return_value = {
        "id": "chat1",
        "userId": "user1",
        "usage": {
            "totalTokens": 0,
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheReadTokens": 0,
            "cacheCreationTokens": 0,
            "byAgent": {},
        },
    }
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="user1")):
        resp = await chat_router.get_chat(mock_request, chat_id="chat1")
        result = json.loads(resp.body)
        assert result["usage"]["totalTokens"] == 0
        assert result["usage"]["byAgent"] == {}


@pytest.mark.asyncio
async def test_get_chats_includes_token_usage_per_chat(mock_request):
    """Verify that get_chats includes token usage for each chat."""
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {
            "id": "chat1",
            "userId": "user1",
            "createdAt": 2,
            "usage": {
                "totalTokens": 200,
                "inputTokens": 120,
                "outputTokens": 80,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {"supervisor": {"totalTokens": 200, "inputTokens": 120, "outputTokens": 80, "cacheReadTokens": 0, "cacheCreationTokens": 0}},
            },
        },
        {
            "id": "chat2",
            "userId": "user1",
            "createdAt": 1,
            "usage": {
                "totalTokens": 100,
                "inputTokens": 60,
                "outputTokens": 40,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {"supervisor": {"totalTokens": 100, "inputTokens": 60, "outputTokens": 40, "cacheReadTokens": 0, "cacheCreationTokens": 0}},
            },
        },
    ]
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="user1")):
        resp = await chat_router.get_chats(mock_request)
        assert resp.status_code == status.HTTP_200_OK
        result = json.loads(resp.body)
        assert len(result) == 2
        # Verify both chats have usage
        assert "usage" in result[0]
        assert "usage" in result[1]
        assert result[0]["usage"]["totalTokens"] == 200
        assert result[1]["usage"]["totalTokens"] == 100


@pytest.mark.asyncio
async def test_get_chats_token_usage_sorted_correctly(mock_request):
    """Verify that chats are sorted correctly and usage is preserved."""
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {
            "id": "chat1",
            "userId": "user1",
            "createdAt": 2,
            "usage": {"totalTokens": 50, "inputTokens": 30, "outputTokens": 20, "cacheReadTokens": 0, "cacheCreationTokens": 0, "byAgent": {}},
        },
        {
            "id": "chat2",
            "userId": "user1",
            "createdAt": 1,
            "usage": {"totalTokens": 100, "inputTokens": 60, "outputTokens": 40, "cacheReadTokens": 0, "cacheCreationTokens": 0, "byAgent": {}},
        },
    ]
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="user1")):
        # Test descending sort (default)
        resp = await chat_router.get_chats(mock_request, sort="createdAt:desc")
        result = json.loads(resp.body)
        assert result[0]["createdAt"] == 2
        assert result[0]["usage"]["totalTokens"] == 50
        assert result[1]["createdAt"] == 1
        assert result[1]["usage"]["totalTokens"] == 100


@pytest.mark.asyncio
async def test_get_chat_with_multiple_agents_token_usage(mock_request):
    """Verify token usage breakdown across multiple agents."""
    mock_request.app.memory_manager.fetch_chat.return_value = {
        "id": "chat1",
        "userId": "user1",
        "usage": {
            "totalTokens": 500,
            "inputTokens": 300,
            "outputTokens": 200,
            "cacheReadTokens": 10,
            "cacheCreationTokens": 5,
            "byAgent": {
                "supervisor": {
                    "totalTokens": 200,
                    "inputTokens": 120,
                    "outputTokens": 80,
                    "cacheReadTokens": 5,
                    "cacheCreationTokens": 2,
                },
                "kubernetes": {
                    "totalTokens": 150,
                    "inputTokens": 90,
                    "outputTokens": 60,
                    "cacheReadTokens": 3,
                    "cacheCreationTokens": 1,
                },
                "rancher": {
                    "totalTokens": 150,
                    "inputTokens": 90,
                    "outputTokens": 60,
                    "cacheReadTokens": 2,
                    "cacheCreationTokens": 2,
                },
            },
        },
    }
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="user1")):
        resp = await chat_router.get_chat(mock_request, chat_id="chat1")
        result = json.loads(resp.body)
        usage = result["usage"]
        
        # Verify grand totals sum correctly
        agent_totals = sum(agent["totalTokens"] for agent in usage["byAgent"].values())
        assert agent_totals == usage["totalTokens"]
        
        # Verify individual agent data
        assert usage["byAgent"]["supervisor"]["totalTokens"] == 200
        assert usage["byAgent"]["kubernetes"]["totalTokens"] == 150
        assert usage["byAgent"]["rancher"]["totalTokens"] == 150
        
        # Verify cache tokens are tracked per agent
        assert usage["byAgent"]["supervisor"]["cacheReadTokens"] == 5
        assert usage["byAgent"]["kubernetes"]["cacheReadTokens"] == 3


# Analytics Endpoints Tests
@pytest.mark.asyncio
async def test_get_daily_stats(mock_request):
    """Test daily token consumption statistics."""
    now = datetime.now(timezone.utc)
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {
            "id": "chat1",
            "userId": "user1",
            "createdAt": now.isoformat(),
            "usage": {
                "totalTokens": 100,
                "inputTokens": 60,
                "outputTokens": 40,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {},
            },
        },
        {
            "id": "chat2",
            "userId": "user1",
            "createdAt": (now - timedelta(days=1)).isoformat(),
            "usage": {
                "totalTokens": 150,
                "inputTokens": 90,
                "outputTokens": 60,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {},
            },
        },
    ]
    
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="user1")):
        resp = await chat_router.get_daily_stats(mock_request, days=7)
        assert resp.status_code == status.HTTP_200_OK
        result = json.loads(resp.body)
        
        assert "data" in result
        assert len(result["data"]) == 2
        assert result["data"][0]["totalTokens"] == 150  # Day 1 (oldest first after sorting)
        assert result["data"][1]["totalTokens"] == 100  # Today


@pytest.mark.asyncio
async def test_get_agent_stats(mock_request):
    """Test agent-level statistics aggregation."""
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {
            "id": "chat1",
            "userId": "user1",
            "usage": {
                "totalTokens": 500,
                "inputTokens": 300,
                "outputTokens": 200,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {
                    "supervisor": {
                        "totalTokens": 250,
                        "inputTokens": 150,
                        "outputTokens": 100,
                        "cacheReadTokens": 0,
                        "cacheCreationTokens": 0,
                    },
                    "kubernetes": {
                        "totalTokens": 250,
                        "inputTokens": 150,
                        "outputTokens": 100,
                        "cacheReadTokens": 0,
                        "cacheCreationTokens": 0,
                    },
                },
            },
        },
        {
            "id": "chat2",
            "userId": "user1",
            "usage": {
                "totalTokens": 200,
                "inputTokens": 120,
                "outputTokens": 80,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {
                    "supervisor": {
                        "totalTokens": 200,
                        "inputTokens": 120,
                        "outputTokens": 80,
                        "cacheReadTokens": 0,
                        "cacheCreationTokens": 0,
                    },
                },
            },
        },
    ]
    
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="user1")):
        resp = await chat_router.get_agent_stats(mock_request)
        assert resp.status_code == status.HTTP_200_OK
        result = json.loads(resp.body)
        
        # Should be sorted by total tokens descending
        assert result[0]["agent"] == "supervisor"
        assert result[0]["totalTokens"] == 450
        assert result[0]["chatCount"] == 2
        assert result[0]["avgTokensPerChat"] == 225.0
        
        assert result[1]["agent"] == "kubernetes"
        assert result[1]["totalTokens"] == 250
        assert result[1]["chatCount"] == 1


@pytest.mark.asyncio
async def test_get_cost_analytics(mock_request):
    """Test cost breakdown by token type."""
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {
            "id": "chat1",
            "userId": "user1",
            "usage": {
                "totalTokens": 1000,
                "inputTokens": 600,
                "outputTokens": 400,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {},
            },
        },
    ]
    
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="user1")):
        resp = await chat_router.get_cost_analytics(mock_request)
        assert resp.status_code == status.HTTP_200_OK
        result = json.loads(resp.body)
        
        assert "breakdown" in result
        assert "summary" in result
        assert result["breakdown"]["input"]["tokens"] == 600
        assert result["breakdown"]["output"]["tokens"] == 400
        assert result["summary"]["totalTokens"] == 1000
        assert result["summary"]["chatCount"] == 1
        
        # Verify cost calculation (600*0.0005 + 400*0.0015 = 0.3 + 0.6 = 0.9)
        assert result["breakdown"]["input"]["cost"] > 0
        assert result["breakdown"]["output"]["cost"] > 0


@pytest.mark.asyncio
async def test_get_expensive_chats(mock_request):
    """Test retrieval of most expensive chats."""
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {
            "id": "expensive-chat",
            "userId": "user1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "usage": {
                "totalTokens": 5000,
                "inputTokens": 3000,
                "outputTokens": 2000,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {},
            },
        },
        {
            "id": "cheap-chat",
            "userId": "user1",
            "createdAt": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "usage": {
                "totalTokens": 500,
                "inputTokens": 300,
                "outputTokens": 200,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {},
            },
        },
    ]
    
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="user1")):
        resp = await chat_router.get_expensive_chats(mock_request, limit=10)
        assert resp.status_code == status.HTTP_200_OK
        result = json.loads(resp.body)
        
        # Should be sorted by cost descending
        assert result[0]["id"] == "expensive-chat"
        assert result[0]["totalTokens"] == 5000
        assert result[1]["id"] == "cheap-chat"
        assert result[1]["totalTokens"] == 500


@pytest.mark.asyncio
async def test_get_analytics_by_period(mock_request):
    """Test period-based analytics aggregation."""
    now = datetime.now(timezone.utc)
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {
            "id": "chat1",
            "userId": "user1",
            "createdAt": now.isoformat(),
            "usage": {
                "totalTokens": 1000,
                "inputTokens": 600,
                "outputTokens": 400,
                "cacheReadTokens": 50,
                "cacheCreationTokens": 10,
                "byAgent": {
                    "supervisor": {
                        "totalTokens": 1000,
                        "inputTokens": 600,
                        "outputTokens": 400,
                        "cacheReadTokens": 50,
                        "cacheCreationTokens": 10,
                    },
                },
            },
        },
        {
            "id": "chat2",
            "userId": "user1",
            "createdAt": (now - timedelta(days=10)).isoformat(),  # Outside 7d window
            "usage": {
                "totalTokens": 500,
                "inputTokens": 300,
                "outputTokens": 200,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {},
            },
        },
    ]
    
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="user1")):
        resp = await chat_router.get_analytics_by_period(mock_request, period="7d")
        assert resp.status_code == status.HTTP_200_OK
        result = json.loads(resp.body)
        
        assert result["period"]["name"] == "7d"
        assert result["summary"]["chatCount"] == 1  # Only recent chat
        assert result["summary"]["totalTokens"] == 1000
        assert result["summary"]["inputTokens"] == 600
        assert result["summary"]["outputTokens"] == 400
        assert result["summary"]["cacheReadTokens"] == 50


@pytest.mark.asyncio
async def test_get_cost_summary(mock_request):
    """Test total cost summary across all chats."""
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {
            "id": "chat1",
            "userId": "user1",
            "usage": {
                "totalTokens": 1000,
                "inputTokens": 600,
                "outputTokens": 400,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {},
            },
        },
        {
            "id": "chat2",
            "userId": "user1",
            "usage": {
                "totalTokens": 2000,
                "inputTokens": 1200,
                "outputTokens": 800,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {},
            },
        },
    ]
    
    with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="user1")):
        resp = await chat_router.get_cost_summary(mock_request)
        assert resp.status_code == status.HTTP_200_OK
        result = json.loads(resp.body)
        
        assert result["totalTokens"] == 3000
        assert result["chatCount"] == 2
        assert result["avgCostPerChat"] > 0
        assert result["currency"] == "USD"
