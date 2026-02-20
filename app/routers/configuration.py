import logging
import os
import base64
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
from kubernetes import client, config as k8s_config
from kubernetes.client.rest import ApiException

from app.routers.chat import get_user_id_from_request

from ..services.llm import LLMManager
from ..services.auth import get_user_id

router = APIRouter(prefix="/v1/api", tags=["configuration"])

# Hardcoded models for different LLM providers
AVAILABLE_MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "o3-mini",
        "o3",
        "gpt-4.1",
        "gpt-4",
        "gpt-3.5-turbo",
    ],
    "gemini": [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],
    "bedrock": [
        "global.anthropic.claude-opus-4-5-20251101-v1:0",
    ],
}

class SettingsUpdate(BaseModel):
    """Model for updating agent settings."""
    llm_model: str = None
    ollama_url: str = None
    provider: str = None

async def check_k8s_permission(user_id: str, verb: str = "patch", resource: str = "secrets", namespace: str = None) -> bool:
    """
    Check if a user has permission to perform an action on a Kubernetes resource using SubjectAccessReview.
    
    Args:
        user_id: The user ID to check permissions for
        verb: The action to perform (e.g., 'patch', 'update', 'get')
        resource: The resource type (e.g., 'secrets', 'configmaps')
        namespace: The namespace to check permissions in (defaults to the AI agent namespace)
    
    Returns:
        bool: True if the user has permission, False otherwise
    """
    if not namespace:
        namespace = os.environ.get("AGENT_NAMESPACE", "cattle-system")
    
    try:
        # Load Kubernetes config
        k8s_config.load_incluster_config()
        auth_api = client.AuthorizationV1Api()
        
        # Create SubjectAccessReview
        sar = client.V1SubjectAccessReview(
            metadata=client.V1ObjectMeta(),
            spec=client.V1SubjectAccessReviewSpec(
                user=user_id,
                verb=verb,
                resource=resource,
                namespace=namespace,
            )
        )
        
        # Send SubjectAccessReview to Kubernetes API
        response = auth_api.create_subject_access_review(sar)
        
        logging.info(f"Permission check for user {user_id}: {verb} {resource} in {namespace} - Allowed: {response.status.allowed}")
        return response.status.allowed
    
    except ApiException as e:
        logging.error(f"Kubernetes API error during permission check: {e}")
        return False
    except Exception as e:
        logging.error(f"Error checking Kubernetes permissions: {e}")
        return False

@router.get("/models")
async def get_models(request: Request):
    """
    Endpoint to retrieve available LLM models.
    """
    models = AVAILABLE_MODELS.copy()
      
    try:
        ollama_url = request.query_params.get("ollama_url")
        
        if ollama_url:
            async with httpx.AsyncClient(timeout=5.0) as http_client:
                response = await http_client.get(f"{ollama_url}/api/tags")
                if response.status_code == 200:
                    ollama_data = response.json()
                    ollama_models = [model["name"] for model in ollama_data.get("models", [])]
                    models["ollama"] = ollama_models
                else:
                    models["ollama"] = []
    except Exception as e:
        logging.warning(f"Failed to fetch Ollama models: {e}")
        models["ollama"] = []
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"models": models}
    )
    
@router.get("/settings")
async def get_settings(request: Request):
    """
    Endpoint to retrieve current agent settings.
    """
    try:
        user_id = await get_user_id_from_request(request)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        storage_type = request.app.memory_manager.storage_type.value
        
        settings = {"storageType": storage_type}
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"settings": settings}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error retrieving settings: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

@router.put("/settings")
async def update_settings(settings: SettingsUpdate, request: Request):
    """
    Endpoint to update agent settings by patching the llm-config secret.
    Requires permission to patch secrets in the agent namespace.
    """
    try:
        # Get user ID from request
        user_id = await get_user_id_from_request(request)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        # Check if user has permission to update llm-config secret
        agent_namespace = os.environ.get("AGENT_NAMESPACE", "cattle-system")
        has_permission = await check_k8s_permission(
            user_id=user_id,
            verb="patch",
            resource="secrets",
            namespace=agent_namespace
        )
        
        if not has_permission:
            logging.warning(f"User {user_id} attempted to update settings without permission")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": f"User does not have permission to update settings in namespace {agent_namespace}"}
            )
        
        logging.info(f"User {user_id} is updating settings: {settings}")
        
        try:
            # Load Kubernetes config
            k8s_config.load_incluster_config()
            v1 = client.CoreV1Api()
            
            # Get the llm-config secret
            secret_name = os.environ.get("LLM_CONFIG_SECRET", "llm-config")
            secret = v1.read_namespaced_secret(secret_name, agent_namespace)
            
            # Update secret data with new values
            secret_data = secret.data or {}
            
            if settings.llm_model:
                secret_data["SELECTED_LLM_MODEL"] = base64.b64encode(settings.llm_model.encode()).decode()
                logging.info(f"Updated LLM model in secret to {settings.llm_model}")
            
            if settings.ollama_url:
                secret_data["OLLAMA_URL"] = base64.b64encode(settings.ollama_url.encode()).decode()
                logging.info(f"Updated Ollama URL in secret to {settings.ollama_url}")
            
            if settings.provider:
                secret_data["ACTIVE_LLM"] = base64.b64encode(settings.provider.encode()).decode()
                logging.info(f"Updated LLM provider in secret to {settings.provider}")
            
            # Patch the secret with new data
            secret.data = secret_data
            v1.patch_namespaced_secret(secret_name, agent_namespace, secret)
            
            # Reset LLMManager singleton to force reinitialization
            LLMManager._instance = None
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": "Settings updated successfully in llm-config secret",
                    "updated_settings": {
                        "llm_model": settings.llm_model,
                        "ollama_url": settings.ollama_url,
                        "provider": settings.provider,
                    }
                }
            )
        
        except ApiException as e:
            logging.error(f"Kubernetes API error updating llm-config secret: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Failed to update llm-config secret: {str(e)}"}
            )
    
    except Exception as e:
        logging.error(f"Error updating settings: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Failed to update settings: {str(e)}"}
        )
