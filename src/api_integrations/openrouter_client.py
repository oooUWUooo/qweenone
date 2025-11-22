"""
OpenRouter Client for Agentic Architecture System
Provides a dedicated interface for OpenRouter API interactions
"""
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import aiohttp
import requests

from .api_manager import APIConfig, APIResponse, BaseAPIProvider, RateLimiter
from .llm_provider import ProviderType, ModelInfo, llm_provider_registry
from ..utils.logger import setup_logger


@dataclass
class OpenRouterConfig:
    """Configuration specific to OpenRouter"""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_requests: int = 60
    default_model: str = "openchat/openchat-7b"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    custom_headers: Dict[str, str] = field(default_factory=dict)
    http_referer: str = "https://your-app.com"
    x_title: str = "Agentic Architecture System"


@dataclass
class OpenRouterModel:
    """Detailed information about an OpenRouter model"""
    id: str
    name: str
    description: str
    capabilities: List[str]
    pricing: Dict[str, float]  # input, output costs per million tokens
    context_length: int
    architecture: Dict[str, str]
    top_provider: Dict[str, Any]
    per_request_limits: Optional[Dict[str, int]]
    aliases: List[str]
    is_moderated: bool
    created_at: datetime = field(default_factory=datetime.now)


class OpenRouterClient:
    """Dedicated client for OpenRouter API interactions"""
    
    def __init__(self, config: OpenRouterConfig = None):
        self.config = config or OpenRouterConfig(api_key=os.getenv("OPENROUTER_API_KEY", ""))
        self.logger = setup_logger("OpenRouterClient")
        self.rate_limiter = RateLimiter(self.config.rate_limit_requests)
        self.session = None
        self._initialize_session()
        self._available_models: List[OpenRouterModel] = []
        self._model_info_cache: Dict[str, OpenRouterModel] = {}
    
    def _initialize_session(self):
        """Initialize HTTP session with appropriate settings"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> APIResponse:
        """Make an HTTP request to the OpenRouter API"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.config.base_url}{endpoint}"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add OpenRouter-specific headers
        headers["HTTP-Referer"] = self.config.http_referer
        headers["X-Title"] = self.config.x_title
        
        # Add custom headers if provided
        headers.update(self.config.custom_headers)
        
        # Add headers to kwargs if not already present
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"].update(headers)
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                response_data = await response.json()
                
                api_response = APIResponse(
                    success=response.status in [200, 201, 202],
                    data=response_data,
                    status_code=response.status,
                    headers=dict(response.headers),
                    request_id=response.headers.get("X-Request-ID", "")
                )
                
                if response.status >= 400:
                    api_response.error = response_data.get("error", {}).get("message", "Unknown error")
                
                return api_response
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                status_code=0
            )
    
    async def get_models(self) -> APIResponse:
        """Get list of available models from OpenRouter"""
        response = await self._make_request("GET", "/models")
        
        if response.success and "data" in response.data:
            self._available_models = []
            for model_data in response.data["data"]:
                model = OpenRouterModel(
                    id=model_data["id"],
                    name=model_data.get("name", ""),
                    description=model_data.get("description", ""),
                    capabilities=model_data.get("capabilities", []),
                    pricing=model_data.get("pricing", {}),
                    context_length=model_data.get("context_length", 4096),
                    architecture=model_data.get("architecture", {}),
                    top_provider=model_data.get("top_provider", {}),
                    per_request_limits=model_data.get("per_request_limits"),
                    aliases=model_data.get("aliases", []),
                    is_moderated=model_data.get("is_moderated", False)
                )
                self._available_models.append(model)
                self._model_info_cache[model.id] = model
        
        return response
    
    async def get_model_info(self, model_id: str) -> Optional[OpenRouterModel]:
        """Get detailed information about a specific model"""
        if model_id in self._model_info_cache:
            return self._model_info_cache[model_id]
        
        # Fetch all models to update cache
        await self.get_models()
        return self._model_info_cache.get(model_id)
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        **kwargs
    ) -> APIResponse:
        """Send chat completion request to OpenRouter"""
        url = f"{self.config.base_url}/chat/completions"
        
        # Prepare request payload
        payload = {
            "model": model or self.config.default_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add OpenRouter-specific headers
        headers["HTTP-Referer"] = self.config.http_referer
        headers["X-Title"] = self.config.x_title
        
        # Add custom headers if provided
        headers.update(self.config.custom_headers)
        
        try:
            await self.rate_limiter.wait_if_needed()
            async with self.session.post(url, json=payload, headers=headers) as response:
                response_data = await response.json()
                
                api_response = APIResponse(
                    success=response.status == 200,
                    data=response_data,
                    status_code=response.status,
                    headers=dict(response.headers),
                    request_id=response.headers.get("X-Request-ID", "")
                )
                
                if response.status == 200:
                    # Extract token usage and cost if available
                    usage = response_data.get("usage", {})
                    api_response.prompt_tokens = usage.get("prompt_tokens", 0)
                    api_response.completion_tokens = usage.get("completion_tokens", 0)
                    api_response.tokens_used = usage.get("total_tokens", 0)
                    
                    # Extract cost from response if available
                    if "cost" in response_data:
                        api_response.cost = response_data["cost"]
                    else:
                        # Calculate cost based on model
                        model_info = llm_provider_registry.get_model(payload["model"])
                        if model_info:
                            input_cost = (api_response.prompt_tokens / 1000) * model_info.input_cost_per_token
                            output_cost = (api_response.completion_tokens / 1000) * model_info.output_cost_per_token
                            api_response.cost = input_cost + output_cost
                else:
                    api_response.error = response_data.get("error", {}).get("message", "Unknown error")
                
                return api_response
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                status_code=0
            )
    
    async def text_completion(
        self, 
        prompt: str, 
        model: str = None,
        **kwargs
    ) -> APIResponse:
        """Send text completion request to OpenRouter"""
        # For OpenRouter, convert to chat format
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(messages, model=model, **kwargs)
    
    async def get_routes(self) -> APIResponse:
        """Get available routes from OpenRouter"""
        return await self._make_request("GET", "/routes")
    
    async def get_user_info(self) -> APIResponse:
        """Get user information from OpenRouter"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            await self.rate_limiter.wait_if_needed()
            async with self.session.get(
                f"{self.config.base_url}/user",
                headers=headers
            ) as response:
                response_data = await response.json()
                
                api_response = APIResponse(
                    success=response.status == 200,
                    data=response_data,
                    status_code=response.status,
                    headers=dict(response.headers),
                    request_id=response.headers.get("X-Request-ID", "")
                )
                
                if response.status != 200:
                    api_response.error = response_data.get("error", {}).get("message", "Unknown error")
                
                return api_response
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                status_code=0
            )
    
    async def get_usage_info(self) -> APIResponse:
        """Get usage information from OpenRouter"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            await self.rate_limiter.wait_if_needed()
            async with self.session.get(
                f"{self.config.base_url}/user/usage",
                headers=headers
            ) as response:
                response_data = await response.json()
                
                api_response = APIResponse(
                    success=response.status == 200,
                    data=response_data,
                    status_code=response.status,
                    headers=dict(response.headers),
                    request_id=response.headers.get("X-Request-ID", "")
                )
                
                if response.status != 200:
                    api_response.error = response_data.get("error", {}).get("message", "Unknown error")
                
                return api_response
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                status_code=0
            )
    
    async def get_model_pricing(self, model_id: str) -> APIResponse:
        """Get pricing information for a specific model"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            await self.rate_limiter.wait_if_needed()
            async with self.session.get(
                f"{self.config.base_url}/models/{model_id}",
                headers=headers
            ) as response:
                response_data = await response.json()
                
                api_response = APIResponse(
                    success=response.status == 200,
                    data=response_data,
                    status_code=response.status,
                    headers=dict(response.headers),
                    request_id=response.headers.get("X-Request-ID", "")
                )
                
                if response.status != 200:
                    api_response.error = response_data.get("error", {}).get("message", "Unknown error")
                
                return api_response
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                status_code=0
            )
    
    async def check_model_availability(self, model_id: str) -> bool:
        """Check if a specific model is available"""
        model_info = await self.get_model_info(model_id)
        return model_info is not None
    
    def get_available_models(self) -> List[OpenRouterModel]:
        """Get cached list of available models"""
        return self._available_models
    
    def get_model_by_capability(self, capability: str) -> List[OpenRouterModel]:
        """Get models that support a specific capability"""
        return [model for model in self._available_models if capability in model.capabilities]
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class OpenRouterAgentIntegration:
    """Integration layer between OpenRouter and agents"""
    
    def __init__(self, openrouter_client: OpenRouterClient):
        self.client = openrouter_client
        self.logger = setup_logger("OpenRouterAgentIntegration")
    
    async def process_agent_request(
        self, 
        agent_task: Dict[str, Any], 
        model: str = None
    ) -> APIResponse:
        """Process a request from an agent using OpenRouter"""
        # Convert agent task to OpenRouter-compatible format
        messages = self._format_agent_messages(agent_task)
        
        # Determine the appropriate model based on task requirements
        if not model:
            model = await self._select_optimal_model(agent_task)
        
        # Send request to OpenRouter
        response = await self.client.chat_completion(
            messages=messages,
            model=model,
            temperature=agent_task.get("temperature", 0.7),
            max_tokens=agent_task.get("max_tokens", 2048)
        )
        
        return response
    
    def _format_agent_messages(self, agent_task: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format agent task into OpenRouter-compatible messages"""
        messages = []
        
        # Add system message if provided
        if "system_message" in agent_task:
            messages.append({
                "role": "system",
                "content": agent_task["system_message"]
            })
        
        # Add context if provided
        if "context" in agent_task:
            messages.append({
                "role": "user",
                "content": f"Context: {agent_task['context']}"
            })
        
        # Add the main task
        task_content = agent_task.get("content") or agent_task.get("description") or agent_task.get("prompt", "")
        if task_content:
            messages.append({
                "role": "user",
                "content": task_content
            })
        
        # Add any additional messages
        if "messages" in agent_task:
            messages.extend(agent_task["messages"])
        
        return messages
    
    async def _select_optimal_model(self, agent_task: Dict[str, Any]) -> str:
        """Select the optimal model based on task requirements"""
        # Check for specific model requirements
        required_capabilities = agent_task.get("required_capabilities", [])
        
        # Get models that support the required capabilities
        available_models = self.client.get_available_models()
        
        # Filter models based on capabilities
        suitable_models = []
        for capability in required_capabilities:
            models_with_capability = self.client.get_model_by_capability(capability)
            suitable_models.extend(models_with_capability)
        
        # If we found suitable models, use the first one
        if suitable_models:
            return suitable_models[0].id
        
        # Otherwise, use the default model
        return self.client.config.default_model
    
    async def process_batch_requests(
        self, 
        agent_tasks: List[Dict[str, Any]], 
        model: str = None
    ) -> List[APIResponse]:
        """Process multiple agent requests in batch"""
        responses = []
        
        for task in agent_tasks:
            response = await self.process_agent_request(task, model)
            responses.append(response)
            
            # Add a small delay to respect rate limits
            await asyncio.sleep(0.1)
        
        return responses


# Convenience function to create an OpenRouter client with environment variables
def create_openrouter_client_from_env() -> Optional[OpenRouterClient]:
    """Create an OpenRouter client using environment variables"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    
    config = OpenRouterConfig(api_key=api_key)
    return OpenRouterClient(config)