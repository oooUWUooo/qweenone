"""
API Manager module for Agentic Architecture System
Handles integration with various LLM providers including OpenRouter
"""
import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import aiohttp
import requests

from .llm_provider import LLMProvider, ProviderType
from ..utils.logger import setup_logger


@dataclass
class APIConfig:
    """Configuration for API integration"""
    provider: ProviderType
    api_key: str = ""
    base_url: str = ""
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_requests: int = 60  # requests per minute
    rate_limit_window: int = 60  # seconds
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    custom_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class APIResponse:
    """Standard response from API calls"""
    success: bool
    data: Any = None
    error: str = ""
    status_code: int = 0
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


class RateLimiter:
    """Simple rate limiter to prevent API overuse"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.requests_per_minute:
                # Wait until oldest request is older than 1 minute
                oldest = min(self.requests)
                wait_time = 60 - (now - oldest)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            self.requests.append(now)


class BaseAPIProvider(ABC):
    """Abstract base class for API providers"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.logger = setup_logger(f"{self.__class__.__name__}_{config.provider.value}")
        self.rate_limiter = RateLimiter(config.rate_limit_requests)
        self.session = None
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize HTTP session with appropriate settings"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> APIResponse:
        """Send chat completion request to the API"""
        pass
    
    @abstractmethod
    async def text_completion(self, prompt: str, **kwargs) -> APIResponse:
        """Send text completion request to the API"""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str, **kwargs) -> APIResponse:
        """Generate embeddings for text"""
        pass
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()


class OpenAICompatibleProvider(BaseAPIProvider):
    """Base class for OpenAI-compatible providers like OpenRouter"""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.base_url = config.base_url or self._get_default_base_url()
    
    def _get_default_base_url(self) -> str:
        """Get the default base URL for the provider"""
        if self.config.provider == ProviderType.OPENROUTER:
            return "https://openrouter.ai/api/v1"
        return "https://api.openai.com/v1"
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> APIResponse:
        """Send chat completion request"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/chat/completions"
        
        # Prepare request payload
        payload = {
            "model": kwargs.get("model", self.config.model),
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
        
        # Add custom headers if provided
        headers.update(self.config.custom_headers)
        
        try:
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
                    
                    # Calculate cost based on model (simplified)
                    api_response.cost = self._calculate_cost(
                        api_response.prompt_tokens, 
                        api_response.completion_tokens,
                        payload["model"]
                    )
                else:
                    api_response.error = response_data.get("error", {}).get("message", "Unknown error")
                
                return api_response
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                status_code=0
            )
    
    async def text_completion(self, prompt: str, **kwargs) -> APIResponse:
        """Send text completion request"""
        # For OpenAI-compatible APIs, convert to chat format
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(messages, **kwargs)
    
    async def embed_text(self, text: str, **kwargs) -> APIResponse:
        """Generate embeddings for text"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "input": text,
            "model": kwargs.get("model", "text-embedding-ada-002")
        }
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
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
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate approximate cost based on tokens and model"""
        # Cost per 1k tokens (simplified)
        cost_per_1k = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            # OpenRouter models
            "openchat/openchat-7b": {"input": 0.00025, "output": 0.00025},
            "microsoft/wizardlm-2-8x22b": {"input": 0.0002, "output": 0.0002},
            "meta-llama/llama-3-70b-instruct": {"input": 0.00059, "output": 0.00079},
            "anthropic/claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "anthropic/claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "anthropic/claude-3-opus": {"input": 0.015, "output": 0.075},
        }
        
        model_costs = cost_per_1k.get(model, {"input": 0.001, "output": 0.002})
        input_cost = (prompt_tokens / 1000) * model_costs["input"]
        output_cost = (completion_tokens / 1000) * model_costs["output"]
        
        return input_cost + output_cost


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter-specific provider implementation"""
    
    def __init__(self, config: APIConfig):
        # Set default OpenRouter base URL
        if not config.base_url:
            config.base_url = "https://openrouter.ai/api/v1"
        super().__init__(config)
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> APIResponse:
        """Send chat completion request to OpenRouter"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/chat/completions"
        
        # Prepare request payload
        payload = {
            "model": kwargs.get("model", self.config.model),
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
        headers["HTTP-Referer"] = kwargs.get("http_referer", "https://your-app.com")
        headers["X-Title"] = kwargs.get("x_title", "Agentic Architecture System")
        
        # Add custom headers if provided
        headers.update(self.config.custom_headers)
        
        try:
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
                        api_response.cost = self._calculate_cost(
                            api_response.prompt_tokens, 
                            api_response.completion_tokens,
                            payload["model"]
                        )
                else:
                    api_response.error = response_data.get("error", {}).get("message", "Unknown error")
                
                return api_response
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                status_code=0
            )


class APIManager:
    """Main API manager that handles multiple providers"""
    
    def __init__(self):
        self.providers: Dict[ProviderType, BaseAPIProvider] = {}
        self.logger = setup_logger("APIManager")
        self.active_requests = 0
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.configs: Dict[ProviderType, APIConfig] = {}
    
    def register_provider(self, config: APIConfig) -> bool:
        """Register a new API provider"""
        try:
            if config.provider == ProviderType.OPENROUTER:
                provider = OpenRouterProvider(config)
            elif config.provider in [ProviderType.OPENAI, ProviderType.AZURE_OPENAI, ProviderType.OPENAI_COMPATIBLE]:
                provider = OpenAICompatibleProvider(config)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
            
            self.providers[config.provider] = provider
            self.configs[config.provider] = config
            self.logger.info(f"Registered provider: {config.provider.value}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register provider {config.provider.value}: {str(e)}")
            return False
    
    def get_provider(self, provider_type: ProviderType) -> Optional[BaseAPIProvider]:
        """Get a registered provider"""
        return self.providers.get(provider_type)
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        provider_type: ProviderType = None,
        **kwargs
    ) -> APIResponse:
        """Send chat completion request using the specified provider"""
        if provider_type is None:
            # Use the first available provider
            if not self.providers:
                return APIResponse(success=False, error="No providers registered")
            provider_type = next(iter(self.providers))
        
        provider = self.providers.get(provider_type)
        if not provider:
            return APIResponse(success=False, error=f"Provider {provider_type.value} not registered")
        
        response = await provider.chat_completion(messages, **kwargs)
        
        # Update statistics
        if response.success:
            self.active_requests += 1
            self.total_requests += 1
            self.total_tokens += response.tokens_used
            self.total_cost += response.cost
        
        return response
    
    async def text_completion(
        self, 
        prompt: str, 
        provider_type: ProviderType = None,
        **kwargs
    ) -> APIResponse:
        """Send text completion request using the specified provider"""
        if provider_type is None:
            # Use the first available provider
            if not self.providers:
                return APIResponse(success=False, error="No providers registered")
            provider_type = next(iter(self.providers))
        
        provider = self.providers.get(provider_type)
        if not provider:
            return APIResponse(success=False, error=f"Provider {provider_type.value} not registered")
        
        response = await provider.text_completion(prompt, **kwargs)
        
        # Update statistics
        if response.success:
            self.active_requests += 1
            self.total_requests += 1
            self.total_tokens += response.tokens_used
            self.total_cost += response.cost
        
        return response
    
    async def embed_text(
        self, 
        text: str, 
        provider_type: ProviderType = None,
        **kwargs
    ) -> APIResponse:
        """Generate embeddings using the specified provider"""
        if provider_type is None:
            # Use the first available provider
            if not self.providers:
                return APIResponse(success=False, error="No providers registered")
            provider_type = next(iter(self.providers))
        
        provider = self.providers.get(provider_type)
        if not provider:
            return APIResponse(success=False, error=f"Provider {provider_type.value} not registered")
        
        response = await provider.embed_text(text, **kwargs)
        
        return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "registered_providers": list(self.providers.keys()),
            "provider_configs": {k.value: v.config.model for k, v in self.providers.items()}
        }
    
    async def close_all(self):
        """Close all provider sessions"""
        for provider in self.providers.values():
            await provider.close()
        self.logger.info("Closed all provider sessions")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()