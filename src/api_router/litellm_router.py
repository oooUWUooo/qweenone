"""
LiteLLM Router Integration for Unified LLM API Access

Provides a simplified, production-ready interface to 100+ LLM providers
through LiteLLM's universal API gateway. Handles routing, fallbacks,
retries, and load balancing automatically.

Key Features:
- Universal API for 100+ LLM providers
- Automatic fallback on provider failures
- Built-in retry logic with exponential backoff
- Cost tracking and budget management
- Provider-agnostic request format
- Streaming support
- Function calling support
"""

import asyncio
import os
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from litellm import Router, completion, acompletion
    from litellm import get_supported_openai_params
    from litellm.router import LiteLLMRouter
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    Router = None
    completion = None
    acompletion = None

try:
    from ..utils.logger import setup_logger
    from ..api_integrations.llm_provider import ProviderType, ModelInfo
except ImportError:
    setup_logger = lambda x: None
    ProviderType = None
    ModelInfo = None


@dataclass
class LiteLLMConfig:
    """Configuration for LiteLLM router"""
    model_list: List[Dict[str, Any]] = field(default_factory=list)
    routing_strategy: str = "least-busy"
    num_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 300
    fallbacks: List[str] = field(default_factory=list)
    enable_caching: bool = True
    enable_logging: bool = True
    redis_host: Optional[str] = None
    redis_port: Optional[int] = None
    budget_config: Optional[Dict[str, Any]] = None


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    model: str
    provider: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    success: bool = False
    error: Optional[str] = None
    retry_count: int = 0
    fallback_used: bool = False


class LiteLLMUnifiedRouter:
    """
    Unified LLM router using LiteLLM for universal provider access
    
    Simplifies LLM integration by providing a single interface to 100+ providers
    including OpenAI, Anthropic, Google, Cohere, Azure, AWS, and more.
    """
    
    def __init__(self, config: LiteLLMConfig = None):
        """Initialize LiteLLM router"""
        
        if not LITELLM_AVAILABLE:
            raise RuntimeError(
                "LiteLLM not available. Install with: pip install litellm"
            )
        
        self.config = config or LiteLLMConfig()
        self.logger = setup_logger("LiteLLMUnifiedRouter")
        
        # Initialize model list if empty
        if not self.config.model_list:
            self.config.model_list = self._create_default_model_list()
        
        # Create LiteLLM router
        try:
            self.router = Router(
                model_list=self.config.model_list,
                routing_strategy=self.config.routing_strategy,
                num_retries=self.config.num_retries,
                retry_after=self.config.retry_delay,
                timeout=self.config.timeout,
                fallbacks=self.config.fallbacks if self.config.fallbacks else None,
                set_verbose=self.config.enable_logging,
                redis_host=self.config.redis_host,
                redis_port=self.config.redis_port,
            )
            
            if self.logger:
                self.logger.info(f"LiteLLM router initialized with {len(self.config.model_list)} models")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize LiteLLM router: {e}")
            raise e
        
        # Metrics tracking
        self.request_metrics: List[RequestMetrics] = []
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
    def _create_default_model_list(self) -> List[Dict[str, Any]]:
        """Create default model configuration list"""
        
        models = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "os.environ/OPENAI_API_KEY",
                    "timeout": 300,
                },
                "model_info": {
                    "supports_function_calling": True,
                    "supports_vision": False,
                    "max_tokens": 4096,
                }
            },
            {
                "model_name": "gpt-4-turbo",
                "litellm_params": {
                    "model": "gpt-4-turbo-preview",
                    "api_key": "os.environ/OPENAI_API_KEY",
                    "timeout": 300,
                },
                "model_info": {
                    "supports_function_calling": True,
                    "supports_vision": True,
                    "max_tokens": 128000,
                }
            },
            {
                "model_name": "claude-3-sonnet",
                "litellm_params": {
                    "model": "claude-3-sonnet-20240229",
                    "api_key": "os.environ/ANTHROPIC_API_KEY",
                    "timeout": 300,
                },
                "model_info": {
                    "supports_function_calling": True,
                    "supports_vision": True,
                    "max_tokens": 4096,
                }
            },
            {
                "model_name": "claude-3-haiku",
                "litellm_params": {
                    "model": "claude-3-haiku-20240307",
                    "api_key": "os.environ/ANTHROPIC_API_KEY",
                    "timeout": 300,
                },
                "model_info": {
                    "supports_function_calling": True,
                    "supports_vision": True,
                    "max_tokens": 4096,
                }
            },
            {
                "model_name": "gemini-2.0-flash",
                "litellm_params": {
                    "model": "gemini/gemini-2.0-flash",
                    "api_key": "os.environ/GOOGLE_API_KEY",
                    "timeout": 300,
                },
                "model_info": {
                    "supports_function_calling": True,
                    "supports_vision": True,
                    "max_tokens": 8192,
                }
            },
            {
                "model_name": "openrouter-auto",
                "litellm_params": {
                    "model": "openrouter/auto",
                    "api_key": "os.environ/OPENROUTER_API_KEY",
                    "timeout": 300,
                },
                "model_info": {
                    "supports_function_calling": True,
                    "supports_vision": False,
                    "max_tokens": 4096,
                }
            },
        ]
        scrapybara_key = os.getenv("SCRAPYBARA_API_KEY")
        if scrapybara_key:
            api_base = os.getenv("SCRAPYBARA_API_BASE", "https://api.scrapybara.com")
            models.append(
                {
                    "model_name": "scrapybara-research",
                    "litellm_params": {
                        "model": "scrapybara/research",
                        "api_key": "os.environ/SCRAPYBARA_API_KEY",
                        "api_base": api_base,
                        "timeout": int(os.getenv("SCRAPYBARA_TASK_TIMEOUT", "600")),
                    },
                    "model_info": {
                        "supports_function_calling": False,
                        "supports_vision": True,
                        "max_tokens": 8192,
                    }
                }
            )
        return models
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Universal chat completion across all providers
        
        Args:
            messages: List of message dictionaries with role and content
            model: Model name (auto-routes to appropriate provider)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Standardized response dictionary
        """
        
        request_id = f"req_{int(time.time() * 1000)}"
        start_time = datetime.now()
        
        metrics = RequestMetrics(
            request_id=request_id,
            model=model,
            provider="litellm",
            start_time=start_time
        )
        
        try:
            if self.logger:
                self.logger.info(f"[{request_id}] Chat completion request: model={model}, messages={len(messages)}")
            
            # Make async request through LiteLLM router
            response = await self.router.acompletion(
                model=model,
                messages=messages,
                **kwargs
            )
            
            # Extract response data
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse response
            content = response.choices[0].message.content if response.choices else ""
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            # Calculate cost (LiteLLM provides this automatically)
            cost = 0.0
            if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
                cost = response._hidden_params['response_cost']
            
            # Update metrics
            metrics.end_time = end_time
            metrics.duration = duration
            metrics.tokens_used = tokens_used
            metrics.cost = cost
            metrics.success = True
            
            # Update global stats
            self.total_requests += 1
            self.total_tokens += tokens_used
            self.total_cost += cost
            
            self.request_metrics.append(metrics)
            
            if self.logger:
                self.logger.info(
                    f"[{request_id}] Success: {tokens_used} tokens, "
                    f"${cost:.6f}, {duration:.2f}s"
                )
            
            return {
                "success": True,
                "content": content,
                "model": model,
                "tokens_used": tokens_used,
                "cost": cost,
                "duration": duration,
                "provider": "litellm",
                "request_id": request_id,
                "raw_response": response
            }
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            metrics.end_time = end_time
            metrics.duration = duration
            metrics.success = False
            metrics.error = str(e)
            
            self.request_metrics.append(metrics)
            self.total_requests += 1
            
            if self.logger:
                self.logger.error(f"[{request_id}] Failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "duration": duration,
                "provider": "litellm",
                "request_id": request_id
            }
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        **kwargs
    ):
        """
        Streaming chat completion
        
        Yields response chunks as they arrive for real-time streaming
        """
        
        try:
            if self.logger:
                self.logger.info(f"Starting streaming completion: model={model}")
            
            # Enable streaming
            kwargs['stream'] = True
            
            # Use LiteLLM router for streaming
            response = await self.router.acompletion(
                model=model,
                messages=messages,
                **kwargs
            )
            
            # Yield chunks as they arrive
            async for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield {
                            "success": True,
                            "content": delta.content,
                            "chunk": chunk,
                            "finish_reason": chunk.choices[0].finish_reason if hasattr(chunk.choices[0], 'finish_reason') else None
                        }
                        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Streaming failed: {e}")
            yield {
                "success": False,
                "error": str(e)
            }
    
    def add_model(self, model_config: Dict[str, Any]):
        """Add a new model to the router"""
        self.config.model_list.append(model_config)
        
        # Reinitialize router with updated model list
        self.router = Router(
            model_list=self.config.model_list,
            routing_strategy=self.config.routing_strategy,
            num_retries=self.config.num_retries,
        )
        
        if self.logger:
            self.logger.info(f"Added model: {model_config.get('model_name', 'unknown')}")
    
    def get_available_models(self) -> List[str]:
        """Get list of all available models"""
        return [model["model_name"] for model in self.config.model_list]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        
        successful_requests = sum(1 for m in self.request_metrics if m.success)
        failed_requests = self.total_requests - successful_requests
        
        avg_duration = 0.0
        if successful_requests > 0:
            avg_duration = sum(m.duration for m in self.request_metrics if m.success) / successful_requests
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "avg_duration": avg_duration,
            "available_models": len(self.config.model_list),
            "routing_strategy": self.config.routing_strategy,
            "litellm_enabled": True
        }
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get per-model performance statistics"""
        
        model_stats = {}
        
        for metric in self.request_metrics:
            if metric.model not in model_stats:
                model_stats[metric.model] = {
                    "requests": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "total_duration": 0.0
                }
            
            stats = model_stats[metric.model]
            stats["requests"] += 1
            
            if metric.success:
                stats["successes"] += 1
                stats["total_tokens"] += metric.tokens_used
                stats["total_cost"] += metric.cost
                stats["total_duration"] += metric.duration
            else:
                stats["failures"] += 1
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats["successes"] > 0:
                stats["avg_tokens"] = stats["total_tokens"] / stats["successes"]
                stats["avg_cost"] = stats["total_cost"] / stats["successes"]
                stats["avg_duration"] = stats["total_duration"] / stats["successes"]
                stats["success_rate"] = (stats["successes"] / stats["requests"]) * 100
            else:
                stats["avg_tokens"] = 0
                stats["avg_cost"] = 0.0
                stats["avg_duration"] = 0.0
                stats["success_rate"] = 0.0
        
        return model_stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of LiteLLM router and available models"""
        
        health_status = {
            "litellm_available": LITELLM_AVAILABLE,
            "router_initialized": self.router is not None,
            "models_configured": len(self.config.model_list),
            "timestamp": datetime.now().isoformat(),
            "model_health": {}
        }
        
        # Test a simple completion on each model (optional)
        for model_config in self.config.model_list[:3]:  # Test first 3 models only
            model_name = model_config["model_name"]
            try:
                test_response = await self.chat_completion(
                    messages=[{"role": "user", "content": "Hi"}],
                    model=model_name,
                    max_tokens=5
                )
                health_status["model_health"][model_name] = {
                    "status": "healthy" if test_response["success"] else "unhealthy",
                    "response_time": test_response.get("duration", 0)
                }
            except Exception as e:
                health_status["model_health"][model_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return health_status


async def quick_completion(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    **kwargs
) -> str:
    """
    Quick helper function for simple completions
    
    Args:
        prompt: User prompt
        model: Model to use
        **kwargs: Additional parameters
        
    Returns:
        Response content as string
    """
    
    if not LITELLM_AVAILABLE:
        raise RuntimeError("LiteLLM not available")
    
    messages = [{"role": "user", "content": prompt}]
    
    response = await acompletion(
        model=model,
        messages=messages,
        **kwargs
    )
    
    return response.choices[0].message.content if response.choices else ""


if __name__ == "__main__":
    """Demo LiteLLM router"""
    
    async def demo_litellm():
        """Demonstrate LiteLLM unified routing"""
        
        print("🚀 === LITELLM UNIFIED ROUTER DEMO ===")
        
        if not LITELLM_AVAILABLE:
            print("❌ LiteLLM not available. Install with: pip install litellm")
            return
        
        # Create router
        router = LiteLLMUnifiedRouter()
        
        # Test different models
        test_prompts = [
            ("What is Python?", "gpt-3.5-turbo"),
            ("Explain machine learning briefly", "claude-3-haiku"),
        ]
        
        for prompt, model in test_prompts:
            print(f"\n🎯 Testing {model}: {prompt}")
            
            result = await router.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=100
            )
            
            if result["success"]:
                print(f"✅ Success!")
                print(f"   Response: {result['content'][:100]}...")
                print(f"   Tokens: {result['tokens_used']}, Cost: ${result['cost']:.6f}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Show statistics
        stats = router.get_statistics()
        print(f"\n📊 Router Statistics:")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        print(f"   Total cost: ${stats['total_cost']:.6f}")
        print(f"   Available models: {stats['available_models']}")
    
    asyncio.run(demo_litellm())
