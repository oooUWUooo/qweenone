"""
LLM Provider definitions for Agentic Architecture System
"""
from enum import Enum
from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass, field
from datetime import datetime


class ProviderType(Enum):
    """Enumeration of supported LLM providers"""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    AZURE_OPENAI = "azure_openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    CUSTOM = "custom"


@dataclass
class ModelInfo:
    """Information about a specific model"""
    name: str
    provider: ProviderType
    max_tokens: int = 4096
    context_window: int = 8192
    input_cost_per_token: float = 0.0  # Cost per 1k tokens
    output_cost_per_token: float = 0.0  # Cost per 1k tokens
    supports_vision: bool = False
    supports_tools: bool = False
    supports_functions: bool = False
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    is_available: bool = True


class LLMProvider:
    """Class to manage LLM providers and their models"""
    
    def __init__(self):
        self.providers: Dict[ProviderType, Dict[str, Any]] = {}
        self.models: Dict[str, ModelInfo] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models for each provider"""
        # OpenAI models
        openai_models = [
            ModelInfo(
                name="gpt-3.5-turbo",
                provider=ProviderType.OPENAI,
                max_tokens=4096,
                context_window=16385,
                input_cost_per_token=0.0005,
                output_cost_per_token=0.0015,
                supports_tools=True,
                supports_functions=True,
                description="Fast and capable model for most tasks"
            ),
            ModelInfo(
                name="gpt-4",
                provider=ProviderType.OPENAI,
                max_tokens=8192,
                context_window=128000,
                input_cost_per_token=0.03,
                output_cost_per_token=0.06,
                supports_vision=True,
                supports_tools=True,
                supports_functions=True,
                description="Most capable model, great for complex tasks"
            ),
            ModelInfo(
                name="gpt-4-turbo",
                provider=ProviderType.OPENAI,
                max_tokens=128000,
                context_window=128000,
                input_cost_per_token=0.01,
                output_cost_per_token=0.03,
                supports_vision=True,
                supports_tools=True,
                supports_functions=True,
                description="Latest high-intelligence model"
            ),
        ]
        
        # Anthropic models
        anthropic_models = [
            ModelInfo(
                name="claude-3-haiku",
                provider=ProviderType.ANTHROPIC,
                max_tokens=4096,
                context_window=200000,
                input_cost_per_token=0.00025,
                output_cost_per_token=0.00125,
                supports_vision=True,
                supports_tools=True,
                description="Fast and affordable model"
            ),
            ModelInfo(
                name="claude-3-sonnet",
                provider=ProviderType.ANTHROPIC,
                max_tokens=4096,
                context_window=200000,
                input_cost_per_token=0.003,
                output_cost_per_token=0.015,
                supports_vision=True,
                supports_tools=True,
                description="Ideal balance of intelligence and speed"
            ),
            ModelInfo(
                name="claude-3-opus",
                provider=ProviderType.ANTHROPIC,
                max_tokens=4096,
                context_window=200000,
                input_cost_per_token=0.015,
                output_cost_per_token=0.075,
                supports_vision=True,
                supports_tools=True,
                description="Most powerful model for complex tasks"
            ),
        ]
        
        # OpenRouter models
        openrouter_models = [
            ModelInfo(
                name="openchat/openchat-7b",
                provider=ProviderType.OPENROUTER,
                max_tokens=4096,
                context_window=8192,
                input_cost_per_token=0.00025,
                output_cost_per_token=0.00025,
                description="Open-source model from OpenChat"
            ),
            ModelInfo(
                name="microsoft/wizardlm-2-8x22b",
                provider=ProviderType.OPENROUTER,
                max_tokens=65536,
                context_window=65536,
                input_cost_per_token=0.0002,
                output_cost_per_token=0.0002,
                description="High-performance model from Microsoft"
            ),
            ModelInfo(
                name="meta-llama/llama-3-70b-instruct",
                provider=ProviderType.OPENROUTER,
                max_tokens=8192,
                context_window=8192,
                input_cost_per_token=0.00059,
                output_cost_per_token=0.00079,
                description="Powerful open-source model from Meta"
            ),
            ModelInfo(
                name="anthropic/claude-3-haiku",
                provider=ProviderType.OPENROUTER,
                max_tokens=4096,
                context_window=200000,
                input_cost_per_token=0.00025,
                output_cost_per_token=0.00125,
                supports_vision=True,
                supports_tools=True,
                description="Fast and affordable model via OpenRouter"
            ),
            ModelInfo(
                name="anthropic/claude-3-sonnet",
                provider=ProviderType.OPENROUTER,
                max_tokens=4096,
                context_window=200000,
                input_cost_per_token=0.003,
                output_cost_per_token=0.015,
                supports_vision=True,
                supports_tools=True,
                description="Balanced model via OpenRouter"
            ),
            ModelInfo(
                name="anthropic/claude-3-opus",
                provider=ProviderType.OPENROUTER,
                max_tokens=4096,
                context_window=200000,
                input_cost_per_token=0.015,
                output_cost_per_token=0.075,
                supports_vision=True,
                supports_tools=True,
                description="Most powerful model via OpenRouter"
            ),
            ModelInfo(
                name="google/gemini-pro",
                provider=ProviderType.OPENROUTER,
                max_tokens=8192,
                context_window=32768,
                input_cost_per_token=0.000125,
                output_cost_per_token=0.000375,
                supports_vision=True,
                description="Google's Gemini Pro via OpenRouter"
            ),
            ModelInfo(
                name="openai/gpt-3.5-turbo",
                provider=ProviderType.OPENROUTER,
                max_tokens=4096,
                context_window=16385,
                input_cost_per_token=0.0005,
                output_cost_per_token=0.0015,
                supports_tools=True,
                supports_functions=True,
                description="OpenAI GPT-3.5 via OpenRouter"
            ),
            ModelInfo(
                name="openai/gpt-4",
                provider=ProviderType.OPENROUTER,
                max_tokens=8192,
                context_window=128000,
                input_cost_per_token=0.03,
                output_cost_per_token=0.06,
                supports_vision=True,
                supports_tools=True,
                supports_functions=True,
                description="OpenAI GPT-4 via OpenRouter"
            ),
        ]
        
        # Add all models to the registry
        all_models = openai_models + anthropic_models + openrouter_models
        for model in all_models:
            self.models[model.name] = model
    
    def register_provider(self, provider_type: ProviderType, config: Dict[str, Any]):
        """Register a new provider with its configuration"""
        self.providers[provider_type] = config
    
    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        return self.models.get(model_name)
    
    def list_models(self, provider_type: Optional[ProviderType] = None) -> List[ModelInfo]:
        """List all available models, optionally filtered by provider"""
        if provider_type:
            return [model for model in self.models.values() if model.provider == provider_type]
        return list(self.models.values())
    
    def list_providers(self) -> List[ProviderType]:
        """List all registered provider types"""
        return list(self.providers.keys())
    
    def get_provider_models(self, provider_type: ProviderType) -> List[ModelInfo]:
        """Get all models for a specific provider"""
        return [model for model in self.models.values() if model.provider == provider_type]
    
    def validate_model(self, model_name: str, provider_type: ProviderType) -> bool:
        """Validate if a model is available for a specific provider"""
        model = self.get_model(model_name)
        return model is not None and model.provider == provider_type and model.is_available
    
    def get_model_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for using a model with given token counts"""
        model = self.get_model(model_name)
        if not model:
            return 0.0
        
        input_cost = (input_tokens / 1000) * model.input_cost_per_token
        output_cost = (output_tokens / 1000) * model.output_cost_per_token
        return input_cost + output_cost
    
    def get_recommended_model(self, task_type: str = "general") -> Optional[ModelInfo]:
        """Get a recommended model based on task type"""
        if task_type == "creative":
            # Return a model good for creative tasks
            return self.models.get("gpt-4") or self.models.get("claude-3-opus")
        elif task_type == "fast":
            # Return a fast, cost-effective model
            return self.models.get("gpt-3.5-turbo") or self.models.get("claude-3-haiku")
        elif task_type == "vision":
            # Return a model that supports vision
            vision_models = [m for m in self.models.values() if m.supports_vision]
            return vision_models[0] if vision_models else self.models.get("gpt-4")
        else:
            # Return a general-purpose model
            return self.models.get("gpt-3.5-turbo") or self.models.get("claude-3-sonnet")
    
    def get_provider_capabilities(self, provider_type: ProviderType) -> Dict[str, Any]:
        """Get capabilities and features of a specific provider"""
        if provider_type == ProviderType.OPENAI:
            return {
                "name": "OpenAI",
                "supports_streaming": True,
                "supports_functions": True,
                "supports_tools": True,
                "supports_vision": True,
                "max_concurrent_requests": 1000,
                "typical_response_time": "0.5-2s"
            }
        elif provider_type == ProviderType.ANTHROPIC:
            return {
                "name": "Anthropic",
                "supports_streaming": True,
                "supports_functions": True,
                "supports_tools": True,
                "supports_vision": True,
                "max_concurrent_requests": 500,
                "typical_response_time": "1-3s"
            }
        elif provider_type == ProviderType.OPENROUTER:
            return {
                "name": "OpenRouter",
                "supports_streaming": True,
                "supports_functions": True,
                "supports_tools": True,
                "supports_vision": True,
                "max_concurrent_requests": 2000,
                "typical_response_time": "0.5-5s",
                "model_count": len(self.get_provider_models(provider_type)),
                "cost_savings": "Up to 50% compared to direct providers"
            }
        else:
            return {
                "name": provider_type.value,
                "supports_streaming": True,
                "supports_functions": False,
                "supports_tools": False,
                "supports_vision": False,
                "max_concurrent_requests": 100,
                "typical_response_time": "1-5s"
            }
    
    def get_optimal_provider(self, requirements: Dict[str, Any]) -> Optional[ProviderType]:
        """Get the optimal provider based on requirements"""
        required_features = requirements.get("features", [])
        budget_limit = requirements.get("budget", float('inf'))
        response_time_limit = requirements.get("response_time", float('inf'))
        model_preference = requirements.get("model", "")
        
        # For now, return the first provider that meets the requirements
        for provider_type in self.providers:
            capabilities = self.get_provider_capabilities(provider_type)
            
            # Check if provider supports required features
            supports_features = True
            if "vision" in required_features and not capabilities.get("supports_vision"):
                supports_features = False
            if "functions" in required_features and not capabilities.get("supports_functions"):
                supports_features = False
            if "tools" in required_features and not capabilities.get("supports_tools"):
                supports_features = False
            
            if supports_features:
                return provider_type
        
        # If no provider meets requirements, return the first available one
        return next(iter(self.providers)) if self.providers else None


# Global instance for easy access
llm_provider_registry = LLMProvider()