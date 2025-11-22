"""
Provider Router for Agentic Architecture System
Manages routing logic specific to different LLM providers
"""
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from ..api_integrations import APIManager, APIConfig, APIResponse
from ..api_integrations.llm_provider import ProviderType, llm_provider_registry
from ..utils.logger import setup_logger


@dataclass
class ProviderHealth:
    """Health metrics for a provider"""
    provider: ProviderType
    is_healthy: bool = True
    last_check: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    error_rate: float = 0.0
    success_rate: float = 1.0
    available_models: List[str] = field(default_factory=list)
    quota_remaining: Optional[int] = None
    estimated_reset_time: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


@dataclass
class ProviderPreference:
    """Preference settings for a provider"""
    provider: ProviderType
    priority: int = 1  # Lower number = higher priority
    max_concurrent_requests: int = 10
    max_requests_per_minute: int = 100
    fallback_providers: List[ProviderType] = field(default_factory=list)
    enabled: bool = True
    weight: float = 1.0  # For weighted routing


class ProviderRouter:
    """Router that manages different LLM providers and their routing"""
    
    def __init__(self):
        self.logger = setup_logger("ProviderRouter")
        self.api_manager = APIManager()
        self.provider_preferences: Dict[ProviderType, ProviderPreference] = {}
        self.provider_health: Dict[ProviderType, ProviderHealth] = {}
        self.active_requests: Dict[ProviderType, int] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.router_config = {}
        self._request_id_counter = 0
        
        # Initialize health records for all provider types
        for provider in ProviderType:
            self.provider_health[provider] = ProviderHealth(provider=provider)
    
    def register_provider(self, config: APIConfig) -> bool:
        """Register a provider with its configuration and preferences"""
        success = self.api_manager.register_provider(config)
        if not success:
            return False
        
        # Initialize provider preference with default values
        if config.provider not in self.provider_preferences:
            self.provider_preferences[config.provider] = ProviderPreference(
                provider=config.provider,
                priority=1,
                fallback_providers=[],
                enabled=True,
                weight=1.0
            )
        
        # Initialize active request counter
        self.active_requests[config.provider] = 0
        
        # Update health information
        self._update_provider_health(config.provider)
        
        self.logger.info(f"Registered provider: {config.provider.value}")
        return True
    
    def set_provider_preference(self, preference: ProviderPreference):
        """Set preference settings for a provider"""
        self.provider_preferences[preference.provider] = preference
        self.logger.info(f"Updated preferences for provider: {preference.provider.value}")
    
    def add_fallback_provider(self, primary: ProviderType, fallback: ProviderType):
        """Add a fallback provider for when the primary is unavailable"""
        if primary not in self.provider_preferences:
            self.provider_preferences[primary] = ProviderPreference(provider=primary)
        
        prefs = self.provider_preferences[primary]
        if fallback not in prefs.fallback_providers:
            prefs.fallback_providers.append(fallback)
    
    async def route_request(
        self, 
        messages: List[Dict[str, str]], 
        preferred_provider: ProviderType = None,
        **kwargs
    ) -> APIResponse:
        """Route a request to an appropriate provider"""
        request_id = self._get_next_request_id()
        
        # Determine the best provider to use
        target_provider = await self._select_best_provider(
            messages, 
            preferred_provider, 
            **kwargs
        )
        
        if not target_provider:
            return APIResponse(
                success=False,
                error="No available providers",
                status_code=503
            )
        
        # Check if provider is healthy and within limits
        if not await self._can_use_provider(target_provider):
            # Try fallback providers
            fallback_provider = await self._get_fallback_provider(target_provider)
            if fallback_provider:
                target_provider = fallback_provider
            else:
                return APIResponse(
                    success=False,
                    error=f"Provider {target_provider.value} unavailable and no fallbacks",
                    status_code=503
                )
        
        # Track active request
        self.active_requests[target_provider] += 1
        
        try:
            # Make the API call
            response = await self.api_manager.chat_completion(
                messages=messages,
                provider_type=target_provider,
                **kwargs
            )
            
            # Update health metrics
            await self._update_provider_health_after_request(
                target_provider, 
                response, 
                request_id
            )
            
            # Add routing information to response
            if response.success:
                response.data = response.data or {}
                response.data["routing_info"] = {
                    "provider_used": target_provider.value,
                    "request_id": request_id,
                    "provider_selection_reason": "automatic"
                }
            
            return response
            
        except Exception as e:
            # Update health metrics for failure
            await self._update_provider_health_after_request(
                target_provider, 
                APIResponse(success=False, error=str(e)), 
                request_id
            )
            
            # Try fallback if this was the primary provider
            fallback_provider = await self._get_fallback_provider(target_provider)
            if fallback_provider and fallback_provider != target_provider:
                self.logger.info(f"Retrying with fallback provider: {fallback_provider.value}")
                self.active_requests[target_provider] -= 1  # Decrement for original provider
                self.active_requests[fallback_provider] += 1  # Increment for fallback
                
                try:
                    fallback_response = await self.api_manager.chat_completion(
                        messages=messages,
                        provider_type=fallback_provider,
                        **kwargs
                    )
                    
                    await self._update_provider_health_after_request(
                        fallback_provider, 
                        fallback_response, 
                        f"{request_id}_fallback"
                    )
                    
                    if fallback_response.success:
                        fallback_response.data = fallback_response.data or {}
                        fallback_response.data["routing_info"] = {
                            "provider_used": fallback_provider.value,
                            "request_id": f"{request_id}_fallback",
                            "provider_selection_reason": "fallback"
                        }
                    
                    return fallback_response
                except Exception as fallback_error:
                    self.logger.error(f"Fallback provider {fallback_provider.value} also failed: {str(fallback_error)}")
            
            return APIResponse(
                success=False,
                error=f"Request failed with provider {target_provider.value}: {str(e)}",
                status_code=500
            )
        
        finally:
            # Decrement active request counter
            self.active_requests[target_provider] -= 1
    
    async def _select_best_provider(
        self, 
        messages: List[Dict[str, str]], 
        preferred_provider: ProviderType = None,
        **kwargs
    ) -> Optional[ProviderType]:
        """Select the best provider based on preferences and availability"""
        
        # If a preferred provider is specified, try to use it
        if preferred_provider:
            if self._is_provider_available(preferred_provider):
                return preferred_provider
        
        # Get all available providers
        available_providers = [
            provider for provider in self.provider_preferences.keys()
            if self._is_provider_available(provider)
        ]
        
        if not available_providers:
            return None
        
        # Sort by priority (lower number = higher priority)
        available_providers.sort(
            key=lambda p: self.provider_preferences[p].priority
        )
        
        # If there are multiple providers with the same priority, 
        # use weighted selection based on preferences
        highest_priority = self.provider_preferences[available_providers[0]].priority
        same_priority_providers = [
            p for p in available_providers
            if self.provider_preferences[p].priority == highest_priority
        ]
        
        if len(same_priority_providers) == 1:
            return same_priority_providers[0]
        
        # Use weighted selection for providers with same priority
        return self._weighted_provider_selection(same_priority_providers)
    
    def _is_provider_available(self, provider: ProviderType) -> bool:
        """Check if a provider is available based on health and limits"""
        health = self.provider_health.get(provider)
        if not health or not health.is_healthy:
            return False
        
        prefs = self.provider_preferences.get(provider)
        if not prefs or not prefs.enabled:
            return False
        
        # Check if we're at the concurrent request limit
        active_requests = self.active_requests.get(provider, 0)
        if prefs.max_concurrent_requests and active_requests >= prefs.max_concurrent_requests:
            return False
        
        return True
    
    async def _can_use_provider(self, provider: ProviderType) -> bool:
        """Check if a provider can be used right now"""
        if not self._is_provider_available(provider):
            return False
        
        # Additional checks could go here (rate limits, quotas, etc.)
        return True
    
    async def _get_fallback_provider(self, primary_provider: ProviderType) -> Optional[ProviderType]:
        """Get a fallback provider for the given primary provider"""
        prefs = self.provider_preferences.get(primary_provider)
        if not prefs or not prefs.fallback_providers:
            return None
        
        # Find the first available fallback provider
        for fallback in prefs.fallback_providers:
            if self._is_provider_available(fallback):
                return fallback
        
        return None
    
    def _weighted_provider_selection(self, providers: List[ProviderType]) -> ProviderType:
        """Select a provider using weighted random selection"""
        total_weight = sum(
            self.provider_preferences[p].weight 
            for p in providers 
            if p in self.provider_preferences
        )
        
        if total_weight <= 0:
            return providers[0]  # Fallback to first provider
        
        # Select based on weights
        import random
        rand_val = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for provider in providers:
            weight = self.provider_preferences[provider].weight
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return provider
        
        # Fallback
        return providers[0]
    
    async def _update_provider_health_after_request(
        self, 
        provider: ProviderType, 
        response: APIResponse, 
        request_id: str
    ):
        """Update health metrics after a request"""
        health = self.provider_health[provider]
        
        health.total_requests += 1
        
        if response.success:
            health.successful_requests += 1
        else:
            health.failed_requests += 1
        
        # Update success rate
        health.success_rate = health.successful_requests / health.total_requests
        health.error_rate = health.failed_requests / health.total_requests
        
        # Update health status based on error rate
        health.is_healthy = health.error_rate < 0.5  # Mark unhealthy if >50% error rate
        health.last_check = datetime.now()
    
    def _update_provider_health(self, provider: ProviderType):
        """Update health information for a provider"""
        # This would typically make a health check API call in a real implementation
        # For now, we'll just update the timestamp
        health = self.provider_health[provider]
        health.last_check = datetime.now()
        
        # In a real implementation, you might check:
        # - API endpoint availability
        # - Rate limits
        # - Quotas
        # - Response times
        # - Error rates from recent requests
    
    async def get_provider_status(self, provider: ProviderType) -> Optional[ProviderHealth]:
        """Get the health status of a specific provider"""
        # Refresh health info
        self._update_provider_health(provider)
        return self.provider_health.get(provider)
    
    async def get_all_provider_status(self) -> Dict[ProviderType, ProviderHealth]:
        """Get health status of all providers"""
        for provider in self.provider_health.keys():
            self._update_provider_health(provider)
        return self.provider_health.copy()
    
    def get_provider_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all providers"""
        stats = {
            "total_requests": sum(h.total_requests for h in self.provider_health.values()),
            "successful_requests": sum(h.successful_requests for h in self.provider_health.values()),
            "failed_requests": sum(h.failed_requests for h in self.provider_health.values()),
            "active_requests": dict(self.active_requests),
            "provider_count": len(self.provider_preferences),
            "providers": {}
        }
        
        for provider, health in self.provider_health.items():
            if health.total_requests > 0:  # Only include providers that have been used
                stats["providers"][provider.value] = {
                    "total_requests": health.total_requests,
                    "successful_requests": health.successful_requests,
                    "failed_requests": health.failed_requests,
                    "success_rate": health.success_rate,
                    "error_rate": health.error_rate,
                    "is_healthy": health.is_healthy,
                    "last_check": health.last_check.isoformat(),
                    "active_requests": self.active_requests.get(provider, 0)
                }
        
        return stats
    
    def _get_next_request_id(self) -> str:
        """Generate a unique request ID"""
        self._request_id_counter += 1
        return f"req_{self._request_id_counter:08d}"
    
    async def test_provider_connectivity(self, provider: ProviderType) -> bool:
        """Test connectivity to a specific provider"""
        try:
            # Send a simple test request
            test_messages = [{"role": "user", "content": "Test connectivity"}]
            response = await self.api_manager.chat_completion(
                messages=test_messages,
                provider_type=provider,
                max_tokens=5  # Minimal response
            )
            
            success = response.success
            if success:
                self.logger.info(f"Connectivity test successful for provider: {provider.value}")
            else:
                self.logger.warning(f"Connectivity test failed for provider: {provider.value} - {response.error}")
            
            # Update health based on test result
            health = self.provider_health[provider]
            health.is_healthy = success
            health.last_check = datetime.now()
            
            return success
        except Exception as e:
            self.logger.error(f"Connectivity test error for provider {provider.value}: {str(e)}")
            health = self.provider_health[provider]
            health.is_healthy = False
            health.last_check = datetime.now()
            return False
    
    async def run_health_check_all_providers(self) -> Dict[ProviderType, bool]:
        """Run health checks for all registered providers"""
        results = {}
        
        for provider in self.provider_preferences.keys():
            results[provider] = await self.test_provider_connectivity(provider)
        
        return results
    
    async def close(self):
        """Close the provider router and clean up resources"""
        await self.api_manager.close_all()
        self.logger.info("Provider Router closed")


class AdvancedProviderRouter(ProviderRouter):
    """Advanced provider router with additional features"""
    
    def __init__(self):
        super().__init__()
        self.cost_optimizer_enabled = True
        self.performance_predictor_enabled = True
        self.adaptive_routing_enabled = True
        self.provider_cost_cache: Dict[ProviderType, float] = {}
        self.performance_history: Dict[ProviderType, List[Dict[str, Any]]] = {}
        
        # Initialize performance history for all providers
        for provider in ProviderType:
            self.performance_history[provider] = []
    
    async def route_request(
        self, 
        messages: List[Dict[str, str]], 
        preferred_provider: ProviderType = None,
        **kwargs
    ) -> APIResponse:
        """Advanced routing with cost optimization and performance prediction"""
        
        # If cost optimization is enabled, consider cost in provider selection
        if self.cost_optimizer_enabled and not preferred_provider:
            preferred_provider = await self._select_lowest_cost_provider(messages, **kwargs)
        
        # If performance prediction is enabled, consider performance in selection
        if self.performance_predictor_enabled and not preferred_provider:
            preferred_provider = await self._select_best_performance_provider(messages, **kwargs)
        
        # Use parent routing logic with the selected provider
        return await super().route_request(messages, preferred_provider, **kwargs)
    
    async def _select_lowest_cost_provider(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Optional[ProviderType]:
        """Select the provider with the lowest expected cost"""
        available_providers = [
            provider for provider in self.provider_preferences.keys()
            if self._is_provider_available(provider)
        ]
        
        if not available_providers:
            return None
        
        min_cost = float('inf')
        best_provider = available_providers[0]
        
        for provider in available_providers:
            # Estimate cost based on model and expected token usage
            model = kwargs.get("model", self.provider_preferences[provider].provider.value.split('_')[0] if self.provider_preferences[provider].provider.value else "default")
            estimated_tokens = len(" ".join([m.get("content", "") for m in messages])) // 4  # Rough estimate
            
            # Get cost from registry
            model_info = llm_provider_registry.get_model(model)
            if model_info:
                cost = llm_provider_registry.get_model_cost(model, estimated_tokens, estimated_tokens // 2)
            else:
                # Fallback cost estimation
                cost = estimated_tokens * 0.000001  # Placeholder cost per token
            
            if cost < min_cost:
                min_cost = cost
                best_provider = provider
        
        return best_provider
    
    async def _select_best_performance_provider(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Optional[ProviderType]:
        """Select the provider expected to perform best for this request"""
        available_providers = [
            provider for provider in self.provider_preferences.keys()
            if self._is_provider_available(provider)
        ]
        
        if not available_providers:
            return None
        
        # Analyze the request to determine what type of task it is
        content = " ".join([m.get("content", "") for m in messages])
        task_type = self._analyze_task_type(content)
        
        best_provider = available_providers[0]
        best_score = -1
        
        for provider in available_providers:
            score = await self._calculate_performance_score(provider, task_type, **kwargs)
            if score > best_score:
                best_score = score
                best_provider = provider
        
        return best_provider
    
    def _analyze_task_type(self, content: str) -> str:
        """Analyze the content to determine the task type"""
        content_lower = content.lower()
        
        if any(keyword in content_lower for keyword in ["code", "program", "function", "debug", "algorithm", "implementation"]):
            return "coding"
        elif any(keyword in content_lower for keyword in ["analyze", "analysis", "review", "evaluate", "compare", "assess"]):
            return "analysis"
        elif any(keyword in content_lower for keyword in ["creative", "story", "write", "generate", "create", "draft"]):
            return "creative"
        elif any(keyword in content_lower for keyword in ["math", "calculate", "equation", "formula", "compute", "solve"]):
            return "mathematical"
        elif any(keyword in content_lower for keyword in ["translate", "language", "linguistic", "interpret"]):
            return "translation"
        else:
            return "general"
    
    async def _calculate_performance_score(
        self, 
        provider: ProviderType, 
        task_type: str, 
        **kwargs
    ) -> float:
        """Calculate a performance score for a provider on a specific task type"""
        # Base score from historical performance
        historical_score = self._get_historical_performance_score(provider, task_type)
        
        # Factor in provider capabilities
        capability_score = self._get_capability_score(provider, task_type)
        
        # Factor in current health
        health = self.provider_health[provider]
        health_score = health.success_rate if health.total_requests > 0 else 0.8
        
        # Combine scores (weights can be adjusted)
        final_score = (historical_score * 0.4) + (capability_score * 0.4) + (health_score * 0.2)
        
        return final_score
    
    def _get_historical_performance_score(self, provider: ProviderType, task_type: str) -> float:
        """Get historical performance score for a provider on a task type"""
        history = self.performance_history[provider]
        if not history:
            return 0.7  # Default score if no history
        
        # Filter for similar task types
        relevant_history = [record for record in history 
                           if record.get("task_type") == task_type]
        
        if not relevant_history:
            # Use overall performance if no specific task history
            recent_history = history[-10:]  # Last 10 requests
            if not recent_history:
                return 0.7
            
            success_count = sum(1 for record in recent_history if record.get("success", False))
            return success_count / len(recent_history)
        
        # Calculate success rate for this task type
        success_count = sum(1 for record in relevant_history if record.get("success", False))
        return success_count / len(relevant_history) if relevant_history else 0.7
    
    def _get_capability_score(self, provider: ProviderType, task_type: str) -> float:
        """Get score based on provider capabilities for the task type"""
        capabilities = llm_provider_registry.get_provider_capabilities(provider)
        
        if task_type == "coding":
            return 0.9 if capabilities.get("supports_functions", False) else 0.6
        elif task_type == "creative":
            return 0.8  # Most providers handle creative tasks reasonably well
        elif task_type == "analysis":
            return 0.85 if capabilities.get("supports_tools", False) else 0.7
        elif task_type == "mathematical":
            return 0.8  # Most modern providers handle math well
        elif task_type == "translation":
            return 0.75  # Depends on model, but most handle it decently
        else:
            return 0.8  # General score
    
    def update_performance_history(
        self, 
        provider: ProviderType, 
        task_type: str, 
        success: bool, 
        response_time: float, 
        cost: float
    ):
        """Update performance history for a provider"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "success": success,
            "response_time": response_time,
            "cost": cost
        }
        
        self.performance_history[provider].append(record)
        
        # Keep only recent history
        if len(self.performance_history[provider]) > 100:
            self.performance_history[provider] = self.performance_history[provider][-100:]
    
    def enable_cost_optimization(self, enabled: bool = True):
        """Enable or disable cost optimization"""
        self.cost_optimizer_enabled = enabled
    
    def enable_performance_prediction(self, enabled: bool = True):
        """Enable or disable performance prediction"""
        self.performance_predictor_enabled = enabled
    
    def enable_adaptive_routing(self, enabled: bool = True):
        """Enable or disable adaptive routing"""
        self.adaptive_routing_enabled = enabled