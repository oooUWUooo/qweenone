"""
API Router for Agentic Architecture System
Provides intelligent routing between different LLM providers
"""
import asyncio
import json
import random
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..api_integrations import APIManager, APIConfig, APIResponse
from ..api_integrations.llm_provider import ProviderType, llm_provider_registry
from ..utils.logger import setup_logger


@dataclass
class RouteRule:
    """Rule for routing requests to specific providers"""
    name: str
    condition: str  # e.g., "model_contains", "task_type", "budget_under", etc.
    condition_value: Any
    provider: ProviderType
    priority: int = 1  # Lower number means higher priority
    enabled: bool = True
    description: str = ""


@dataclass
class RouteMetrics:
    """Metrics for a specific route/provider"""
    provider: ProviderType
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    last_used: datetime = field(default_factory=datetime.now)
    error_rate: float = 0.0


class RoutingStrategy(Enum):
    """Different strategies for routing requests"""
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    LEAST_COST = "least_cost"
    HIGHEST_RATED = "highest_rated"
    CUSTOM_RULES = "custom_rules"
    LOAD_BALANCED = "load_balanced"


class APIRouter:
    """Main API router that intelligently routes requests to different providers"""
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.CUSTOM_RULES):
        self.strategy = strategy
        self.logger = setup_logger("APIRouter")
        self.api_manager = APIManager()
        self.route_rules: List[RouteRule] = []
        self.route_metrics: Dict[ProviderType, RouteMetrics] = {}
        self.provider_weights: Dict[ProviderType, float] = {}
        self.request_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self._last_request_times: Dict[ProviderType, float] = {}
        self._concurrent_requests: Dict[ProviderType, int] = {}
        
        # Initialize metrics for known providers
        for provider in ProviderType:
            self.route_metrics[provider] = RouteMetrics(provider=provider)
    
    def register_provider(self, config: APIConfig) -> bool:
        """Register a provider with the underlying API manager"""
        success = self.api_manager.register_provider(config)
        if success:
            # Initialize metrics for this provider if not already done
            if config.provider not in self.route_metrics:
                self.route_metrics[config.provider] = RouteMetrics(provider=config.provider)
            # Initialize concurrent request counter
            self._concurrent_requests[config.provider] = 0
        return success
    
    def add_route_rule(self, rule: RouteRule):
        """Add a routing rule"""
        self.route_rules.append(rule)
        self.route_rules.sort(key=lambda r: r.priority)  # Sort by priority
        self.logger.info(f"Added route rule: {rule.name}")
    
    def remove_route_rule(self, rule_name: str) -> bool:
        """Remove a routing rule by name"""
        for i, rule in enumerate(self.route_rules):
            if rule.name == rule_name:
                del self.route_rules[i]
                self.logger.info(f"Removed route rule: {rule_name}")
                return True
        return False
    
    def set_provider_weight(self, provider: ProviderType, weight: float):
        """Set a weight for a provider (for weighted routing strategies)"""
        self.provider_weights[provider] = max(0.0, min(1.0, weight))  # Clamp between 0 and 1
    
    async def route_request(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> APIResponse:
        """Route a request to the most appropriate provider"""
        start_time = time.time()
        
        # Determine the target provider based on strategy and rules
        target_provider = await self._determine_target_provider(messages, **kwargs)
        
        if not target_provider:
            return APIResponse(
                success=False,
                error="No suitable provider found",
                status_code=500
            )
        
        # Track concurrent requests
        self._concurrent_requests[target_provider] += 1
        
        try:
            # Make the API call
            response = await self.api_manager.chat_completion(
                messages=messages,
                provider_type=target_provider,
                **kwargs
            )
            
            # Update metrics
            end_time = time.time()
            response_time = end_time - start_time
            
            self._update_metrics(
                target_provider, 
                response, 
                response_time,
                messages,
                **kwargs
            )
            
            # Add provider info to response
            if response.success:
                response.data = response.data or {}
                response.data["routing_info"] = {
                    "provider_used": target_provider.value,
                    "routing_strategy": self.strategy.value,
                    "response_time": response_time
                }
            
            return response
            
        finally:
            # Decrement concurrent request counter
            self._concurrent_requests[target_provider] -= 1
    
    async def _determine_target_provider(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Optional[ProviderType]:
        """Determine which provider to use based on routing strategy and rules"""
        
        # First, check custom rules
        if self.strategy == RoutingStrategy.CUSTOM_RULES:
            provider = await self._apply_custom_rules(messages, **kwargs)
            if provider:
                return provider
        
        # Get available providers
        available_providers = [p for p in self.api_manager.providers.keys() 
                              if self.api_manager.providers[p] is not None]
        
        if not available_providers:
            return None
        
        # Apply the selected strategy
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_route(available_providers)
        elif self.strategy == RoutingStrategy.LEAST_USED:
            return self._least_used_route(available_providers)
        elif self.strategy == RoutingStrategy.LEAST_COST:
            return self._least_cost_route(available_providers)
        elif self.strategy == RoutingStrategy.HIGHEST_RATED:
            return self._highest_rated_route(available_providers)
        elif self.strategy == RoutingStrategy.LOAD_BALANCED:
            return self._load_balanced_route(available_providers)
        elif self.strategy == RoutingStrategy.CUSTOM_RULES:
            # If custom rules didn't match, fall back to another strategy
            return self._load_balanced_route(available_providers)
        else:
            # Default to load balanced
            return self._load_balanced_route(available_providers)
    
    async def _apply_custom_rules(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Optional[ProviderType]:
        """Apply custom routing rules"""
        for rule in self.route_rules:
            if not rule.enabled:
                continue
            
            if self._rule_matches(rule, messages, **kwargs):
                return rule.provider
        
        return None
    
    def _rule_matches(
        self, 
        rule: RouteRule, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> bool:
        """Check if a rule matches the current request"""
        try:
            if rule.condition == "model_contains":
                model = kwargs.get("model", "")
                return rule.condition_value.lower() in model.lower()
            elif rule.condition == "task_type":
                # Look for task type in messages or kwargs
                content = " ".join([msg.get("content", "") for msg in messages])
                return rule.condition_value.lower() in content.lower()
            elif rule.condition == "budget_under":
                budget = kwargs.get("budget", float('inf'))
                return budget < rule.condition_value
            elif rule.condition == "provider_available":
                return rule.condition_value in self.api_manager.providers
            elif rule.condition == "capability_required":
                # Check if a specific capability is required
                required_cap = rule.condition_value
                for provider_type in self.api_manager.providers.keys():
                    provider_caps = llm_provider_registry.get_provider_capabilities(provider_type)
                    if required_cap in provider_caps.get("capabilities", []):
                        return True
                return False
            else:
                # For other conditions, return False for now
                return False
        except Exception:
            return False  # If there's an error evaluating the rule, skip it
    
    def _round_robin_route(self, providers: List[ProviderType]) -> ProviderType:
        """Round-robin routing strategy"""
        # Simple round-robin: select next provider in sequence
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        provider = providers[self._round_robin_index % len(providers)]
        self._round_robin_index += 1
        return provider
    
    def _least_used_route(self, providers: List[ProviderType]) -> ProviderType:
        """Route to the provider with the least requests"""
        min_requests = float('inf')
        selected_provider = providers[0]
        
        for provider in providers:
            metrics = self.route_metrics.get(provider, RouteMetrics(provider))
            if metrics.total_requests < min_requests:
                min_requests = metrics.total_requests
                selected_provider = provider
        
        return selected_provider
    
    def _least_cost_route(self, providers: List[ProviderType]) -> ProviderType:
        """Route to the provider with the lowest cost per token"""
        min_cost_per_token = float('inf')
        selected_provider = providers[0]
        
        for provider in providers:
            metrics = self.route_metrics.get(provider, RouteMetrics(provider))
            cost_per_token = (metrics.total_cost / metrics.total_tokens) if metrics.total_tokens > 0 else float('inf')
            
            if cost_per_token < min_cost_per_token:
                min_cost_per_token = cost_per_token
                selected_provider = provider
        
        return selected_provider
    
    def _highest_rated_route(self, providers: List[ProviderType]) -> ProviderType:
        """Route to the provider with the highest success rate"""
        max_success_rate = -1
        selected_provider = providers[0]
        
        for provider in providers:
            metrics = self.route_metrics.get(provider, RouteMetrics(provider))
            success_rate = (metrics.successful_requests / metrics.total_requests) if metrics.total_requests > 0 else 1.0
            
            if success_rate > max_success_rate:
                max_success_rate = success_rate
                selected_provider = provider
        
        return selected_provider
    
    def _load_balanced_route(self, providers: List[ProviderType]) -> ProviderType:
        """Load-balanced routing based on weights and current load"""
        # Consider weights if available
        weighted_providers = []
        
        for provider in providers:
            weight = self.provider_weights.get(provider, 1.0)
            # Adjust weight based on current load (fewer concurrent requests = higher effective weight)
            current_load = self._concurrent_requests.get(provider, 0)
            effective_weight = weight * (1.0 / (current_load + 1))  # Lower load = higher weight
            weighted_providers.append((provider, effective_weight))
        
        # Normalize weights
        total_weight = sum(weight for _, weight in weighted_providers)
        if total_weight <= 0:
            return providers[0] if providers else None
        
        # Select based on weighted random choice
        rand_val = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for provider, weight in weighted_providers:
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return provider
        
        # Fallback
        return providers[0] if providers else None
    
    def _update_metrics(
        self, 
        provider: ProviderType, 
        response: APIResponse, 
        response_time: float,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        """Update metrics for a provider"""
        metrics = self.route_metrics[provider]
        
        metrics.total_requests += 1
        
        if response.success:
            metrics.successful_requests += 1
            metrics.total_tokens += response.tokens_used
            metrics.total_cost += response.cost
            
            # Update average response time
            total_time = metrics.avg_response_time * (metrics.successful_requests - 1) + response_time
            metrics.avg_response_time = total_time / metrics.successful_requests
        else:
            metrics.failed_requests += 1
        
        metrics.last_used = datetime.now()
        metrics.error_rate = metrics.failed_requests / metrics.total_requests
        
        # Add to request history
        request_record = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider.value,
            "success": response.success,
            "tokens_used": response.tokens_used,
            "cost": response.cost,
            "response_time": response_time,
            "model": kwargs.get("model", "unknown"),
            "message_count": len(messages)
        }
        
        self.request_history.append(request_record)
        
        # Trim history if needed
        if len(self.request_history) > self.max_history:
            self.request_history = self.request_history[-self.max_history:]
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        stats = {
            "strategy": self.strategy.value,
            "total_requests": sum(m.total_requests for m in self.route_metrics.values()),
            "successful_requests": sum(m.successful_requests for m in self.route_metrics.values()),
            "failed_requests": sum(m.failed_requests for m in self.route_metrics.values()),
            "total_tokens": sum(m.total_tokens for m in self.route_metrics.values()),
            "total_cost": sum(m.total_cost for m in self.route_metrics.values()),
            "provider_metrics": {},
            "request_history_count": len(self.request_history),
            "active_providers": list(self.api_manager.providers.keys()),
            "route_rules_count": len(self.route_rules),
            "concurrent_requests": dict(self._concurrent_requests)
        }
        
        for provider, metrics in self.route_metrics.items():
            if metrics.total_requests > 0:  # Only include providers that have been used
                stats["provider_metrics"][provider.value] = {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "success_rate": metrics.successful_requests / metrics.total_requests,
                    "total_tokens": metrics.total_tokens,
                    "total_cost": metrics.total_cost,
                    "avg_response_time": metrics.avg_response_time,
                    "error_rate": metrics.error_rate,
                    "last_used": metrics.last_used.isoformat()
                }
        
        return stats
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations for optimizing routing"""
        recommendations = []
        stats = self.get_routing_statistics()
        
        # Check if any provider has high error rate
        for provider_name, metrics in stats["provider_metrics"].items():
            if metrics["error_rate"] > 0.1:  # More than 10% error rate
                recommendations.append(f"Provider {provider_name} has high error rate ({metrics['error_rate']:.2%}), consider reviewing configuration")
        
        # Check if routing is unbalanced
        if len(stats["provider_metrics"]) > 1:
            request_counts = [m["total_requests"] for m in stats["provider_metrics"].values()]
            if max(request_counts) / min(request_counts) > 3:  # Highly unbalanced
                recommendations.append("Request distribution is highly unbalanced, consider adjusting provider weights")
        
        # Check cost efficiency
        if stats["total_cost"] > 0:
            avg_cost_per_token = stats["total_cost"] / stats["total_tokens"] if stats["total_tokens"] > 0 else 0
            if avg_cost_per_token > 0.01:  # High cost per token
                recommendations.append(f"Average cost per token is high (${avg_cost_per_token:.6f}), consider more cost-effective models")
        
        if not recommendations:
            recommendations.append("Routing appears to be working well. No immediate optimizations needed.")
        
        return recommendations
    
    async def close(self):
        """Close the router and clean up resources"""
        await self.api_manager.close_all()
        self.logger.info("API Router closed")


class SmartRouter(APIRouter):
    """Enhanced router with machine learning-based routing decisions"""
    
    def __init__(self):
        super().__init__(strategy=RoutingStrategy.CUSTOM_RULES)
        self.performance_history: List[Dict[str, Any]] = []
        self.ml_model_trained = False
    
    async def route_request(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> APIResponse:
        """Route request using smart decision making"""
        # First, try to make a smart decision based on historical performance
        target_provider = await self._smart_route_decision(messages, **kwargs)
        
        if target_provider is None:
            # Fall back to parent routing logic
            return await super().route_request(messages, **kwargs)
        
        # Track concurrent requests
        self._concurrent_requests[target_provider] += 1
        
        try:
            # Make the API call
            response = await self.api_manager.chat_completion(
                messages=messages,
                provider_type=target_provider,
                **kwargs
            )
            
            # Update performance history
            self._update_performance_history(
                target_provider, 
                response, 
                messages,
                **kwargs
            )
            
            # Update metrics
            start_time = time.time() - (response.data.get("response_time", 0) if response.data else 0)
            end_time = time.time()
            response_time = end_time - start_time
            
            self._update_metrics(
                target_provider, 
                response, 
                response_time,
                messages,
                **kwargs
            )
            
            # Add provider info to response
            if response.success:
                response.data = response.data or {}
                response.data["routing_info"] = {
                    "provider_used": target_provider.value,
                    "routing_strategy": "smart_ml",
                    "response_time": response_time,
                    "confidence": 0.85  # Placeholder confidence score
                }
            
            return response
            
        finally:
            # Decrement concurrent request counter
            self._concurrent_requests[target_provider] -= 1
    
    async def _smart_route_decision(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Optional[ProviderType]:
        """Make a smart routing decision based on context and history"""
        # Check custom rules first
        provider = await self._apply_custom_rules(messages, **kwargs)
        if provider:
            return provider
        
        # Analyze the request to determine requirements
        request_analysis = self._analyze_request(messages, **kwargs)
        
        # Find the best provider based on historical performance for similar requests
        best_provider = self._find_best_provider_for_request(request_analysis)
        
        return best_provider
    
    def _analyze_request(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Analyze a request to determine its characteristics"""
        content = " ".join([msg.get("content", "") for msg in messages])
        
        analysis = {
            "length": len(content),
            "complexity": self._estimate_complexity(content),
            "task_type": self._infer_task_type(content),
            "required_capabilities": kwargs.get("required_capabilities", []),
            "model_preference": kwargs.get("model", ""),
            "budget_constraint": kwargs.get("budget", float('inf')),
            "response_time_constraint": kwargs.get("max_response_time", float('inf'))
        }
        
        return analysis
    
    def _estimate_complexity(self, content: str) -> str:
        """Estimate the complexity of a request"""
        words = content.split()
        if len(words) < 50:
            return "simple"
        elif len(words) < 200:
            return "medium"
        else:
            return "complex"
    
    def _infer_task_type(self, content: str) -> str:
        """Infer the task type from content"""
        content_lower = content.lower()
        
        if any(keyword in content_lower for keyword in ["code", "program", "function", "algorithm"]):
            return "coding"
        elif any(keyword in content_lower for keyword in ["analyze", "analysis", "review", "evaluate"]):
            return "analysis"
        elif any(keyword in content_lower for keyword in ["creative", "story", "write", "generate"]):
            return "creative"
        elif any(keyword in content_lower for keyword in ["math", "calculate", "equation", "formula"]):
            return "mathematical"
        else:
            return "general"
    
    def _find_best_provider_for_request(self, request_analysis: Dict[str, Any]) -> Optional[ProviderType]:
        """Find the best provider for a specific request based on historical performance"""
        # This is a simplified implementation - in a real system, you'd use ML models
        task_type = request_analysis["task_type"]
        
        # Simple heuristic based on task type
        if task_type == "coding":
            # Look for providers with good coding performance in history
            for provider_type in self.api_manager.providers.keys():
                provider_caps = llm_provider_registry.get_provider_capabilities(provider_type)
                if provider_caps.get("supports_functions", False):
                    return provider_type
        elif task_type == "creative":
            # For creative tasks, prioritize providers with good creative performance
            for provider_type in [ProviderType.ANTHROPIC, ProviderType.OPENAI, ProviderType.OPENROUTER]:
                if provider_type in self.api_manager.providers:
                    return provider_type
        elif task_type == "mathematical":
            # For mathematical tasks, prioritize accuracy
            for provider_type in [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.OPENROUTER]:
                if provider_type in self.api_manager.providers:
                    return provider_type
        
        # Default to the first available provider
        available_providers = list(self.api_manager.providers.keys())
        return available_providers[0] if available_providers else None
    
    def _update_performance_history(
        self, 
        provider: ProviderType, 
        response: APIResponse, 
        messages: List[Dict[str, str]], 
        **kwargs
    ):
        """Update the performance history for ML model training"""
        request_analysis = self._analyze_request(messages, **kwargs)
        
        performance_record = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider.value,
            "request_analysis": request_analysis,
            "response_success": response.success,
            "response_time": response.data.get("response_time", 0) if response.data else 0,
            "tokens_used": response.tokens_used,
            "cost": response.cost,
            "model_used": kwargs.get("model", "unknown")
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]