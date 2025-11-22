"""
Router Configuration for Agentic Architecture System
Manages configuration for API routing and provider management
"""
import asyncio
import json
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..api_integrations import APIConfig
from ..api_integrations.llm_provider import ProviderType
from ..utils.logger import setup_logger


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for router"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME_BASED = "response_time_based"
    PRIORITY_BASED = "priority_based"
    COST_OPTIMIZED = "cost_optimized"


class FailoverStrategy(Enum):
    """Failover strategies for router"""
    IMMEDIATE_FAILOVER = "immediate_failover"
    DELAYED_FAILOVER = "delayed_failover"
    GRADUAL_FAILOVER = "gradual_failover"
    NO_FAILOVER = "no_failover"


@dataclass
class RouterProviderConfig:
    """Configuration for a specific provider within the router"""
    provider_type: ProviderType
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority
    weight: float = 1.0  # For weighted load balancing
    max_concurrent_requests: int = 10
    max_requests_per_minute: int = 100
    timeout: int = 30
    retry_attempts: int = 3
    fallback_providers: List[ProviderType] = field(default_factory=list)
    health_check_interval: int = 60  # seconds
    circuit_breaker_threshold: float = 0.5  # Error rate threshold
    circuit_breaker_timeout: int = 300  # seconds
    custom_headers: Dict[str, str] = field(default_factory=dict)
    model_preferences: List[str] = field(default_factory=list)
    cost_multiplier: float = 1.0  # Multiplier for cost calculations
    performance_multiplier: float = 1.0  # Multiplier for performance calculations


@dataclass
class RouterConfig:
    """Main configuration for the API router"""
    # Basic routing settings
    default_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    failover_strategy: FailoverStrategy = FailoverStrategy.IMMEDIATE_FAILOVER
    health_check_enabled: bool = True
    health_check_interval: int = 30  # seconds
    circuit_breaker_enabled: bool = True
    request_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Performance and optimization settings
    cost_optimization_enabled: bool = True
    performance_prediction_enabled: bool = True
    adaptive_routing_enabled: bool = True
    load_balancing_enabled: bool = True
    
    # Provider-specific configurations
    provider_configs: List[RouterProviderConfig] = field(default_factory=list)
    
    # Fallback and redundancy settings
    enable_fallback: bool = True
    max_fallback_attempts: int = 2
    fallback_timeout: int = 15
    
    # Monitoring and logging
    log_level: str = "INFO"
    enable_request_logging: bool = True
    enable_response_logging: bool = False
    enable_performance_monitoring: bool = True
    metrics_retention_days: int = 30
    
    # Security and rate limiting
    enable_rate_limiting: bool = True
    default_rate_limit: int = 1000  # requests per hour
    enable_api_key_validation: bool = True
    trusted_proxies: List[str] = field(default_factory=list)
    
    # Advanced settings
    custom_routing_rules: List[Dict[str, Any]] = field(default_factory=list)
    provider_weights: Dict[ProviderType, float] = field(default_factory=dict)
    model_routing_rules: Dict[str, ProviderType] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize configuration after creation"""
        if not self.provider_configs:
            # Add default configurations for common providers
            default_configs = [
                RouterProviderConfig(
                    provider_type=ProviderType.OPENROUTER,
                    enabled=True,
                    priority=1,
                    weight=1.0,
                    max_concurrent_requests=5,
                    max_requests_per_minute=100
                ),
                RouterProviderConfig(
                    provider_type=ProviderType.OPENAI,
                    enabled=True,
                    priority=2,
                    weight=0.8,
                    max_concurrent_requests=3,
                    max_requests_per_minute=100
                ),
                RouterProviderConfig(
                    provider_type=ProviderType.ANTHROPIC,
                    enabled=True,
                    priority=2,
                    weight=0.8,
                    max_concurrent_requests=3,
                    max_requests_per_minute=100
                )
            ]
            self.provider_configs = default_configs


class RouterConfigManager:
    """Manages router configuration"""
    
    def __init__(self, config: RouterConfig = None):
        self.config = config or RouterConfig()
        self.logger = setup_logger("RouterConfigManager")
        self.config_file_path = "/workspace/config/router_config.json"
        self._last_modified = datetime.now()
    
    def get_provider_config(self, provider_type: ProviderType) -> Optional[RouterProviderConfig]:
        """Get configuration for a specific provider"""
        for provider_config in self.config.provider_configs:
            if provider_config.provider_type == provider_type:
                return provider_config
        return None
    
    def set_provider_config(self, provider_config: RouterProviderConfig):
        """Set configuration for a specific provider"""
        # Remove existing config for this provider
        self.config.provider_configs = [
            pc for pc in self.config.provider_configs
            if pc.provider_type != provider_config.provider_type
        ]
        # Add the new config
        self.config.provider_configs.append(provider_config)
        self.logger.info(f"Updated configuration for provider: {provider_config.provider_type.value}")
    
    def enable_provider(self, provider_type: ProviderType, enabled: bool = True):
        """Enable or disable a provider"""
        provider_config = self.get_provider_config(provider_type)
        if provider_config:
            provider_config.enabled = enabled
            status = "enabled" if enabled else "disabled"
            self.logger.info(f"Provider {provider_type.value} {status}")
        else:
            self.logger.warning(f"No configuration found for provider: {provider_type.value}")
    
    def set_provider_priority(self, provider_type: ProviderType, priority: int):
        """Set priority for a provider"""
        provider_config = self.get_provider_config(provider_type)
        if provider_config:
            provider_config.priority = priority
            self.logger.info(f"Set priority {priority} for provider: {provider_type.value}")
        else:
            self.logger.warning(f"No configuration found for provider: {provider_type.value}")
    
    def add_fallback_provider(self, primary: ProviderType, fallback: ProviderType):
        """Add a fallback provider for the primary provider"""
        primary_config = self.get_provider_config(primary)
        if primary_config:
            if fallback not in primary_config.fallback_providers:
                primary_config.fallback_providers.append(fallback)
                self.logger.info(f"Added fallback provider {fallback.value} for {primary.value}")
        else:
            self.logger.warning(f"No configuration found for primary provider: {primary.value}")
    
    def remove_fallback_provider(self, primary: ProviderType, fallback: ProviderType):
        """Remove a fallback provider for the primary provider"""
        primary_config = self.get_provider_config(primary)
        if primary_config and fallback in primary_config.fallback_providers:
            primary_config.fallback_providers.remove(fallback)
            self.logger.info(f"Removed fallback provider {fallback.value} for {primary.value}")
    
    def set_provider_weight(self, provider_type: ProviderType, weight: float):
        """Set weight for a provider (for weighted load balancing)"""
        provider_config = self.get_provider_config(provider_type)
        if provider_config:
            provider_config.weight = max(0.0, min(10.0, weight))  # Clamp between 0 and 10
            self.logger.info(f"Set weight {weight} for provider: {provider_type.value}")
        else:
            self.logger.warning(f"No configuration found for provider: {provider_type.value}")
    
    def set_model_routing_rule(self, model_name: str, provider_type: ProviderType):
        """Set a routing rule for a specific model"""
        self.config.model_routing_rules[model_name] = provider_type
        self.logger.info(f"Set routing rule: model '{model_name}' -> provider '{provider_type.value}'")
    
    def remove_model_routing_rule(self, model_name: str):
        """Remove a routing rule for a specific model"""
        if model_name in self.config.model_routing_rules:
            del self.config.model_routing_rules[model_name]
            self.logger.info(f"Removed routing rule for model: {model_name}")
    
    def add_custom_routing_rule(self, rule: Dict[str, Any]):
        """Add a custom routing rule"""
        self.config.custom_routing_rules.append(rule)
        self.logger.info(f"Added custom routing rule: {rule.get('name', 'unnamed')}")
    
    def remove_custom_routing_rule(self, rule_name: str):
        """Remove a custom routing rule by name"""
        for i, rule in enumerate(self.config.custom_routing_rules):
            if rule.get("name") == rule_name:
                del self.config.custom_routing_rules[i]
                self.logger.info(f"Removed custom routing rule: {rule_name}")
                return True
        return False
    
    def get_effective_provider_configs(self) -> List[RouterProviderConfig]:
        """Get list of enabled provider configurations"""
        return [pc for pc in self.config.provider_configs if pc.enabled]
    
    def validate_config(self) -> List[str]:
        """Validate the configuration and return any issues"""
        issues = []
        
        # Check for duplicate priorities
        priorities = [pc.priority for pc in self.config.provider_configs if pc.enabled]
        if len(priorities) != len(set(priorities)):
            issues.append("Duplicate priorities found among enabled providers")
        
        # Check provider weights
        for pc in self.config.provider_configs:
            if pc.weight <= 0:
                issues.append(f"Provider {pc.provider_type.value} has non-positive weight: {pc.weight}")
        
        # Check rate limits
        if self.config.default_rate_limit <= 0:
            issues.append("Default rate limit must be positive")
        
        # Check timeouts
        if self.config.request_timeout <= 0:
            issues.append("Request timeout must be positive")
        if self.config.health_check_interval <= 0:
            issues.append("Health check interval must be positive")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {
            "default_strategy": self.config.default_strategy.value,
            "failover_strategy": self.config.failover_strategy.value,
            "health_check_enabled": self.config.health_check_enabled,
            "health_check_interval": self.config.health_check_interval,
            "circuit_breaker_enabled": self.config.circuit_breaker_enabled,
            "request_timeout": self.config.request_timeout,
            "max_retries": self.config.max_retries,
            "retry_delay": self.config.retry_delay,
            "cost_optimization_enabled": self.config.cost_optimization_enabled,
            "performance_prediction_enabled": self.config.performance_prediction_enabled,
            "adaptive_routing_enabled": self.config.adaptive_routing_enabled,
            "load_balancing_enabled": self.config.load_balancing_enabled,
            "enable_fallback": self.config.enable_fallback,
            "max_fallback_attempts": self.config.max_fallback_attempts,
            "fallback_timeout": self.config.fallback_timeout,
            "log_level": self.config.log_level,
            "enable_request_logging": self.config.enable_request_logging,
            "enable_response_logging": self.config.enable_response_logging,
            "enable_performance_monitoring": self.config.enable_performance_monitoring,
            "metrics_retention_days": self.config.metrics_retention_days,
            "enable_rate_limiting": self.config.enable_rate_limiting,
            "default_rate_limit": self.config.default_rate_limit,
            "enable_api_key_validation": self.config.enable_api_key_validation,
            "trusted_proxies": self.config.trusted_proxies,
            "custom_routing_rules": self.config.custom_routing_rules,
            "model_routing_rules": {k: v.value for k, v in self.config.model_routing_rules.items()},
            "provider_configs": []
        }
        
        # Convert provider configs
        for pc in self.config.provider_configs:
            pc_dict = {
                "provider_type": pc.provider_type.value,
                "enabled": pc.enabled,
                "priority": pc.priority,
                "weight": pc.weight,
                "max_concurrent_requests": pc.max_concurrent_requests,
                "max_requests_per_minute": pc.max_requests_per_minute,
                "timeout": pc.timeout,
                "retry_attempts": pc.retry_attempts,
                "fallback_providers": [fp.value for fp in pc.fallback_providers],
                "health_check_interval": pc.health_check_interval,
                "circuit_breaker_threshold": pc.circuit_breaker_threshold,
                "circuit_breaker_timeout": pc.circuit_breaker_timeout,
                "custom_headers": pc.custom_headers,
                "model_preferences": pc.model_preferences,
                "cost_multiplier": pc.cost_multiplier,
                "performance_multiplier": pc.performance_multiplier
            }
            result["provider_configs"].append(pc_dict)
        
        return result
    
    def from_dict(self, data: Dict[str, Any]):
        """Load configuration from dictionary"""
        # Update basic settings
        self.config.default_strategy = LoadBalancingStrategy(data.get("default_strategy", "round_robin"))
        self.config.failover_strategy = FailoverStrategy(data.get("failover_strategy", "immediate_failover"))
        self.config.health_check_enabled = data.get("health_check_enabled", True)
        self.config.health_check_interval = data.get("health_check_interval", 30)
        self.config.circuit_breaker_enabled = data.get("circuit_breaker_enabled", True)
        self.config.request_timeout = data.get("request_timeout", 60)
        self.config.max_retries = data.get("max_retries", 3)
        self.config.retry_delay = data.get("retry_delay", 1.0)
        self.config.cost_optimization_enabled = data.get("cost_optimization_enabled", True)
        self.config.performance_prediction_enabled = data.get("performance_prediction_enabled", True)
        self.config.adaptive_routing_enabled = data.get("adaptive_routing_enabled", True)
        self.config.load_balancing_enabled = data.get("load_balancing_enabled", True)
        self.config.enable_fallback = data.get("enable_fallback", True)
        self.config.max_fallback_attempts = data.get("max_fallback_attempts", 2)
        self.config.fallback_timeout = data.get("fallback_timeout", 15)
        self.config.log_level = data.get("log_level", "INFO")
        self.config.enable_request_logging = data.get("enable_request_logging", True)
        self.config.enable_response_logging = data.get("enable_response_logging", False)
        self.config.enable_performance_monitoring = data.get("enable_performance_monitoring", True)
        self.config.metrics_retention_days = data.get("metrics_retention_days", 30)
        self.config.enable_rate_limiting = data.get("enable_rate_limiting", True)
        self.config.default_rate_limit = data.get("default_rate_limit", 1000)
        self.config.enable_api_key_validation = data.get("enable_api_key_validation", True)
        self.config.trusted_proxies = data.get("trusted_proxies", [])
        self.config.custom_routing_rules = data.get("custom_routing_rules", [])
        self.config.model_routing_rules = {
            k: ProviderType(v) for k, v in data.get("model_routing_rules", {}).items()
        }
        
        # Update provider configs
        self.config.provider_configs = []
        for pc_data in data.get("provider_configs", []):
            pc = RouterProviderConfig(
                provider_type=ProviderType(pc_data["provider_type"]),
                enabled=pc_data.get("enabled", True),
                priority=pc_data.get("priority", 1),
                weight=pc_data.get("weight", 1.0),
                max_concurrent_requests=pc_data.get("max_concurrent_requests", 10),
                max_requests_per_minute=pc_data.get("max_requests_per_minute", 100),
                timeout=pc_data.get("timeout", 30),
                retry_attempts=pc_data.get("retry_attempts", 3),
                fallback_providers=[ProviderType(fp) for fp in pc_data.get("fallback_providers", [])],
                health_check_interval=pc_data.get("health_check_interval", 60),
                circuit_breaker_threshold=pc_data.get("circuit_breaker_threshold", 0.5),
                circuit_breaker_timeout=pc_data.get("circuit_breaker_timeout", 300),
                custom_headers=pc_data.get("custom_headers", {}),
                model_preferences=pc_data.get("model_preferences", []),
                cost_multiplier=pc_data.get("cost_multiplier", 1.0),
                performance_multiplier=pc_data.get("performance_multiplier", 1.0)
            )
            self.config.provider_configs.append(pc)
    
    async def save_to_file(self, file_path: str = None):
        """Save configuration to file"""
        path = file_path or self.config_file_path
        try:
            config_dict = self.to_dict()
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            self.logger.info(f"Configuration saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {path}: {str(e)}")
            raise
    
    async def load_from_file(self, file_path: str = None):
        """Load configuration from file"""
        path = file_path or self.config_file_path
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            self.from_dict(config_dict)
            self.logger.info(f"Configuration loaded from {path}")
        except FileNotFoundError:
            self.logger.warning(f"Configuration file not found: {path}. Using default configuration.")
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {path}: {str(e)}")
            raise
    
    def get_optimized_config_for_task(self, task_requirements: Dict[str, Any]) -> RouterConfig:
        """Get an optimized configuration based on task requirements"""
        # Create a copy of the current config
        optimized_config = RouterConfig(**{k: v for k, v in self.config.__dict__.items()})
        
        # Adjust based on task requirements
        required_capabilities = task_requirements.get("capabilities", [])
        budget_constraint = task_requirements.get("budget", float('inf'))
        response_time_constraint = task_requirements.get("max_response_time", float('inf'))
        
        # Adjust provider priorities based on capabilities needed
        for provider_config in optimized_config.provider_configs:
            capability_bonus = 0
            for capability in required_capabilities:
                # This is a simplified check - in reality, you'd have more detailed capability mapping
                if capability.lower() in ["coding", "functions"] and provider_config.provider_type in [ProviderType.OPENAI, ProviderType.ANTHROPIC]:
                    capability_bonus += 0.2
                elif capability.lower() in ["creative", "text"] and provider_config.provider_type in [ProviderType.ANTHROPIC, ProviderType.OPENROUTER]:
                    capability_bonus += 0.1
            
            # Adjust weight based on budget if needed
            if budget_constraint < float('inf'):
                # Prefer lower-cost providers
                if provider_config.provider_type in [ProviderType.OPENROUTER]:
                    provider_config.cost_multiplier *= 0.8  # Slight preference for OpenRouter for budget-conscious tasks
            
            # Adjust priority based on response time requirements
            if response_time_constraint < 5:  # Less than 5 seconds
                # Prioritize faster providers
                provider_config.priority = max(1, provider_config.priority - 1)
        
        return optimized_config
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security-related configuration"""
        return {
            "enable_api_key_validation": self.config.enable_api_key_validation,
            "enable_rate_limiting": self.config.enable_rate_limiting,
            "default_rate_limit": self.config.default_rate_limit,
            "trusted_proxies": self.config.trusted_proxies,
            "request_timeout": self.config.request_timeout,
            "max_retries": self.config.max_retries
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration"""
        return {
            "load_balancing_enabled": self.config.load_balancing_enabled,
            "cost_optimization_enabled": self.config.cost_optimization_enabled,
            "performance_prediction_enabled": self.config.performance_prediction_enabled,
            "adaptive_routing_enabled": self.config.adaptive_routing_enabled,
            "circuit_breaker_enabled": self.config.circuit_breaker_enabled,
            "health_check_enabled": self.config.health_check_enabled,
            "health_check_interval": self.config.health_check_interval,
            "default_strategy": self.config.default_strategy.value
        }


class ConfigurableAPIRouter:
    """API router that uses the configuration system"""
    
    def __init__(self, config: RouterConfig = None):
        self.config_manager = RouterConfigManager(config)
        self.logger = setup_logger("ConfigurableAPIRouter")
        self.is_running = False
    
    async def initialize(self):
        """Initialize the router with configuration"""
        # Load configuration from file if it exists
        await self.config_manager.load_from_file()
        
        # Validate configuration
        issues = self.config_manager.validate_config()
        if issues:
            self.logger.warning(f"Configuration issues found: {issues}")
        
        self.is_running = True
        self.logger.info("Configurable API Router initialized")
    
    def get_config(self) -> RouterConfig:
        """Get the current router configuration"""
        return self.config_manager.config
    
    def update_config(self, new_config: RouterConfig):
        """Update the router configuration"""
        self.config_manager.config = new_config
        self.logger.info("Router configuration updated")
    
    def get_provider_config(self, provider_type: ProviderType) -> Optional[RouterProviderConfig]:
        """Get configuration for a specific provider"""
        return self.config_manager.get_provider_config(provider_type)
    
    def set_provider_config(self, provider_config: RouterProviderConfig):
        """Set configuration for a specific provider"""
        self.config_manager.set_provider_config(provider_config)
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.config_manager.get_security_config()
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.config_manager.get_performance_config()
    
    async def save_config(self):
        """Save current configuration to file"""
        await self.config_manager.save_to_file()
    
    async def shutdown(self):
        """Shutdown the router"""
        self.is_running = False
        await self.save_config()
        self.logger.info("Configurable API Router shutdown")