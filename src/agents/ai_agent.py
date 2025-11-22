"""
AI Agent module for Agentic Architecture System
Implements agents that can use various LLM providers including OpenRouter
"""
import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

from ..api_integrations import APIManager, APIConfig, APIResponse
from ..api_integrations.llm_provider import ProviderType, llm_provider_registry
from ..utils.logger import setup_logger
from .base_agent import BaseAgent, AgentCapability, AgentStatus


class AIAgentConfig:
    """Configuration for AI agents"""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        provider_type: ProviderType = ProviderType.OPENROUTER,
        model: str = "openchat/openchat-7b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key: str = "",
        base_url: str = "",
        timeout: int = 30,
        max_retries: int = 3,
        capabilities: List[AgentCapability] = None,
        system_prompt: str = "",
        memory_size: int = 100,
        context_window: int = 4096
    ):
        self.name = name
        self.description = description
        self.provider_type = provider_type
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.capabilities = capabilities or []
        self.system_prompt = system_prompt
        self.memory_size = memory_size
        self.context_window = context_window


class ConversationMemory:
    """Manages conversation memory for AI agents"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.messages: List[Dict[str, Any]] = []
        self.turn_count = 0
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the conversation memory"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "turn": self.turn_count,
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        self.turn_count += 1
        
        # Trim memory if it exceeds max size
        if len(self.messages) > self.max_size:
            self.messages = self.messages[-self.max_size:]
    
    def get_context(self, max_tokens: int = 3000) -> List[Dict[str, str]]:
        """Get recent conversation context, respecting token limits"""
        # This is a simplified approach - in a real implementation, 
        # we'd count actual tokens rather than just message count
        return self.messages[-10:]  # Return last 10 messages as approximation
    
    def clear(self):
        """Clear all conversation memory"""
        self.messages = []
        self.turn_count = 0
    
    def get_summary(self) -> str:
        """Get a summary of the conversation"""
        if not self.messages:
            return "No conversation history"
        
        # Get the first few and last few messages as a summary
        start_messages = self.messages[:3] if len(self.messages) > 6 else self.messages[:len(self.messages)//2]
        end_messages = self.messages[-3:] if len(self.messages) > 6 else self.messages[len(self.messages)//2:]
        
        summary_parts = []
        for msg in start_messages + end_messages:
            summary_parts.append(f"{msg['role']}: {msg['content'][:100]}...")
        
        return " | ".join(summary_parts)


class AIAgent(BaseAgent):
    """Base class for AI-powered agents that use LLMs"""
    
    def __init__(self, config: AIAgentConfig):
        super().__init__(config.name, config.description)
        self.config = config
        self.agent_type = "ai_agent"
        self.api_manager = None
        self.conversation_memory = ConversationMemory(config.memory_size)
        self.system_prompt = config.system_prompt
        self.logger = setup_logger(f"AIAgent_{self.id}")
        
        # Add the capabilities from config
        for capability in config.capabilities:
            self.add_capability(capability)
        
        # Initialize API manager
        self._initialize_api_manager()
    
    def _initialize_api_manager(self):
        """Initialize the API manager with the configured provider"""
        self.api_manager = APIManager()
        
        api_config = APIConfig(
            provider=self.config.provider_type,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        success = self.api_manager.register_provider(api_config)
        if not success:
            self.logger.error(f"Failed to register API provider: {self.config.provider_type}")
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a task using the AI model"""
        self.update_status(AgentStatus.WORKING)
        
        try:
            # Prepare the prompt for the AI model
            prompt = self._prepare_prompt(task)
            
            # Add to conversation memory
            self.conversation_memory.add_message("user", prompt, {"task_id": task.get("id")})
            
            # Call the AI model
            response = await self._call_ai_model(prompt)
            
            if not response.success:
                self.logger.error(f"AI model call failed: {response.error}")
                self.update_status(AgentStatus.ERROR)
                return {"error": response.error, "status": "failed"}
            
            # Add AI response to conversation memory
            ai_response = response.data.get("choices", [{}])[0].get("message", {}).get("content", "")
            self.conversation_memory.add_message("assistant", ai_response, {"task_id": task.get("id")})
            
            # Format the result
            result = {
                "status": "completed",
                "agent_id": self.id,
                "task_id": task.get("id"),
                "result": ai_response,
                "tokens_used": response.tokens_used,
                "cost": response.cost,
                "model_used": self.config.model,
                "provider": self.config.provider_type.value
            }
            
            self.update_status(AgentStatus.IDLE)
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            self.update_status(AgentStatus.ERROR)
            return {"error": str(e), "status": "failed"}
    
    def _prepare_prompt(self, task: Dict[str, Any]) -> str:
        """Prepare the prompt for the AI model based on the task"""
        task_description = task.get("description", task.get("title", ""))
        task_params = task.get("params", {})
        
        # Build the prompt
        prompt_parts = []
        
        # Add system context if available
        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}")
        
        # Add task description
        prompt_parts.append(f"Task: {task_description}")
        
        # Add task parameters if any
        if task_params:
            prompt_parts.append(f"Parameters: {json.dumps(task_params, indent=2)}")
        
        # Add any additional context from memory
        recent_context = self.conversation_memory.get_context()
        if recent_context:
            context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_context[-3:]])
            if context_str.strip():
                prompt_parts.append(f"Context: {context_str}")
        
        # Add specific instructions based on agent capabilities
        capabilities_str = ", ".join([cap.value for cap in self.capabilities])
        prompt_parts.append(f"Capabilities: {capabilities_str}")
        prompt_parts.append("Please provide a detailed response addressing the task.")
        
        return "\n\n".join(prompt_parts)
    
    async def _call_ai_model(self, prompt: str) -> APIResponse:
        """Call the AI model with the prepared prompt"""
        messages = [
            {"role": "system", "content": self.system_prompt or f"You are {self.name}, an AI agent."}
        ]
        
        # Add conversation history
        context_messages = self.conversation_memory.get_context()
        messages.extend(context_messages)
        
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Call the API manager
        return await self.api_manager.chat_completion(
            messages=messages,
            provider_type=self.config.provider_type,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
    
    async def chat(self, message: str) -> str:
        """Have a conversation with the AI agent"""
        self.update_status(AgentStatus.WORKING)
        
        try:
            # Add user message to memory
            self.conversation_memory.add_message("user", message)
            
            # Prepare messages for AI
            messages = [
                {"role": "system", "content": self.system_prompt or f"You are {self.name}, an AI agent."}
            ]
            
            # Add conversation history
            context_messages = self.conversation_memory.get_context()
            messages.extend(context_messages)
            
            # Call the AI model
            response = await self.api_manager.chat_completion(
                messages=messages,
                provider_type=self.config.provider_type,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            if not response.success:
                self.logger.error(f"Chat failed: {response.error}")
                return f"Error: {response.error}"
            
            # Extract the AI response
            ai_response = response.data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Add AI response to memory
            self.conversation_memory.add_message("assistant", ai_response)
            
            self.update_status(AgentStatus.IDLE)
            return ai_response
            
        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            self.update_status(AgentStatus.ERROR)
            return f"Error: {str(e)}"
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation history"""
        return self.conversation_memory.get_summary()
    
    async def close(self):
        """Clean up resources"""
        if self.api_manager:
            await self.api_manager.close_all()
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        if self.api_manager:
            return self.api_manager.get_statistics()
        return {}


class SpecializedAIAgent(AIAgent):
    """Specialized AI agent for specific tasks"""
    
    def __init__(self, config: AIAgentConfig, specialization: str = "general"):
        super().__init__(config)
        self.specialization = specialization
        self.task_history: List[Dict[str, Any]] = []
        
        # Update system prompt based on specialization
        self._update_system_prompt()
    
    def _update_system_prompt(self):
        """Update the system prompt based on specialization"""
        base_prompt = f"You are {self.name}, a specialized AI agent."
        
        if self.specialization == "code":
            self.system_prompt = (
                f"{base_prompt} You are an expert programmer. "
                "You write clean, efficient, and well-documented code. "
                "You follow best practices and consider edge cases. "
                "You provide explanations for your code when appropriate."
            )
        elif self.specialization == "research":
            self.system_prompt = (
                f"{base_prompt} You are an expert researcher. "
                "You analyze information critically, synthesize data from multiple sources, "
                "and provide evidence-based conclusions. You cite sources when possible."
            )
        elif self.specialization == "analysis":
            self.system_prompt = (
                f"{base_prompt} You are an expert analyst. "
                "You break down complex problems, identify patterns, "
                "and provide data-driven insights. You consider multiple perspectives."
            )
        elif self.specialization == "planning":
            self.system_prompt = (
                f"{base_prompt} You are an expert planner. "
                "You create detailed, actionable plans with clear milestones. "
                "You consider dependencies, resources, and potential risks."
            )
        else:
            self.system_prompt = base_prompt
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a task with specialization-specific handling"""
        # Add task to history
        task_record = {
            "id": task.get("id", str(uuid.uuid4())),
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "specialization": self.specialization
        }
        self.task_history.append(task_record)
        
        # Execute the task using parent method
        result = await super().execute_task(task)
        
        # Add result to task record
        task_record["result"] = result
        
        return result
    
    def get_specialization_insights(self) -> Dict[str, Any]:
        """Get insights based on the agent's specialization and task history"""
        if not self.task_history:
            return {"message": "No task history available"}
        
        # Count completed tasks
        completed_tasks = [t for t in self.task_history if t.get("result", {}).get("status") == "completed"]
        
        insights = {
            "specialization": self.specialization,
            "total_tasks": len(self.task_history),
            "completed_tasks": len(completed_tasks),
            "success_rate": len(completed_tasks) / len(self.task_history) if self.task_history else 0,
            "recent_tasks": [t["task"].get("title", "Unknown") for t in self.task_history[-5:]]
        }
        
        return insights


class MultiModelAIAgent(AIAgent):
    """AI agent that can switch between different models/providers"""
    
    def __init__(self, configs: List[AIAgentConfig]):
        if not configs:
            raise ValueError("At least one configuration is required")
        
        # Use the first config for basic initialization
        super().__init__(configs[0])
        self.configs = {config.provider_type: config for config in configs}
        self.current_provider = configs[0].provider_type
        
        # Initialize API manager with all providers
        self._initialize_multi_provider_api_manager()
    
    def _initialize_multi_provider_api_manager(self):
        """Initialize API manager with multiple providers"""
        self.api_manager = APIManager()
        
        for config in self.configs.values():
            api_config = APIConfig(
                provider=config.provider_type,
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=config.max_retries,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            self.api_manager.register_provider(api_config)
    
    async def switch_provider(self, provider_type: ProviderType) -> bool:
        """Switch to a different provider"""
        if provider_type not in self.configs:
            return False
        
        self.current_provider = provider_type
        self.config = self.configs[provider_type]  # Update current config
        return True
    
    async def execute_task_with_provider(self, task: Dict[str, Any], provider_type: ProviderType) -> Any:
        """Execute a task with a specific provider"""
        # Temporarily switch provider
        original_provider = self.current_provider
        await self.switch_provider(provider_type)
        
        try:
            result = await self.execute_task(task)
            result["provider_used"] = provider_type.value
            return result
        finally:
            # Restore original provider
            await self.switch_provider(original_provider)
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a task using the current provider"""
        # Determine the best provider for this task
        best_provider = await self._select_best_provider(task)
        
        if best_provider and best_provider != self.current_provider:
            # Switch to the best provider temporarily
            original_provider = self.current_provider
            await self.switch_provider(best_provider)
            
            try:
                result = await super().execute_task(task)
                result["provider_used"] = best_provider.value
                return result
            finally:
                # Restore original provider
                await self.switch_provider(original_provider)
        else:
            # Use current provider
            result = await super().execute_task(task)
            result["provider_used"] = self.current_provider.value
            return result
    
    async def _select_best_provider(self, task: Dict[str, Any]) -> Optional[ProviderType]:
        """Select the best provider for a given task"""
        # Simple heuristic: check task requirements and match to provider capabilities
        required_capabilities = task.get("required_capabilities", [])
        
        # Check each registered provider
        for provider_type, config in self.configs.items():
            # In a real implementation, we'd check if the provider supports the required capabilities
            # For now, we'll just return the first one that matches a general requirement
            if required_capabilities:
                # This is a simplified check - in reality, you'd have more sophisticated logic
                model_info = llm_provider_registry.get_model(config.model)
                if model_info and any(cap in model_info.capabilities for cap in required_capabilities):
                    return provider_type
        
        # If no specific requirement is met, return current provider
        return self.current_provider


class CostAwareAIAgent(AIAgent):
    """AI agent that tracks and optimizes API costs"""
    
    def __init__(self, config: AIAgentConfig):
        super().__init__(config)
        self.cost_tracker = CostTracker()
        self.budget_limit = config.max_tokens * 0.0001  # Default budget based on max tokens
        self.cost_history: List[Dict[str, Any]] = []
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a task with cost tracking"""
        # Check if we're within budget
        current_cost = self.cost_tracker.get_total_cost()
        if current_cost > self.budget_limit:
            return {
                "status": "failed",
                "error": "Budget limit exceeded",
                "current_cost": current_cost,
                "budget_limit": self.budget_limit
            }
        
        # Execute the task
        result = await super().execute_task(task)
        
        # Track the cost
        if "cost" in result:
            self.cost_tracker.add_cost(result["cost"], result.get("tokens_used", 0))
            self.cost_history.append({
                "task_id": task.get("id"),
                "cost": result["cost"],
                "tokens_used": result.get("tokens_used", 0),
                "timestamp": datetime.now().isoformat(),
                "model": result.get("model_used", self.config.model)
            })
        
        return result
    
    def set_budget_limit(self, limit: float):
        """Set a new budget limit"""
        self.budget_limit = limit
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get cost analysis and optimization suggestions"""
        total_cost = self.cost_tracker.get_total_cost()
        total_tokens = self.cost_tracker.get_total_tokens()
        
        analysis = {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "budget_limit": self.budget_limit,
            "remaining_budget": self.budget_limit - total_cost,
            "cost_per_token": total_cost / total_tokens if total_tokens > 0 else 0,
            "cost_efficiency": "high" if total_cost < self.budget_limit * 0.5 else 
                             "medium" if total_cost < self.budget_limit * 0.8 else "low",
            "cost_history": self.cost_history[-10:]  # Last 10 cost entries
        }
        
        return analysis


class CostTracker:
    """Simple cost tracker for API usage"""
    
    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens = 0
        self.cost_history = []
    
    def add_cost(self, cost: float, tokens: int = 0):
        """Add cost to the tracker"""
        self.total_cost += cost
        self.total_tokens += tokens
        self.cost_history.append({
            "cost": cost,
            "tokens": tokens,
            "timestamp": datetime.now().isoformat(),
            "cumulative_cost": self.total_cost
        })
    
    def get_total_cost(self) -> float:
        """Get the total cost"""
        return self.total_cost
    
    def get_total_tokens(self) -> int:
        """Get the total tokens used"""
        return self.total_tokens
    
    def reset(self):
        """Reset the cost tracker"""
        self.total_cost = 0.0
        self.total_tokens = 0
        self.cost_history = []