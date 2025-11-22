# Default Agent Implementation
from src.agents.base_agent import BaseAgent, Task, AgentResponse
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime


class DefaultAgent(BaseAgent):
    """
    Default implementation of an AI agent that can handle various types of tasks.
    This agent uses a simple approach to process tasks and communicate with LLMs.
    """
    
    def __init__(self, name: str, description: str = "", agent_id: Optional[str] = None, **kwargs):
        super().__init__(name, description, agent_id)
        self.capabilities = kwargs.get("capabilities", [])
        self.personality = kwargs.get("personality", "helpful")
        self.communication_style = kwargs.get("communication_style", "professional")
        self.max_retries = kwargs.get("max_retries", 3)
        self.timeout = kwargs.get("timeout", 30)
        
    async def execute_task(self, task: Task) -> AgentResponse:
        """
        Execute a given task using the agent's capabilities.
        
        Args:
            task: The task to execute
            
        Returns:
            AgentResponse containing the result of the task execution
        """
        try:
            # Prepare the prompt for the LLM based on the task
            prompt = self._prepare_prompt(task)
            
            # Use the API manager to get a response from the LLM
            if self.api_manager:
                response = await self.api_manager.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    **task.config
                )
                
                if response.success:
                    # Process the response and return it
                    result = self._process_response(response.data, task)
                    return AgentResponse(
                        success=True,
                        data=result,
                        metadata={
                            "agent_id": self.agent_id,
                            "task_id": task.task_id,
                            "timestamp": datetime.now().isoformat(),
                            "model_used": getattr(response, 'model_used', 'unknown'),
                            "tokens_used": getattr(response, 'tokens_used', 0)
                        }
                    )
                else:
                    return AgentResponse(
                        success=False,
                        error=response.error or "Unknown error occurred during task execution",
                        metadata={
                            "agent_id": self.agent_id,
                            "task_id": task.task_id,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
            else:
                # Fallback if no API manager is set
                return AgentResponse(
                    success=False,
                    error="No API manager configured for this agent",
                    metadata={
                        "agent_id": self.agent_id,
                        "task_id": task.task_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Error executing task: {str(e)}",
                metadata={
                    "agent_id": self.agent_id,
                    "task_id": task.task_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def _prepare_prompt(self, task: Task) -> str:
        """
        Prepare the prompt for the LLM based on the task.
        
        Args:
            task: The task to prepare a prompt for
            
        Returns:
            Formatted prompt string
        """
        base_prompt = f"Task: {task.description}\n"
        
        if task.context:
            base_prompt += f"Context: {task.context}\n"
            
        if task.constraints:
            base_prompt += f"Constraints: {', '.join(task.constraints)}\n"
            
        if task.expected_output:
            base_prompt += f"Expected Output: {task.expected_output}\n"
            
        base_prompt += f"Please complete this task and provide a detailed response."
        
        return base_prompt
    
    def _process_response(self, response_data: Dict[str, Any], task: Task) -> Any:
        """
        Process the response from the LLM and format it appropriately.
        
        Args:
            response_data: Raw response data from the LLM
            task: The original task
            
        Returns:
            Processed response data
        """
        # Extract the actual content from the response
        if isinstance(response_data, dict):
            choices = response_data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", str(response_data))
            else:
                content = str(response_data)
        else:
            content = str(response_data)
            
        return content