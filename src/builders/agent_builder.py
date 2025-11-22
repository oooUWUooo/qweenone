#!/usr/bin/env python3
"""
Agent Builder for the Agentic Architecture System
Handles the creation of new agents with specific capabilities
"""

from typing import Dict, Any, Type
from src.agents.base_agent import BaseAgent, AgentCapability
from src.utils.logger import setup_logger

class AgentBuilder:
    """
    Component responsible for building new agents with specific configurations
    """
    
    def __init__(self):
        self.logger = setup_logger("AgentBuilder")
        self.agent_registry = {}  # Registry of available agent types
        
        # Register default agent types
        self._register_default_agents()
    
    def _register_default_agents(self):
        """
        Register default agent types that can be created
        """
        # In a real system, this would register actual agent classes
        # For now, we'll just define the types that will be implemented
        self.agent_types = {
            "code_writer": {
                "name": "Code Writer Agent",
                "description": "Specialized in writing and reviewing code",
                "capabilities": [AgentCapability.CODE_WRITING, AgentCapability.ANALYSIS]
            },
            "tester": {
                "name": "Testing Agent", 
                "description": "Specialized in testing code and validating solutions",
                "capabilities": [AgentCapability.TESTING, AgentCapability.ANALYSIS]
            },
            "task_manager": {
                "name": "Task Manager Agent",
                "description": "Specialized in managing and decomposing tasks",
                "capabilities": [AgentCapability.DECOMPOSITION, AgentCapability.TASK_EXECUTION]
            },
            "communication": {
                "name": "Communication Agent",
                "description": "Specialized in agent-to-agent communication",
                "capabilities": [AgentCapability.COMMUNICATION]
            }
        }
    
    def create_agent(self, config: Dict[str, Any]) -> BaseAgent:
        """
        Create a new agent based on the provided configuration
        """
        agent_type = config.get("type", "default")
        name = config.get("name", f"Agent_{agent_type}")
        description = config.get("description", f"Default {agent_type} agent")
        capabilities = config.get("capabilities", [])
        
        # Create the appropriate agent based on type
        if agent_type == "code_writer":
            agent = CodeAgent(name, description)
        elif agent_type == "tester":
            agent = TestingAgent(name, description)
        elif agent_type == "task_manager":
            agent = TaskAgent(name, description)
        elif agent_type == "communication":
            agent = CommunicationAgent(name, description)
        else:
            # Default agent if type not recognized
            agent = self._create_default_agent(name, description, capabilities)
        
        # Add any additional capabilities from config
        for cap_name in capabilities:
            try:
                capability = AgentCapability(cap_name)
                agent.add_capability(capability)
            except ValueError:
                self.logger.warning(f"Unknown capability: {cap_name}")
        
        self.logger.info(f"Created agent: {agent.name} (ID: {agent.id})")
        return agent
    
    def _create_default_agent(self, name: str, description: str, capabilities: list) -> BaseAgent:
        """
        Create a default agent when specific type is not available
        """
        from src.agents.default_agent import DefaultAgent
        agent = DefaultAgent(name, description)
        
        # Add capabilities
        for cap_name in capabilities:
            try:
                capability = AgentCapability(cap_name)
                agent.add_capability(capability)
            except ValueError:
                self.logger.warning(f"Unknown capability: {cap_name}")
        
        return agent
    
    def register_agent_type(self, agent_type: str, agent_class: Type[BaseAgent], 
                           default_capabilities: list = None, description: str = ""):
        """
        Register a new agent type that can be created by the builder
        """
        self.agent_registry[agent_type] = {
            "class": agent_class,
            "default_capabilities": default_capabilities or [],
            "description": description
        }
        self.logger.info(f"Registered new agent type: {agent_type}")
    
    def list_available_agents(self) -> Dict[str, Any]:
        """
        List all available agent types that can be created
        """
        return {
            "default_types": self.agent_types,
            "custom_types": self.agent_registry
        }


# Default agent implementations that will be referenced by the builder

class DefaultAgent(BaseAgent):
    """
    Default agent implementation for general purposes
    """
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "default"
        self.add_capability(AgentCapability.TASK_EXECUTION)
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """
        Execute a task - in the default implementation, just log what would be done
        """
        self.update_status("working")
        self.logger = setup_logger(f"DefaultAgent_{self.id}")
        
        # Store task in memory
        self.store_in_memory("current_task", task)
        
        # Log the task execution
        self.logger.info(f"Executing task: {task.get('title', 'Unknown task')}")
        
        # Simulate task execution
        result = {
            "status": "completed",
            "agent_id": self.id,
            "task_id": task.get("id"),
            "result": f"Default execution of task: {task.get('title', 'Unknown task')}"
        }
        
        self.update_status("idle")
        return result


class CodeAgent(BaseAgent):
    """
    Specialized agent for code writing and development tasks
    """
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "code_writer"
        self.add_capability(AgentCapability.CODE_WRITING)
        self.add_capability(AgentCapability.ANALYSIS)
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """
        Execute a code-related task
        """
        self.update_status("working")
        self.logger = setup_logger(f"CodeAgent_{self.id}")
        
        self.logger.info(f"Writing code for task: {task.get('title', 'Unknown task')}")
        
        # This is where actual code generation would happen
        # For now, we'll simulate the process
        result = {
            "status": "completed",
            "agent_id": self.id,
            "task_id": task.get("id"),
            "result": f"Code generated for: {task.get('title', 'Unknown task')}",
            "files_created": [],
            "code_snippets": []
        }
        
        self.update_status("idle")
        return result


class TestingAgent(BaseAgent):
    """
    Specialized agent for testing and validation tasks
    """
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "tester"
        self.add_capability(AgentCapability.TESTING)
        self.add_capability(AgentCapability.ANALYSIS)
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """
        Execute a testing-related task
        """
        self.update_status("working")
        self.logger = setup_logger(f"TestingAgent_{self.id}")
        
        self.logger.info(f"Testing task: {task.get('title', 'Unknown task')}")
        
        # This is where actual testing would happen
        # For now, we'll simulate the process
        result = {
            "status": "completed",
            "agent_id": self.id,
            "task_id": task.get("id"),
            "result": f"Tests executed for: {task.get('title', 'Unknown task')}",
            "test_results": {
                "passed": 0,
                "failed": 0,
                "errors": []
            }
        }
        
        self.update_status("idle")
        return result


class TaskAgent(BaseAgent):
    """
    Specialized agent for task management and decomposition
    """
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "task_manager"
        self.add_capability(AgentCapability.DECOMPOSITION)
        self.add_capability(AgentCapability.TASK_EXECUTION)
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """
        Execute a task management task
        """
        self.update_status("working")
        self.logger = setup_logger(f"TaskAgent_{self.id}")
        
        self.logger.info(f"Managing task: {task.get('title', 'Unknown task')}")
        
        # This might involve decomposing complex tasks
        result = {
            "status": "completed",
            "agent_id": self.id,
            "task_id": task.get("id"),
            "result": f"Task managed: {task.get('title', 'Unknown task')}",
            "subtasks_created": 0
        }
        
        self.update_status("idle")
        return result


class CommunicationAgent(BaseAgent):
    """
    Specialized agent for handling agent-to-agent communication
    """
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "communication"
        self.add_capability(AgentCapability.COMMUNICATION)
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """
        Execute a communication task
        """
        self.update_status("working")
        self.logger = setup_logger(f"CommunicationAgent_{self.id}")
        
        self.logger.info(f"Handling communication task: {task.get('title', 'Unknown task')}")
        
        # This would handle A2A communication
        result = {
            "status": "completed",
            "agent_id": self.id,
            "task_id": task.get("id"),
            "result": f"Communication task completed: {task.get('title', 'Unknown task')}",
            "messages_sent": 0,
            "messages_received": 0
        }
        
        self.update_status("idle")
        return result