#!/usr/bin/env python3
"""
Base Agent class for the Agentic Architecture System
All specific agents will inherit from this class
"""

import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum
import asyncio
import json

class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    PAUSED = "paused"
    ERROR = "error"

class AgentCapability(Enum):
    TASK_EXECUTION = "task_execution"
    CODE_WRITING = "code_writing"
    TESTING = "testing"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    DECOMPOSITION = "decomposition"

class BaseAgent(ABC):
    """
    Base class for all agents in the system
    """
    
    def __init__(self, name: str, description: str = "", capabilities: List[AgentCapability] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.status = AgentStatus.IDLE
        self.memory = {}  # Short-term memory for the agent
        self.long_term_memory = {}  # Long-term storage if needed
        self.agent_type = "base"
        
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """
        Execute a specific task assigned to the agent
        This must be implemented by subclasses
        """
        pass
    
    def update_status(self, status: AgentStatus):
        """Update the agent's status"""
        self.status = status
    
    def add_capability(self, capability: AgentCapability):
        """Add a capability to the agent"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if the agent has a specific capability"""
        return capability in self.capabilities
    
    def store_in_memory(self, key: str, value: Any):
        """Store data in the agent's short-term memory"""
        self.memory[key] = value
    
    def get_from_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve data from the agent's short-term memory"""
        return self.memory.get(key, default)
    
    def clear_memory(self):
        """Clear the agent's short-term memory"""
        self.memory.clear()
    
    def communicate_with_agent(self, target_agent_id: str, message: Dict[str, Any]) -> bool:
        """
        Send a message to another agent
        This will be handled by the A2A communication manager
        """
        # This method will be implemented in a concrete subclass
        # or through integration with the communication manager
        pass
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "agent_type": self.agent_type,
            "memory_size": len(self.memory)
        }
    
    def __str__(self) -> str:
        return f"Agent({self.name}, ID: {self.id}, Type: {self.agent_type})"
    
    def __repr__(self) -> str:
        return self.__str__()


class Task:
    """
    Represents a task that can be assigned to an agent
    """
    
    def __init__(self, 
                 task_id: str = None,
                 title: str = "",
                 description: str = "",
                 priority: int = 1,  # 1-5 scale, 5 is highest
                 dependencies: List[str] = None,
                 assigned_to: str = None,  # Agent ID
                 status: str = "pending",
                 params: Dict[str, Any] = None):
        
        self.id = task_id or str(uuid.uuid4())
        self.title = title
        self.description = description
        self.priority = priority
        self.dependencies = dependencies or []
        self.assigned_to = assigned_to
        self.status = status
        self.params = params or {}
        self.created_at = asyncio.get_event_loop().time()
        self.completed_at = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "assigned_to": self.assigned_to,
            "status": self.status,
            "params": self.params,
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }
    
    def __str__(self) -> str:
        return f"Task({self.title}, ID: {self.id}, Status: {self.status})"
    
    def __repr__(self) -> str:
        return self.__str__()