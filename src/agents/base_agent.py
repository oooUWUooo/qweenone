#!/usr/bin/env python3

import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum
import asyncio
import json
from dataclasses import dataclass, field


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


@dataclass
class AgentResponse:
    """Standard response from agent operations"""
    success: bool
    data: Any = None
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    
    def __init__(self, name: str, description: str = "", agent_id: Optional[str] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.capabilities = []
        self.status = AgentStatus.IDLE
        self.memory = {}  # Short-term memory for the agent
        self.long_term_memory = {}  # Long-term storage if needed
        self.agent_type = "base"
        self.api_manager = None  # Will be set by the agent builder or system
        
    @abstractmethod
    async def execute_task(self, task: 'Task') -> AgentResponse:
        pass
    
    def update_status(self, status: AgentStatus):
        self.status = status
    
    def add_capability(self, capability: AgentCapability):
        if capability not in self.capabilities:
            self.capabilities.append(capability)
    
    def has_capability(self, capability: AgentCapability) -> bool:
        return capability in self.capabilities
    
    def store_in_memory(self, key: str, value: Any):
        self.memory[key] = value
    
    def get_from_memory(self, key: str, default: Any = None) -> Any:
        return self.memory.get(key, default)
    
    def clear_memory(self):
        self.memory.clear()
    
    async def communicate_with_agent(self, message: Dict[str, Any]) -> bool:
        """
        Handle incoming messages from other agents
        This method should be implemented by subclasses to handle specific communication needs
        """
        # Default implementation - just log the message
        print(f"Agent {self.name} received message: {message}")
        return True
    
    def get_agent_info(self) -> Dict[str, Any]:
        return {
            "id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "agent_type": self.agent_type,
            "memory_size": len(self.memory)
        }
    
    def __str__(self) -> str:
        return f"Agent({self.name}, ID: {self.agent_id}, Type: {self.agent_type})"
    
    def __repr__(self) -> str:
        return self.__str__()


class Task:
    
    def __init__(self, 
                 task_id: str = None,
                 title: str = "",
                 description: str = "",
                 priority: int = 1,  # 1-5 scale, 5 is highest
                 dependencies: List[str] = None,
                 assigned_to: str = None,  # Agent ID
                 status: str = "pending",
                 params: Dict[str, Any] = None,
                 context: str = "",
                 constraints: List[str] = None,
                 expected_output: str = "",
                 config: Dict[str, Any] = None):
        
        self.task_id = task_id or str(uuid.uuid4())
        self.title = title
        self.description = description
        self.priority = priority
        self.dependencies = dependencies or []
        self.assigned_to = assigned_to
        self.status = status
        self.params = params or {}
        self.context = context
        self.constraints = constraints or []
        self.expected_output = expected_output
        self.config = config or {}
        self.created_at = asyncio.get_event_loop().time()
        self.completed_at = None
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "assigned_to": self.assigned_to,
            "status": self.status,
            "params": self.params,
            "context": self.context,
            "constraints": self.constraints,
            "expected_output": self.expected_output,
            "config": self.config,
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }
    
    def __str__(self) -> str:
        return f"Task({self.title}, ID: {self.task_id}, Status: {self.status})"
    
    def __repr__(self) -> str:
        return self.__str__()