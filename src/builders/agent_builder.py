#!/usr/bin/env python3

from typing import Dict, Any, Type, List, Optional, Callable
from enum import Enum
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from src.agents.base_agent import BaseAgent, AgentCapability
from src.utils.logger import setup_logger
from src.agents.default_agent import DefaultAgent

try:  # Optional SGR integration
    from src.enhanced_agents.sgr_adapter import register_sgr_agent
except Exception:  # pragma: no cover
    register_sgr_agent = None  # type: ignore

try:  # Optional Scrapybara integration
    from src.web_research.scrapybara_adapter import register_scrapybara_agent
except Exception:  # pragma: no cover
    register_scrapybara_agent = None  # type: ignore

class AgentConfig:
    """Configuration class for agent creation"""
    
    def __init__(self, 
                 name: str = None,
                 description: str = "",
                 agent_type: str = "default",
                 capabilities: List[AgentCapability] = None,
                 parameters: Dict[str, Any] = None,
                 dependencies: List[str] = None,
                 max_memory_size: int = 1000,
                 timeout: int = 300,
                 priority: int = 1,
                 working_directory: str = "/workspace",
                 model_config: Dict[str, Any] = None,
                 tools: List[str] = None):
        self.name = name or f"Agent_{uuid.uuid4().hex[:8]}"
        self.description = description
        self.agent_type = agent_type
        self.capabilities = capabilities or []
        self.parameters = parameters or {}
        self.dependencies = dependencies or []
        self.max_memory_size = max_memory_size
        self.timeout = timeout
        self.priority = priority
        self.working_directory = working_directory
        self.model_config = model_config or {}
        self.tools = tools or []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "capabilities": [cap.value for cap in self.capabilities],
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "max_memory_size": self.max_memory_size,
            "timeout": self.timeout,
            "priority": self.priority,
            "working_directory": self.working_directory,
            "model_config": self.model_config,
            "tools": self.tools
        }

class AgentProfile:
    """Profile definition for different agent types"""
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 default_capabilities: List[AgentCapability], 
                 default_parameters: Dict[str, Any] = None,
                 required_parameters: List[str] = None):
        self.name = name
        self.description = description
        self.default_capabilities = default_capabilities
        self.default_parameters = default_parameters or {}
        self.required_parameters = required_parameters or []

class AgentBuilder:
    
    def __init__(self):
        self.logger = setup_logger("AgentBuilder")
        self.agent_registry = {}  # Registry of available agent types
        self.agent_profiles = {}  # Predefined agent profiles
        self.created_agents = {}  # Track created agents
        self.agent_id_counter = 0
        
        # Register default agent types
        self._register_default_agents()
        self._register_agent_profiles()
        self._register_optional_agents()
    
    def _register_optional_agents(self):
        if register_sgr_agent:
            try:
                register_sgr_agent(self)
            except Exception as exc:  # pragma: no cover - optional integration errors
                self.logger.warning(f"Failed to register SGR agent: {exc}")
        if register_scrapybara_agent:
            try:
                register_scrapybara_agent(self)
            except Exception as exc:  # pragma: no cover
                self.logger.warning(f"Failed to register Scrapybara agent: {exc}")
    
    def _register_default_agents(self):
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
            },
            "researcher": {
                "name": "Research Agent",
                "description": "Specialized in gathering information and research",
                "capabilities": [AgentCapability.ANALYSIS, AgentCapability.DATA_PROCESSING]
            },
            "planner": {
                "name": "Planning Agent",
                "description": "Specialized in creating execution plans and strategies",
                "capabilities": [AgentCapability.DECOMPOSITION, AgentCapability.PLANNING]
            },
            "reviewer": {
                "name": "Review Agent",
                "description": "Specialized in reviewing and validating work",
                "capabilities": [AgentCapability.ANALYSIS, AgentCapability.VALIDATION]
            },
            "debugger": {
                "name": "Debug Agent",
                "description": "Specialized in identifying and fixing issues",
                "capabilities": [AgentCapability.ANALYSIS, AgentCapability.TROUBLESHOOTING]
            }
        }
    
    def _register_agent_profiles(self):
        """Register predefined agent profiles"""
        self.agent_profiles = {
            "basic": AgentProfile(
                name="Basic Agent",
                description="Simple agent with basic capabilities",
                default_capabilities=[AgentCapability.TASK_EXECUTION],
                default_parameters={"max_retries": 3}
            ),
            "developer": AgentProfile(
                name="Developer Agent",
                description="Agent specialized for development tasks",
                default_capabilities=[
                    AgentCapability.CODE_WRITING,
                    AgentCapability.CODE_REVIEW,
                    AgentCapability.TESTING
                ],
                default_parameters={
                    "language": "python",
                    "style_guide": "PEP8"
                }
            ),
            "tester": AgentProfile(
                name="Tester Agent",
                description="Agent specialized for testing tasks",
                default_capabilities=[
                    AgentCapability.TESTING,
                    AgentCapability.ANALYSIS
                ],
                default_parameters={
                    "test_framework": "pytest",
                    "coverage_threshold": 80
                }
            ),
            "manager": AgentProfile(
                name="Manager Agent",
                description="Agent specialized for task management",
                default_capabilities=[
                    AgentCapability.DECOMPOSITION,
                    AgentCapability.TASK_EXECUTION,
                    AgentCapability.COMMUNICATION
                ],
                default_parameters={
                    "max_subtasks": 10,
                    "timeout": 300
                }
            )
        }
    
    def create_agent(self, config: Dict[str, Any]) -> BaseAgent:
        """
        Create an agent based on the provided configuration.
        
        Args:
            config: Dictionary containing agent configuration parameters
            
        Returns:
            BaseAgent: An instance of the requested agent type
        """
        agent_type = config.get("type", "default")
        name = config.get("name", f"Agent_{agent_type}")
        description = config.get("description", f"Default {agent_type} agent")
        capabilities = config.get("capabilities", [])
        
        # Allow custom agent registry overrides first
        registered_entry = self.agent_registry.get(agent_type)
        if registered_entry:
            agent_class_or_factory = registered_entry["class"]
            default_caps = registered_entry.get("default_capabilities", [])
            try:
                agent = agent_class_or_factory(
                    name=name,
                    description=description,
                    config=config
                )
            except TypeError:
                agent = agent_class_or_factory(name, description)
            for capability in default_caps:
                if isinstance(capability, AgentCapability):
                    agent.add_capability(capability)
                elif isinstance(capability, str):
                    try:
                        agent.add_capability(AgentCapability(capability))
                    except ValueError:
                        self.logger.warning(f"Unknown default capability: {capability}")
        else:
            # Create the appropriate agent based on type
            if agent_type == "code_writer":
                agent = CodeAgent(name, description)
            elif agent_type == "tester":
                agent = TestingAgent(name, description)
            elif agent_type == "task_manager":
                agent = TaskAgent(name, description)
            elif agent_type == "communication":
                agent = CommunicationAgent(name, description)
            elif agent_type == "researcher":
                agent = ResearchAgent(name, description)
            elif agent_type == "planner":
                agent = PlanningAgent(name, description)
            elif agent_type == "reviewer":
                agent = ReviewAgent(name, description)
            elif agent_type == "debugger":
                agent = DebugAgent(name, description)
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
        
        # Store agent in registry
        self.agent_id_counter += 1
        agent_id = f"{agent_type}_{self.agent_id_counter}"
        agent.id = agent_id
        self.created_agents[agent_id] = agent
        
        self.logger.info(f"Created agent: {agent.name} (ID: {agent.id})")
        return agent
    
    def create_agent_from_profile(self, profile_name: str, custom_config: Dict[str, Any] = None) -> BaseAgent:
        """
        Create an agent using a predefined profile.
        
        Args:
            profile_name: Name of the agent profile to use
            custom_config: Custom configuration to override profile defaults
            
        Returns:
            BaseAgent: An instance of the agent based on the profile
        """
        if profile_name not in self.agent_profiles:
            raise ValueError(f"Unknown agent profile: {profile_name}")
        
        profile = self.agent_profiles[profile_name]
        custom_config = custom_config or {}
        
        # Merge profile defaults with custom config
        merged_config = {
            "name": custom_config.get("name", f"{profile.name}_{uuid.uuid4().hex[:8]}"),
            "description": custom_config.get("description", profile.description),
            "type": custom_config.get("type", "default"),
            "capabilities": profile.default_capabilities + custom_config.get("capabilities", []),
            "parameters": {**profile.default_parameters, **custom_config.get("parameters", {})}
        }
        
        return self.create_agent(merged_config)
    
    def create_agent_from_config(self, agent_config: AgentConfig) -> BaseAgent:
        """
        Create an agent from an AgentConfig object.
        
        Args:
            agent_config: AgentConfig object containing configuration
            
        Returns:
            BaseAgent: An instance of the requested agent type
        """
        config_dict = agent_config.to_dict()
        config_dict["type"] = agent_config.agent_type
        return self.create_agent(config_dict)
    
    def batch_create_agents(self, configs: List[Dict[str, Any]]) -> List[BaseAgent]:
        """
        Create multiple agents in batch.
        
        Args:
            configs: List of agent configuration dictionaries
            
        Returns:
            List[BaseAgent]: List of created agents
        """
        agents = []
        for config in configs:
            try:
                agent = self.create_agent(config)
                agents.append(agent)
            except Exception as e:
                self.logger.error(f"Failed to create agent with config {config}: {str(e)}")
                continue
        return agents
    
    def create_dynamic_agent(self, name: str, description: str, capabilities: List[AgentCapability]) -> BaseAgent:
        """
        Create an agent with custom capabilities without predefined type.
        
        Args:
            name: Name of the agent
            description: Description of the agent
            capabilities: List of capabilities to assign to the agent
            
        Returns:
            BaseAgent: A dynamically created agent
        """
        agent = DefaultAgent(name, description)
        for capability in capabilities:
            agent.add_capability(capability)
        return agent
    
    def _create_default_agent(self, name: str, description: str, capabilities: list) -> BaseAgent:
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
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Retrieve an agent by its ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            BaseAgent: The requested agent or None if not found
        """
        return self.created_agents.get(agent_id)
    
    def list_agents(self) -> Dict[str, BaseAgent]:
        """
        List all created agents.
        
        Returns:
            Dict: Dictionary mapping agent IDs to agent instances
        """
        return self.created_agents.copy()
    
    def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent by its ID.
        
        Args:
            agent_id: ID of the agent to delete
            
        Returns:
            bool: True if agent was deleted, False if not found
        """
        if agent_id in self.created_agents:
            del self.created_agents[agent_id]
            return True
        return False
    
    def update_agent_config(self, agent_id: str, new_config: Dict[str, Any]) -> bool:
        """
        Update an existing agent's configuration.
        
        Args:
            agent_id: ID of the agent to update
            new_config: New configuration parameters
            
        Returns:
            bool: True if agent was updated, False if not found
        """
        agent = self.get_agent(agent_id)
        if agent:
            # Update agent properties based on new config
            if "name" in new_config:
                agent.name = new_config["name"]
            if "description" in new_config:
                agent.description = new_config["description"]
            if "capabilities" in new_config:
                # Clear existing capabilities and add new ones
                agent.capabilities.clear()
                for cap_name in new_config["capabilities"]:
                    try:
                        capability = AgentCapability(cap_name)
                        agent.add_capability(capability)
                    except ValueError:
                        self.logger.warning(f"Unknown capability: {cap_name}")
            return True
        return False


# Agent implementations that will be referenced by the builder

class ResearchAgent(BaseAgent):
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "researcher"
        self.add_capability(AgentCapability.ANALYSIS)
        self.add_capability(AgentCapability.DATA_PROCESSING)
        self.knowledge_base = {}
        self.search_history = []
        
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        self.update_status("working")
        self.logger = setup_logger(f"ResearchAgent_{self.id}")
        
        self.logger.info(f"Researching: {task.get('title', 'Unknown task')}")
        
        # Simulate research process
        research_topic = task.get('research_topic', task.get('title', 'general research'))
        
        # Store research in memory
        self.store_in_memory("current_research", research_topic)
        
        # Simulate search process
        search_results = self._simulate_search(research_topic)
        self.search_history.append({
            "topic": research_topic,
            "results": search_results,
            "timestamp": datetime.now().isoformat()
        })
        
        result = {
            "status": "completed",
            "agent_id": self.id,
            "task_id": task.get("id"),
            "result": f"Research completed on: {research_topic}",
            "findings": search_results,
            "sources_used": len(search_results),
            "confidence_score": 0.85
        }
        
        self.update_status("idle")
        return result
    
    def _simulate_search(self, topic: str) -> List[Dict[str, Any]]:
        """Simulate a search process"""
        return [
            {
                "title": f"Research result for {topic}",
                "content": f"Detailed information about {topic}",
                "source": "simulated_database",
                "relevance_score": 0.9
            }
        ]


class PlanningAgent(BaseAgent):
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "planner"
        self.add_capability(AgentCapability.DECOMPOSITION)
        self.add_capability(AgentCapability.PLANNING)
        self.planning_strategies = {}
        self.execution_history = []
        
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        self.update_status("working")
        self.logger = setup_logger(f"PlanningAgent_{self.id}")
        
        self.logger.info(f"Creating plan for: {task.get('title', 'Unknown task')}")
        
        # Store task in memory
        self.store_in_memory("current_task", task)
        
        # Generate execution plan
        plan = self._generate_plan(task)
        self.execution_history.append({
            "task": task,
            "plan": plan,
            "timestamp": datetime.now().isoformat()
        })
        
        result = {
            "status": "completed",
            "agent_id": self.id,
            "task_id": task.get("id"),
            "result": f"Plan created for: {task.get('title', 'Unknown task')}",
            "plan": plan,
            "estimated_duration": plan.get("estimated_duration", "unknown"),
            "required_resources": plan.get("required_resources", [])
        }
        
        self.update_status("idle")
        return result
    
    def _generate_plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an execution plan for the given task"""
        return {
            "task_id": task.get("id"),
            "title": task.get("title", "Unnamed Task"),
            "steps": [
                {"id": 1, "description": "Analyze requirements", "estimated_time": 30},
                {"id": 2, "description": "Design solution", "estimated_time": 60},
                {"id": 3, "description": "Implement solution", "estimated_time": 120},
                {"id": 4, "description": "Test implementation", "estimated_time": 60},
                {"id": 5, "description": "Deploy solution", "estimated_time": 30}
            ],
            "estimated_duration": 300,
            "required_resources": ["developer", "tester", "reviewer"],
            "dependencies": [],
            "priority": task.get("priority", 1)
        }


class ReviewAgent(BaseAgent):
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "reviewer"
        self.add_capability(AgentCapability.ANALYSIS)
        self.add_capability(AgentCapability.VALIDATION)
        self.review_criteria = {}
        self.review_history = []
        
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        self.update_status("working")
        self.logger = setup_logger(f"ReviewAgent_{self.id}")
        
        self.logger.info(f"Reviewing: {task.get('title', 'Unknown task')}")
        
        # Store task in memory
        self.store_in_memory("current_review", task)
        
        # Perform review
        review_result = self._perform_review(task)
        self.review_history.append({
            "task": task,
            "review": review_result,
            "timestamp": datetime.now().isoformat()
        })
        
        result = {
            "status": "completed",
            "agent_id": self.id,
            "task_id": task.get("id"),
            "result": f"Review completed for: {task.get('title', 'Unknown task')}",
            "review": review_result,
            "issues_found": len(review_result.get("issues", [])),
            "recommendations": review_result.get("recommendations", [])
        }
        
        self.update_status("idle")
        return result
    
    def _perform_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a review of the given task"""
        return {
            "task_id": task.get("id"),
            "quality_score": 8.5,
            "issues": [
                {
                    "type": "potential_improvement",
                    "description": "Consider alternative approach",
                    "severity": "medium"
                }
            ],
            "recommendations": [
                "Implement additional error handling",
                "Add more comprehensive tests"
            ],
            "compliance_check": {
                "standards_met": True,
                "issues": []
            }
        }


class DebugAgent(BaseAgent):
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "debugger"
        self.add_capability(AgentCapability.ANALYSIS)
        self.add_capability(AgentCapability.TROUBLESHOOTING)
        self.debugging_tools = []
        self.issue_history = []
        
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        self.update_status("working")
        self.logger = setup_logger(f"DebugAgent_{self.id}")
        
        self.logger.info(f"Debugging: {task.get('title', 'Unknown task')}")
        
        # Store task in memory
        self.store_in_memory("current_debug", task)
        
        # Perform debugging
        debug_result = self._perform_debugging(task)
        self.issue_history.append({
            "task": task,
            "debug_result": debug_result,
            "timestamp": datetime.now().isoformat()
        })
        
        result = {
            "status": "completed",
            "agent_id": self.id,
            "task_id": task.get("id"),
            "result": f"Debugging completed for: {task.get('title', 'Unknown task')}",
            "debug_result": debug_result,
            "issues_found": len(debug_result.get("issues", [])),
            "solutions_applied": len(debug_result.get("solutions", []))
        }
        
        self.update_status("idle")
        return result
    
    def _perform_debugging(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform debugging of the given task"""
        return {
            "task_id": task.get("id"),
            "issues": [
                {
                    "id": 1,
                    "type": "logic_error",
                    "description": "Potential null pointer exception",
                    "location": "line 45",
                    "severity": "high"
                }
            ],
            "solutions": [
                {
                    "issue_id": 1,
                    "solution": "Add null check before accessing object",
                    "implementation": "if obj is not None: ..."
                }
            ],
            "debugging_steps": [
                "Analyze error logs",
                "Reproduce the issue",
                "Identify root cause",
                "Propose solution",
                "Verify fix"
            ]
        }
    
    def register_agent_type(self, agent_type: str, agent_class: Type[BaseAgent], 
                           default_capabilities: list = None, description: str = ""):
        """
        Register a new agent type in the builder.
        
        Args:
            agent_type: Name of the agent type
            agent_class: Class of the agent to register
            default_capabilities: Default capabilities for this agent type
            description: Description of the agent type
            
        Returns:
            None
        """
        self.agent_registry[agent_type] = {
            "class": agent_class,
            "default_capabilities": default_capabilities or [],
            "description": description
        }
        self.logger.info(f"Registered new agent type: {agent_type}")
    
    def create_agent_with_dependencies(self, config: Dict[str, Any], 
                                     dependency_graph: Dict[str, List[str]] = None) -> BaseAgent:
        """
        Create an agent with specified dependencies.
        
        Args:
            config: Configuration for the main agent
            dependency_graph: Dictionary mapping agent IDs to their dependencies
            
        Returns:
            BaseAgent: The created agent with dependencies
        """
        agent = self.create_agent(config)
        
        if dependency_graph:
            # Set up dependencies for the agent
            agent.dependencies = dependency_graph.get(agent.id, [])
        
        return agent
    
    def create_cooperative_agent_network(self, 
                                       agent_configs: List[Dict[str, Any]], 
                                       communication_protocol: str = "a2a") -> List[BaseAgent]:
        """
        Create a network of cooperative agents with communication capabilities.
        
        Args:
            agent_configs: List of configurations for agents to create
            communication_protocol: Protocol for agent-to-agent communication
            
        Returns:
            List[BaseAgent]: List of created cooperative agents
        """
        agents = []
        
        # Create all agents first
        for config in agent_configs:
            agent = self.create_agent(config)
            agents.append(agent)
        
        # Set up communication between agents
        for agent in agents:
            agent.add_capability(AgentCapability.COMMUNICATION)
            # Add other agents as communication partners
            agent.communication_partners = [a.id for a in agents if a.id != agent.id]
        
        return agents
    
    def export_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """
        Export the configuration of an agent.
        
        Args:
            agent_id: ID of the agent to export
            
        Returns:
            Dict: Configuration of the agent
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent with ID {agent_id} not found")
        
        return {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "type": getattr(agent, 'agent_type', 'unknown'),
            "capabilities": [cap.value for cap in agent.capabilities],
            "status": agent.status.value,
            "created_at": getattr(agent, 'created_at', datetime.now().isoformat()),
            "dependencies": getattr(agent, 'dependencies', []),
            "communication_partners": getattr(agent, 'communication_partners', [])
        }
    
    def import_agent_config(self, config: Dict[str, Any]) -> BaseAgent:
        """
        Create an agent from an exported configuration.
        
        Args:
            config: Configuration dictionary to import
            
        Returns:
            BaseAgent: The created agent
        """
        # Create a new agent based on the configuration
        agent_config = AgentConfig(
            name=config.get("name"),
            description=config.get("description"),
            agent_type=config.get("type", "default"),
            capabilities=[AgentCapability(cap) for cap in config.get("capabilities", [])],
            parameters=config.get("parameters", {})
        )
        
        agent = self.create_agent_from_config(agent_config)
        
        # Restore additional properties
        agent.status = agent.status.__class__(config.get("status", "idle"))
        if "dependencies" in config:
            agent.dependencies = config["dependencies"]
        if "communication_partners" in config:
            agent.communication_partners = config["communication_partners"]
        
        return agent
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about created agents.
        
        Returns:
            Dict: Statistics about agents
        """
        stats = {
            "total_agents": len(self.created_agents),
            "agent_types": {},
            "capabilities_distribution": {},
            "average_capabilities_per_agent": 0
        }
        
        for agent_id, agent in self.created_agents.items():
            # Count agent types
            agent_type = getattr(agent, 'agent_type', 'unknown')
            stats["agent_types"][agent_type] = stats["agent_types"].get(agent_type, 0) + 1
            
            # Count capabilities
            for capability in agent.capabilities:
                cap_name = capability.value
                stats["capabilities_distribution"][cap_name] = stats["capabilities_distribution"].get(cap_name, 0) + 1
        
        if stats["total_agents"] > 0:
            total_capabilities = sum(len(agent.capabilities) for agent in self.created_agents.values())
            stats["average_capabilities_per_agent"] = total_capabilities / stats["total_agents"]
        
        return stats
    
    def cleanup_agents(self, max_age_hours: int = 24) -> int:
        """
        Clean up agents that have been idle for too long.
        
        Args:
            max_age_hours: Maximum age in hours before an agent is considered for cleanup
            
        Returns:
            int: Number of agents cleaned up
        """
        current_time = datetime.now()
        agents_to_remove = []
        
        for agent_id, agent in self.created_agents.items():
            # In a real implementation, we would check the agent's last activity time
            # For now, we'll just use a simple heuristic
            if agent.status.value == "idle":
                agents_to_remove.append(agent_id)
        
        for agent_id in agents_to_remove:
            del self.created_agents[agent_id]
        
        return len(agents_to_remove)
    
    def list_available_agents(self) -> Dict[str, Any]:
        """
        List all available agent types and profiles.
        
        Returns:
            Dict: Information about available agent types and profiles
        """
        return {
            "default_types": self.agent_types,
            "custom_types": self.agent_registry,
            "profiles": {name: {
                "name": profile.name,
                "description": profile.description,
                "default_capabilities": [cap.value for cap in profile.default_capabilities],
                "default_parameters": profile.default_parameters
            } for name, profile in self.agent_profiles.items()}
        }


# Default agent implementations that will be referenced by the builder

class DefaultAgent(BaseAgent):
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "default"
        self.add_capability(AgentCapability.TASK_EXECUTION)
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
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
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "code_writer"
        self.add_capability(AgentCapability.CODE_WRITING)
        self.add_capability(AgentCapability.ANALYSIS)
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
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
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "tester"
        self.add_capability(AgentCapability.TESTING)
        self.add_capability(AgentCapability.ANALYSIS)
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
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
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "task_manager"
        self.add_capability(AgentCapability.DECOMPOSITION)
        self.add_capability(AgentCapability.TASK_EXECUTION)
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
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
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.agent_type = "communication"
        self.add_capability(AgentCapability.COMMUNICATION)
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
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