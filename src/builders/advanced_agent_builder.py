#!/usr/bin/env python3
"""
Advanced Agent Builder System
This module provides a comprehensive framework for creating, configuring, and managing
complex agent networks with various capabilities, communication protocols, and task management.
"""

from typing import Dict, Any, List, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
import inspect
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
import copy

from src.agents.base_agent import BaseAgent, AgentCapability
from src.utils.logger import setup_logger


class AgentNetworkProtocol(Enum):
    """Communication protocols for agent networks"""
    A2A_SYNC = "a2a_sync"  # Agent-to-Agent synchronous
    A2A_ASYNC = "a2a_async"  # Agent-to-Agent asynchronous
    CENTRALIZED = "centralized"  # Centralized communication
    DECENTRALIZED = "decentralized"  # Decentralized communication
    HYBRID = "hybrid"  # Hybrid communication model


class AgentState(Enum):
    """States of an agent in the system"""
    CREATED = "created"
    INITIALIZED = "initialized"
    IDLE = "idle"
    WORKING = "working"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentRole(Enum):
    """Roles of agents in the system"""
    WORKER = "worker"
    MANAGER = "manager"
    COORDINATOR = "coordinator"
    ANALYST = "analyst"
    VALIDATOR = "validator"
    MONITOR = "monitor"
    COMMUNICATOR = "communicator"
    PLANNER = "planner"
    EXECUTOR = "executor"


@dataclass
class AgentResource:
    """Resource allocation for agents"""
    cpu_cores: int = 1
    memory_mb: int = 512
    disk_space_mb: int = 100
    network_bandwidth: str = "100Mbps"
    priority: int = 1
    max_concurrent_tasks: int = 1


@dataclass
class AgentMetrics:
    """Performance metrics for agents"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_duration: float = 0.0
    total_execution_time: float = 0.0
    memory_usage_peak: float = 0.0
    cpu_usage_average: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)


class AgentConstraint:
    """Constraints and requirements for agents"""
    
    def __init__(self, 
                 max_execution_time: int = 3600,  # seconds
                 max_memory_usage: int = 1024,   # MB
                 allowed_capabilities: List[AgentCapability] = None,
                 forbidden_capabilities: List[AgentCapability] = None,
                 required_tools: List[str] = None,
                 forbidden_tools: List[str] = None):
        self.max_execution_time = max_execution_time
        self.max_memory_usage = max_memory_usage
        self.allowed_capabilities = allowed_capabilities or list(AgentCapability)
        self.forbidden_capabilities = forbidden_capabilities or []
        self.required_tools = required_tools or []
        self.forbidden_tools = forbidden_tools or []


class AgentTemplate:
    """Template for creating agents with predefined configurations"""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 agent_type: str,
                 capabilities: List[AgentCapability],
                 default_config: Dict[str, Any] = None,
                 dependencies: List[str] = None,
                 resources: AgentResource = None,
                 constraints: AgentConstraint = None):
        self.name = name
        self.description = description
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.default_config = default_config or {}
        self.dependencies = dependencies or []
        self.resources = resources or AgentResource()
        self.constraints = constraints or AgentConstraint()
    
    def create_agent_config(self, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a configuration based on this template with optional overrides"""
        config = copy.deepcopy(self.default_config)
        if overrides:
            config.update(overrides)
        return config


class TaskQueue:
    """Queue for managing tasks across agents"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.tasks = []
        self.lock = threading.Lock()
    
    def add_task(self, task: Dict[str, Any]):
        """Add a task to the queue"""
        with self.lock:
            self.tasks.append(task)
    
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get the next task from the queue"""
        with self.lock:
            if self.tasks:
                return self.tasks.pop(0)
        return None
    
    def size(self) -> int:
        """Get the size of the queue"""
        with self.lock:
            return len(self.tasks)


class AgentNetwork:
    """Network of interconnected agents"""
    
    def __init__(self, 
                 name: str,
                 protocol: AgentNetworkProtocol = AgentNetworkProtocol.A2A_ASYNC):
        self.name = name
        self.protocol = protocol
        self.agents = {}
        self.connections = {}
        self.logger = setup_logger(f"AgentNetwork_{name}")
        self.task_queues = {}
        self.network_topology = {}
        self.communication_stats = {}
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the network"""
        self.agents[agent.id] = agent
        self.connections[agent.id] = []
        self.communication_stats[agent.id] = {
            "messages_sent": 0,
            "messages_received": 0,
            "last_communication": None
        }
        self.logger.info(f"Added agent {agent.id} to network {self.name}")
    
    def connect_agents(self, agent_id1: str, agent_id2: str):
        """Connect two agents in the network"""
        if agent_id1 in self.agents and agent_id2 in self.agents:
            if agent_id2 not in self.connections[agent_id1]:
                self.connections[agent_id1].append(agent_id2)
            if agent_id1 not in self.connections[agent_id2]:
                self.connections[agent_id2].append(agent_id1)
            self.logger.info(f"Connected agents {agent_id1} and {agent_id2}")
    
    def send_message(self, sender_id: str, receiver_id: str, message: Dict[str, Any]):
        """Send a message between agents"""
        if receiver_id in self.agents:
            self.communication_stats[sender_id]["messages_sent"] += 1
            self.communication_stats[receiver_id]["messages_received"] += 1
            self.communication_stats[sender_id]["last_communication"] = datetime.now()
            self.communication_stats[receiver_id]["last_communication"] = datetime.now()
            
            # In a real implementation, this would use the network protocol
            # For now, we'll simulate by storing the message in the receiver's queue
            receiver_agent = self.agents[receiver_id]
            if not hasattr(receiver_agent, 'message_queue'):
                receiver_agent.message_queue = []
            receiver_agent.message_queue.append({
                "sender": sender_id,
                "message": message,
                "timestamp": datetime.now()
            })
            self.logger.info(f"Message sent from {sender_id} to {receiver_id}")
    
    def broadcast_message(self, sender_id: str, message: Dict[str, Any]):
        """Broadcast a message to all agents in the network"""
        for agent_id in self.agents:
            if agent_id != sender_id:
                self.send_message(sender_id, agent_id, message)
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the network"""
        return {
            "name": self.name,
            "protocol": self.protocol.value,
            "agent_count": len(self.agents),
            "total_connections": sum(len(conns) for conns in self.connections.values()),
            "communication_stats": self.communication_stats
        }


class AdvancedAgentBuilder:
    """Advanced builder for creating complex agent networks and configurations"""
    
    def __init__(self):
        self.logger = setup_logger("AdvancedAgentBuilder")
        self.agent_templates = {}
        self.agent_networks = {}
        self.created_agents = {}
        self.agent_id_counter = 0
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.metrics_collector = {}
        self.task_queues = {}
        self.resource_manager = {}
        
        # Initialize default templates
        self._init_default_templates()
    
    def _init_default_templates(self):
        """Initialize default agent templates"""
        # Developer agent template
        developer_template = AgentTemplate(
            name="developer",
            description="Agent specialized for development tasks",
            agent_type="code_writer",
            capabilities=[
                AgentCapability.CODE_WRITING,
                AgentCapability.CODE_REVIEW,
                AgentCapability.TESTING,
                AgentCapability.ANALYSIS
            ],
            default_config={
                "language": "python",
                "framework": "none",
                "style_guide": "PEP8"
            },
            resources=AgentResource(
                cpu_cores=2,
                memory_mb=1024,
                max_concurrent_tasks=2
            )
        )
        self.agent_templates["developer"] = developer_template
        
        # Tester agent template
        tester_template = AgentTemplate(
            name="tester",
            description="Agent specialized for testing tasks",
            agent_type="tester",
            capabilities=[
                AgentCapability.TESTING,
                AgentCapability.ANALYSIS,
                AgentCapability.VALIDATION
            ],
            default_config={
                "test_framework": "pytest",
                "coverage_threshold": 80
            },
            resources=AgentResource(
                cpu_cores=1,
                memory_mb=512,
                max_concurrent_tasks=1
            )
        )
        self.agent_templates["tester"] = tester_template
        
        # Manager agent template
        manager_template = AgentTemplate(
            name="manager",
            description="Agent specialized for task management",
            agent_type="task_manager",
            capabilities=[
                AgentCapability.DECOMPOSITION,
                AgentCapability.TASK_EXECUTION,
                AgentCapability.COMMUNICATION
            ],
            default_config={
                "max_subtasks": 10,
                "timeout": 300
            },
            resources=AgentResource(
                cpu_cores=1,
                memory_mb=256,
                max_concurrent_tasks=5
            )
        )
        self.agent_templates["manager"] = manager_template
    
    def create_agent_from_template(self, 
                                 template_name: str, 
                                 name: str = None, 
                                 overrides: Dict[str, Any] = None) -> BaseAgent:
        """Create an agent using a predefined template"""
        if template_name not in self.agent_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.agent_templates[template_name]
        config = template.create_agent_config(overrides)
        
        if name:
            config["name"] = name
        
        # This would call the basic agent builder in a real implementation
        # For now, we'll create a basic agent
        agent_id = f"{template.agent_type}_{self.agent_id_counter}"
        self.agent_id_counter += 1
        
        # Create agent based on type
        agent = self._create_agent_by_type(template.agent_type, config.get("name", f"Agent_{agent_id}"), config.get("description", ""))
        
        # Apply capabilities
        for capability in template.capabilities:
            agent.add_capability(capability)
        
        # Store agent
        self.created_agents[agent_id] = agent
        agent.id = agent_id
        
        # Initialize metrics
        self.metrics_collector[agent_id] = AgentMetrics()
        
        self.logger.info(f"Created agent {agent_id} from template {template_name}")
        return agent
    
    def _create_agent_by_type(self, agent_type: str, name: str, description: str) -> BaseAgent:
        """Create an agent instance based on type"""
        # This is a simplified implementation
        # In a real system, this would use the actual agent classes
        from src.agents.default_agent import DefaultAgent
        return DefaultAgent(name, description)
    
    def create_cooperative_network(self, 
                                 network_name: str, 
                                 agent_configs: List[Dict[str, Any]], 
                                 protocol: AgentNetworkProtocol = AgentNetworkProtocol.A2A_ASYNC) -> AgentNetwork:
        """Create a network of cooperative agents"""
        network = AgentNetwork(network_name, protocol)
        
        # Create all agents
        agents = []
        for config in agent_configs:
            agent = self.create_agent_from_config(config)
            agents.append(agent)
            network.add_agent(agent)
        
        # Connect all agents in a mesh topology
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    network.connect_agents(agent1.id, agent2.id)
        
        self.agent_networks[network_name] = network
        self.logger.info(f"Created cooperative network {network_name} with {len(agents)} agents")
        return network
    
    def create_agent_from_config(self, config: Dict[str, Any]) -> BaseAgent:
        """Create an agent from a configuration dictionary"""
        name = config.get("name", f"Agent_{uuid.uuid4().hex[:8]}")
        agent_type = config.get("type", "default")
        description = config.get("description", f"Agent of type {agent_type}")
        
        agent = self._create_agent_by_type(agent_type, name, description)
        
        # Apply capabilities from config
        capabilities = config.get("capabilities", [])
        for cap_name in capabilities:
            try:
                capability = AgentCapability(cap_name)
                agent.add_capability(capability)
            except ValueError:
                self.logger.warning(f"Unknown capability: {cap_name}")
        
        # Store agent
        agent_id = f"{agent_type}_{self.agent_id_counter}"
        self.agent_id_counter += 1
        self.created_agents[agent_id] = agent
        agent.id = agent_id
        
        # Initialize metrics
        self.metrics_collector[agent_id] = AgentMetrics()
        
        return agent
    
    def create_hierarchical_network(self, 
                                  hierarchy_config: Dict[str, Any]) -> AgentNetwork:
        """Create a hierarchical agent network"""
        network_name = hierarchy_config.get("name", "hierarchical_network")
        network = AgentNetwork(network_name, AgentNetworkProtocol.CENTRALIZED)
        
        # Create agents in hierarchy
        agents_by_level = {}
        
        for level_name, level_config in hierarchy_config.get("levels", {}).items():
            agents_by_level[level_name] = []
            
            for agent_config in level_config.get("agents", []):
                agent = self.create_agent_from_config(agent_config)
                network.add_agent(agent)
                agents_by_level[level_name].append(agent)
        
        # Establish hierarchical connections
        level_names = list(hierarchy_config.get("levels", {}).keys())
        for i in range(len(level_names) - 1):
            current_level = level_names[i]
            next_level = level_names[i + 1]
            
            for current_agent in agents_by_level[current_level]:
                for next_agent in agents_by_level[next_level]:
                    network.connect_agents(current_agent.id, next_agent.id)
        
        self.agent_networks[network_name] = network
        return network
    
    def deploy_agent_network(self, network_name: str, deployment_config: Dict[str, Any]) -> bool:
        """Deploy an agent network to specified environment"""
        if network_name not in self.agent_networks:
            self.logger.error(f"Network {network_name} not found for deployment")
            return False
        
        network = self.agent_networks[network_name]
        self.logger.info(f"Deploying network {network_name} with config: {deployment_config}")
        
        # In a real implementation, this would deploy agents to specified environment
        # For now, we'll just log the deployment
        deployment_targets = deployment_config.get("targets", ["local"])
        deployment_strategy = deployment_config.get("strategy", "sequential")
        
        self.logger.info(f"Deploying to targets: {deployment_targets} using {deployment_strategy} strategy")
        
        # Update agent states to reflect deployment
        for agent_id in network.agents:
            agent = network.agents[agent_id]
            # In a real system, this would update the agent's deployment status
            pass
        
        return True
    
    def monitor_agent_network(self, network_name: str) -> Dict[str, Any]:
        """Monitor the status and performance of an agent network"""
        if network_name not in self.agent_networks:
            raise ValueError(f"Network {network_name} not found")
        
        network = self.agent_networks[network_name]
        stats = network.get_network_stats()
        
        # Add agent-specific metrics
        agent_metrics = {}
        for agent_id, agent in network.agents.items():
            if agent_id in self.metrics_collector:
                metrics = self.metrics_collector[agent_id]
                agent_metrics[agent_id] = {
                    "state": agent.status.value if hasattr(agent, 'status') else "unknown",
                    "tasks_completed": metrics.tasks_completed,
                    "tasks_failed": metrics.tasks_failed,
                    "average_task_duration": metrics.average_task_duration,
                    "last_activity": metrics.last_activity.isoformat() if metrics.last_activity else None
                }
        
        stats["agent_metrics"] = agent_metrics
        stats["timestamp"] = datetime.now().isoformat()
        
        return stats
    
    def optimize_agent_network(self, network_name: str, optimization_config: Dict[str, Any]) -> bool:
        """Optimize an agent network based on performance metrics"""
        if network_name not in self.agent_networks:
            raise ValueError(f"Network {network_name} not found")
        
        network = self.agent_networks[network_name]
        
        # Determine optimization strategy
        strategy = optimization_config.get("strategy", "load_balancing")
        
        if strategy == "load_balancing":
            return self._optimize_load_balancing(network)
        elif strategy == "resource_allocation":
            return self._optimize_resource_allocation(network)
        elif strategy == "communication_efficiency":
            return self._optimize_communication_efficiency(network)
        else:
            self.logger.warning(f"Unknown optimization strategy: {strategy}")
            return False
    
    def _optimize_load_balancing(self, network: AgentNetwork) -> bool:
        """Optimize network for load balancing"""
        # Count tasks per agent
        task_counts = {}
        for agent_id in network.agents:
            # In a real implementation, this would check actual task load
            task_counts[agent_id] = 0
        
        # Find agents with highest and lowest loads
        if task_counts:
            max_load_agent = max(task_counts, key=task_counts.get)
            min_load_agent = min(task_counts, key=task_counts.get)
            
            # If load difference is significant, consider rebalancing
            load_difference = task_counts[max_load_agent] - task_counts[min_load_agent]
            if load_difference > 2:  # Threshold for rebalancing
                self.logger.info(f"Load rebalancing needed: {max_load_agent} vs {min_load_agent}")
        
        return True
    
    def _optimize_resource_allocation(self, network: AgentNetwork) -> bool:
        """Optimize network for resource allocation"""
        # Check resource usage per agent
        for agent_id, agent in network.agents.items():
            # In a real implementation, this would check actual resource usage
            # and potentially reallocate resources
            pass
        
        return True
    
    def _optimize_communication_efficiency(self, network: AgentNetwork) -> bool:
        """Optimize network for communication efficiency"""
        # Analyze communication patterns
        for agent_id in network.agents:
            comm_stats = network.communication_stats[agent_id]
            messages_sent = comm_stats["messages_sent"]
            messages_received = comm_stats["messages_received"]
            
            # Log communication patterns for analysis
            self.logger.debug(f"Agent {agent_id}: {messages_sent} sent, {messages_received} received")
        
        return True
    
    def scale_agent_network(self, 
                          network_name: str, 
                          scaling_config: Dict[str, Any]) -> bool:
        """Scale an agent network up or down based on demand"""
        if network_name not in self.agent_networks:
            raise ValueError(f"Network {network_name} not found")
        
        network = self.agent_networks[network_name]
        
        scale_up = scaling_config.get("scale_up", True)
        agent_type = scaling_config.get("agent_type", "default")
        count = scaling_config.get("count", 1)
        
        if scale_up:
            # Add more agents of specified type
            for i in range(count):
                new_agent_config = {
                    "type": agent_type,
                    "name": f"{agent_type}_scaled_{uuid.uuid4().hex[:8]}"
                }
                new_agent = self.create_agent_from_config(new_agent_config)
                network.add_agent(new_agent)
                self.logger.info(f"Added agent {new_agent.id} to network {network_name}")
        else:
            # Remove agents (in a real system, this would be more sophisticated)
            agent_ids = list(network.agents.keys())
            for i in range(min(count, len(agent_ids))):
                agent_id = agent_ids[i]
                del network.agents[agent_id]
                del network.connections[agent_id]
                del network.communication_stats[agent_id]
                if agent_id in self.created_agents:
                    del self.created_agents[agent_id]
                if agent_id in self.metrics_collector:
                    del self.metrics_collector[agent_id]
                self.logger.info(f"Removed agent {agent_id} from network {network_name}")
        
        return True
    
    def backup_agent_network(self, network_name: str, backup_path: str) -> bool:
        """Create a backup of an agent network configuration"""
        if network_name not in self.agent_networks:
            raise ValueError(f"Network {network_name} not found")
        
        network = self.agent_networks[network_name]
        
        backup_data = {
            "network_name": network.name,
            "protocol": network.protocol.value,
            "agents": {},
            "connections": network.connections,
            "backup_timestamp": datetime.now().isoformat()
        }
        
        # Backup agent configurations
        for agent_id, agent in network.agents.items():
            backup_data["agents"][agent_id] = {
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "capabilities": [cap.value for cap in agent.capabilities],
                "status": agent.status.value if hasattr(agent, 'status') else "unknown"
            }
        
        # Write backup to file
        backup_file = Path(backup_path) / f"{network_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        self.logger.info(f"Backup created: {backup_file}")
        return True
    
    def restore_agent_network(self, backup_path: str, new_network_name: str = None) -> AgentNetwork:
        """Restore an agent network from backup"""
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)
        
        network_name = new_network_name or backup_data["network_name"]
        protocol = AgentNetworkProtocol(backup_data["protocol"])
        
        network = AgentNetwork(network_name, protocol)
        
        # Restore agents
        for agent_id, agent_data in backup_data["agents"].items():
            # Create agent from backup data
            agent = self._create_agent_by_type(
                agent_data.get("type", "default"),
                agent_data["name"],
                agent_data["description"]
            )
            
            # Restore capabilities
            for cap_name in agent_data["capabilities"]:
                try:
                    capability = AgentCapability(cap_name)
                    agent.add_capability(capability)
                except ValueError:
                    self.logger.warning(f"Unknown capability in backup: {cap_name}")
            
            agent.id = agent_id
            network.add_agent(agent)
        
        # Restore connections
        network.connections = backup_data["connections"]
        
        self.agent_networks[network_name] = network
        self.logger.info(f"Network restored from backup: {backup_path}")
        return network
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics"""
        total_agents = sum(len(network.agents) for network in self.agent_networks.values())
        total_networks = len(self.agent_networks)
        
        # Calculate overall metrics
        total_tasks_completed = sum(metrics.tasks_completed for metrics in self.metrics_collector.values())
        total_tasks_failed = sum(metrics.tasks_failed for metrics in self.metrics_collector.values())
        
        return {
            "total_agents": total_agents,
            "total_networks": total_networks,
            "total_tasks_completed": total_tasks_completed,
            "total_tasks_failed": total_tasks_failed,
            "agent_templates": list(self.agent_templates.keys()),
            "network_protocols": [net.protocol.value for net in self.agent_networks.values()],
            "timestamp": datetime.now().isoformat()
        }
    
    def cleanup_inactive_agents(self, max_idle_time_minutes: int = 60) -> int:
        """Clean up agents that have been inactive for too long"""
        cutoff_time = datetime.now() - timedelta(minutes=max_idle_time_minutes)
        agents_to_remove = []
        
        for agent_id, metrics in self.metrics_collector.items():
            if metrics.last_activity and metrics.last_activity < cutoff_time:
                agents_to_remove.append(agent_id)
        
        for agent_id in agents_to_remove:
            # Remove from all networks
            for network in self.agent_networks.values():
                if agent_id in network.agents:
                    del network.agents[agent_id]
                    if agent_id in network.connections:
                        del network.connections[agent_id]
                    if agent_id in network.communication_stats:
                        del network.communication_stats[agent_id]
            
            # Remove from main collections
            if agent_id in self.created_agents:
                del self.created_agents[agent_id]
            if agent_id in self.metrics_collector:
                del self.metrics_collector[agent_id]
        
        self.logger.info(f"Cleaned up {len(agents_to_remove)} inactive agents")
        return len(agents_to_remove)
    
    def export_network_schema(self, network_name: str) -> Dict[str, Any]:
        """Export the schema of an agent network for documentation or analysis"""
        if network_name not in self.agent_networks:
            raise ValueError(f"Network {network_name} not found")
        
        network = self.agent_networks[network_name]
        
        schema = {
            "network_name": network.name,
            "protocol": network.protocol.value,
            "agents": {},
            "connections": network.connections,
            "topology": self._analyze_topology(network),
            "export_timestamp": datetime.now().isoformat()
        }
        
        for agent_id, agent in network.agents.items():
            schema["agents"][agent_id] = {
                "type": getattr(agent, 'agent_type', 'unknown'),
                "capabilities": [cap.value for cap in agent.capabilities],
                "role": self._infer_agent_role(agent),
                "resource_allocation": self._get_agent_resource_allocation(agent_id)
            }
        
        return schema
    
    def _analyze_topology(self, network: AgentNetwork) -> Dict[str, Any]:
        """Analyze the network topology"""
        topology = {
            "type": "mesh",  # Could be star, ring, tree, etc.
            "nodes": len(network.agents),
            "edges": sum(len(conns) for conns in network.connections.values()) // 2,
            "density": 0.0,
            "centrality": {}
        }
        
        # Calculate network density
        n = len(network.agents)
        if n > 1:
            max_edges = n * (n - 1) // 2
            actual_edges = topology["edges"]
            topology["density"] = actual_edges / max_edges if max_edges > 0 else 0.0
        
        # Calculate centrality for each node (simplified)
        for agent_id in network.agents:
            connections = len(network.connections[agent_id])
            topology["centrality"][agent_id] = connections / (n - 1) if n > 1 else 0.0
        
        return topology
    
    def _infer_agent_role(self, agent: BaseAgent) -> str:
        """Infer the role of an agent based on its capabilities"""
        capabilities = [cap.value for cap in agent.capabilities]
        
        if AgentCapability.TASK_EXECUTION.value in capabilities and AgentCapability.COMMUNICATION.value in capabilities:
            return AgentRole.MANAGER.value
        elif AgentCapability.DECOMPOSITION.value in capabilities:
            return AgentRole.PLANNER.value
        elif AgentCapability.COMMUNICATION.value in capabilities:
            return AgentRole.COMMUNICATOR.value
        elif AgentCapability.ANALYSIS.value in capabilities:
            return AgentRole.ANALYST.value
        elif AgentCapability.VALIDATION.value in capabilities:
            return AgentRole.VALIDATOR.value
        else:
            return AgentRole.WORKER.value
    
    def _get_agent_resource_allocation(self, agent_id: str) -> Dict[str, Any]:
        """Get resource allocation for an agent"""
        # In a real implementation, this would return actual resource allocation
        return {
            "cpu_cores": 1,
            "memory_mb": 512,
            "disk_space_mb": 100
        }


class AgentLifecycleManager:
    """Manages the lifecycle of agents from creation to termination"""
    
    def __init__(self, builder: AdvancedAgentBuilder):
        self.builder = builder
        self.lifecycle_hooks = {}
        self.agent_states = {}
        self.state_transitions = {}
        self.logger = setup_logger("AgentLifecycleManager")
    
    def register_lifecycle_hook(self, state: AgentState, hook: Callable[[BaseAgent], None]):
        """Register a hook function to be called when an agent enters a specific state"""
        if state not in self.lifecycle_hooks:
            self.lifecycle_hooks[state] = []
        self.lifecycle_hooks[state].append(hook)
    
    def transition_agent_state(self, agent_id: str, new_state: AgentState) -> bool:
        """Transition an agent to a new state"""
        if agent_id not in self.builder.created_agents:
            self.logger.error(f"Agent {agent_id} not found")
            return False
        
        old_state = self.agent_states.get(agent_id, AgentState.CREATED)
        self.agent_states[agent_id] = new_state
        
        # Call lifecycle hooks for the new state
        if new_state in self.lifecycle_hooks:
            agent = self.builder.created_agents[agent_id]
            for hook in self.lifecycle_hooks[new_state]:
                try:
                    hook(agent)
                except Exception as e:
                    self.logger.error(f"Error in lifecycle hook for state {new_state}: {e}")
        
        self.logger.info(f"Agent {agent_id} transitioned from {old_state.value} to {new_state.value}")
        
        # Record state transition
        if agent_id not in self.state_transitions:
            self.state_transitions[agent_id] = []
        self.state_transitions[agent_id].append({
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    def initialize_agent(self, agent_id: str) -> bool:
        """Initialize an agent"""
        return self.transition_agent_state(agent_id, AgentState.INITIALIZED)
    
    def start_agent(self, agent_id: str) -> bool:
        """Start an agent (transition to IDLE state)"""
        return self.transition_agent_state(agent_id, AgentState.IDLE)
    
    def pause_agent(self, agent_id: str) -> bool:
        """Pause an agent"""
        return self.transition_agent_state(agent_id, AgentState.PAUSED)
    
    def resume_agent(self, agent_id: str) -> bool:
        """Resume a paused agent"""
        return self.transition_agent_state(agent_id, AgentState.IDLE)
    
    def terminate_agent(self, agent_id: str) -> bool:
        """Terminate an agent"""
        return self.transition_agent_state(agent_id, AgentState.TERMINATED)
    
    def get_agent_lifecycle_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get the lifecycle history for an agent"""
        return self.state_transitions.get(agent_id, [])


# Utility functions for advanced agent operations

def create_agent_with_retry(builder: AdvancedAgentBuilder, 
                           config: Dict[str, Any], 
                           max_retries: int = 3) -> Optional[BaseAgent]:
    """Create an agent with retry logic"""
    for attempt in range(max_retries):
        try:
            return builder.create_agent_from_config(config)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None


def execute_task_with_agents(network: AgentNetwork, 
                           task: Dict[str, Any], 
                           agent_selector: Callable[[AgentNetwork], List[str]] = None) -> Dict[str, Any]:
    """Execute a task using agents in a network"""
    if agent_selector is None:
        # Default selector: use all available agents
        available_agents = [aid for aid, agent in network.agents.items() 
                           if agent.status.value == "idle" if hasattr(agent, 'status')]
    else:
        available_agents = agent_selector(network)
    
    if not available_agents:
        return {"status": "error", "message": "No available agents"}
    
    # Select an agent to execute the task
    selected_agent_id = available_agents[0]
    selected_agent = network.agents[selected_agent_id]
    
    # In a real implementation, this would actually execute the task
    # For now, we'll simulate the execution
    result = {
        "status": "completed",
        "agent_id": selected_agent_id,
        "task_id": task.get("id", "unknown"),
        "result": f"Task executed by agent {selected_agent_id}",
        "execution_time": 0.1  # seconds
    }
    
    return result


def distribute_task_among_agents(network: AgentNetwork, 
                               task: Dict[str, Any], 
                               distribution_strategy: str = "round_robin") -> List[Dict[str, Any]]:
    """Distribute a task among multiple agents"""
    results = []
    
    if distribution_strategy == "round_robin":
        agent_ids = list(network.agents.keys())
        if not agent_ids:
            return [{"status": "error", "message": "No agents available"}]
        
        # Send task to each agent
        for agent_id in agent_ids:
            agent_task = copy.deepcopy(task)
            agent_task["target_agent"] = agent_id
            # Simulate task execution
            result = {
                "status": "completed",
                "agent_id": agent_id,
                "task_id": task.get("id", "unknown"),
                "result": f"Task part processed by agent {agent_id}",
                "part_id": agent_id
            }
            results.append(result)
    
    return results


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from multiple agents"""
    if not results:
        return {"status": "error", "message": "No results to aggregate"}
    
    # Check if all results are successful
    all_successful = all(r.get("status") == "completed" for r in results)
    
    aggregated = {
        "status": "completed" if all_successful else "partial",
        "total_results": len(results),
        "successful_results": len([r for r in results if r.get("status") == "completed"]),
        "failed_results": len([r for r in results if r.get("status") == "error"]),
        "individual_results": results
    }
    
    return aggregated


def validate_agent_network_configuration(network_config: Dict[str, Any]) -> List[str]:
    """Validate an agent network configuration"""
    errors = []
    
    # Check required fields
    required_fields = ["name", "agents"]
    for field in required_fields:
        if field not in network_config:
            errors.append(f"Missing required field: {field}")
    
    # Validate agents configuration
    if "agents" in network_config:
        agents = network_config["agents"]
        if not isinstance(agents, list):
            errors.append("Agents must be a list")
        else:
            for i, agent_config in enumerate(agents):
                if not isinstance(agent_config, dict):
                    errors.append(f"Agent config at index {i} must be a dictionary")
                elif "type" not in agent_config:
                    errors.append(f"Agent config at index {i} missing 'type' field")
    
    return errors


def create_monitoring_dashboard_data(builder: AdvancedAgentBuilder) -> Dict[str, Any]:
    """Create data for a monitoring dashboard"""
    system_metrics = builder.get_system_metrics()
    
    dashboard_data = {
        "system_metrics": system_metrics,
        "agent_networks": {},
        "recent_activity": [],
        "performance_trends": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Gather data for each network
    for network_name, network in builder.agent_networks.items():
        dashboard_data["agent_networks"][network_name] = {
            "agent_count": len(network.agents),
            "connection_count": sum(len(conns) for conns in network.connections.values()) // 2,
            "protocol": network.protocol.value,
            "status": "active"  # In a real system, this would check actual status
        }
    
    # Add some recent activity (simulated)
    for i in range(5):
        dashboard_data["recent_activity"].append({
            "timestamp": (datetime.now() - timedelta(minutes=i*10)).isoformat(),
            "event": f"Task completed by agent in network",
            "network": "default"
        })
    
    return dashboard_data


# Example usage and testing functions

def example_usage():
    """Example of how to use the AdvancedAgentBuilder"""
    builder = AdvancedAgentBuilder()
    
    # Create a simple agent from template
    agent = builder.create_agent_from_template("developer", name="MyDevAgent")
    print(f"Created agent: {agent.name} with ID: {agent.id}")
    
    # Create a cooperative network
    network_config = [
        {"type": "task_manager", "name": "Manager1"},
        {"type": "code_writer", "name": "Coder1", "capabilities": ["code_writing", "analysis"]},
        {"type": "tester", "name": "Tester1", "capabilities": ["testing", "analysis"]}
    ]
    
    network = builder.create_cooperative_network("dev_network", network_config)
    print(f"Created network with {len(network.agents)} agents")
    
    # Get system metrics
    metrics = builder.get_system_metrics()
    print(f"System metrics: {metrics}")


if __name__ == "__main__":
    example_usage()