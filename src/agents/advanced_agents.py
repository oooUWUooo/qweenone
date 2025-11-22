#!/usr/bin/env python3
"""
Advanced Agents System
This module provides implementations of specialized agents with advanced capabilities
for complex tasks, learning, and adaptation.
"""

from typing import Dict, Any, List, Optional, Callable, Union, Tuple
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
import heapq
from collections import defaultdict, deque
import random
import math
from dataclasses import asdict

from src.agents.base_agent import BaseAgent, AgentCapability
from src.utils.logger import setup_logger


class LearningStrategy(Enum):
    """Learning strategies for adaptive agents"""
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    ONLINE = "online"
    BATCH = "batch"


class AdaptationLevel(Enum):
    """Levels of adaptation for agents"""
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"


class AgentMemoryType(Enum):
    """Types of memory for agents"""
    EPISODIC = "episodic"  # Short-term memory for specific episodes
    SEMANTIC = "semantic"  # Long-term memory for facts and concepts
    PROCEDURAL = "procedural"  # Memory for procedures and skills
    WORKING = "working"  # Active memory for current tasks


@dataclass
class AgentMemory:
    """Memory structure for agents"""
    memory_type: AgentMemoryType
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0  # 0.0 to 1.0
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)


@dataclass
class AgentLearningHistory:
    """Learning history for an agent"""
    task_id: str
    task_result: str
    reward: float
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAdaptationMetrics:
    """Metrics for agent adaptation"""
    learning_rate: float = 0.0
    adaptation_score: float = 0.0
    performance_improvement: float = 0.0
    error_rate: float = 1.0
    efficiency_score: float = 0.0
    last_adaptation: datetime = field(default_factory=datetime.now)


class AdvancedBaseAgent(BaseAgent):
    """Base class for advanced agents with learning and adaptation capabilities"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.memory = defaultdict(list)  # Memory organized by type
        self.learning_history = []  # Historical learning data
        self.adaptation_metrics = AgentAdaptationMetrics()
        self.learning_strategy = LearningStrategy.ONLINE
        self.adaptation_level = AdaptationLevel.BASIC
        self.performance_history = []  # Track performance over time
        self.knowledge_base = {}  # Agent's knowledge
        self.experience_points = 0  # For skill progression
        self.skill_level = 1  # Agent's skill level
        self.personality_traits = {}  # Agent's personality
        self.communication_style = "formal"  # How agent communicates
        self.task_preferences = []  # Preferred task types
        self.learning_callbacks = []  # Callbacks for learning events
        self.adaptation_threshold = 0.7  # Threshold for adaptation
        self.exploration_rate = 0.1  # Rate of exploration vs exploitation
    
    def add_to_memory(self, memory_type: AgentMemoryType, content: Any, importance: float = 1.0):
        """Add content to agent's memory"""
        memory_entry = AgentMemory(
            memory_type=memory_type,
            content=content,
            importance=importance
        )
        self.memory[memory_type.value].append(memory_entry)
        
        # Keep memory size manageable
        if len(self.memory[memory_type.value]) > 1000:  # Limit to 1000 entries per type
            self.memory[memory_type.value] = sorted(
                self.memory[memory_type.value], 
                key=lambda x: x.importance, 
                reverse=True
            )[:500]  # Keep top 500 most important entries
    
    def retrieve_from_memory(self, memory_type: AgentMemoryType, query: str = None, limit: int = 10) -> List[AgentMemory]:
        """Retrieve content from agent's memory"""
        memories = self.memory[memory_type.value]
        
        if query:
            # Simple keyword matching (in a real system, this would be more sophisticated)
            memories = [m for m in memories if query.lower() in str(m.content).lower()]
        
        # Sort by importance and access frequency
        sorted_memories = sorted(
            memories,
            key=lambda x: (x.importance * 2 + (x.access_count / 10.0)),
            reverse=True
        )
        
        # Update access counts
        for memory in sorted_memories[:limit]:
            memory.access_count += 1
            memory.last_access = datetime.now()
        
        return sorted_memories[:limit]
    
    def store_learning_experience(self, task_id: str, task_result: str, reward: float, context: Dict[str, Any] = None):
        """Store a learning experience"""
        experience = AgentLearningHistory(
            task_id=task_id,
            task_result=task_result,
            reward=reward,
            context=context or {}
        )
        self.learning_history.append(experience)
        
        # Update adaptation metrics
        self._update_adaptation_metrics(reward)
        
        # Trigger learning callbacks
        for callback in self.learning_callbacks:
            try:
                callback(self, experience)
            except Exception as e:
                self.logger.error(f"Error in learning callback: {e}")
    
    def _update_adaptation_metrics(self, reward: float):
        """Update adaptation metrics based on reward"""
        self.adaptation_metrics.last_adaptation = datetime.now()
        
        # Calculate learning rate based on recent experiences
        recent_experiences = [
            exp for exp in self.learning_history 
            if (datetime.now() - exp.timestamp).days <= 7
        ]
        
        if recent_experiences:
            avg_reward = sum(exp.reward for exp in recent_experiences) / len(recent_experiences)
            self.adaptation_metrics.learning_rate = avg_reward
            
            # Calculate performance improvement
            if len(recent_experiences) > 1:
                recent_rewards = [exp.reward for exp in recent_experiences[-5:]]
                if len(recent_rewards) > 1:
                    improvement = recent_rewards[-1] - recent_rewards[0]
                    self.adaptation_metrics.performance_improvement = improvement
    
    def adapt_behavior(self):
        """Adapt agent's behavior based on learning history"""
        if len(self.learning_history) < 5:  # Need sufficient history
            return
        
        # Analyze recent performance
        recent_experiences = self.learning_history[-10:]  # Last 10 experiences
        avg_reward = sum(exp.reward for exp in recent_experiences) / len(recent_experiences)
        
        # Adjust exploration rate based on performance
        if avg_reward > self.adaptation_threshold:
            # Good performance - reduce exploration
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
        else:
            # Poor performance - increase exploration
            self.exploration_rate = min(0.5, self.exploration_rate * 1.05)
        
        # Update adaptation level based on performance
        if avg_reward > 0.8:
            self.adaptation_level = AdaptationLevel.MASTER
        elif avg_reward > 0.6:
            self.adaptation_level = AdaptationLevel.EXPERT
        elif avg_reward > 0.4:
            self.adaptation_level = AdaptationLevel.ADVANCED
        else:
            self.adaptation_level = AdaptationLevel.BASIC
        
        # Adjust learning strategy based on performance patterns
        if len(recent_experiences) > 2:
            reward_trend = recent_experiences[-1].reward - recent_experiences[0].reward
            if reward_trend > 0.1:  # Improving
                self.learning_strategy = LearningStrategy.ONLINE
            elif reward_trend < -0.1:  # Deteriorating
                self.learning_strategy = LearningStrategy.BATCH
    
    def get_contextual_response(self, context: Dict[str, Any], query: str) -> str:
        """Generate a response based on context and memory"""
        # Retrieve relevant memories
        relevant_memories = self.retrieve_from_memory(AgentMemoryType.SEMANTIC, query, limit=5)
        
        # Combine context with relevant memories
        combined_context = {
            "query": query,
            "relevant_info": [mem.content for mem in relevant_memories],
            "current_context": context,
            "agent_state": {
                "name": self.name,
                "capabilities": [cap.value for cap in self.capabilities],
                "adaptation_level": self.adaptation_level.value,
                "exploration_rate": self.exploration_rate
            }
        }
        
        # In a real implementation, this would use an LLM or similar
        # For now, we'll return a formatted response
        response = f"Based on my knowledge and context, I can help with: {query}. "
        response += f"I have {len(relevant_memories)} relevant pieces of information. "
        response += f"My adaptation level is {self.adaptation_level.value}."
        
        return response
    
    def register_learning_callback(self, callback: Callable[[BaseAgent, AgentLearningHistory], None]):
        """Register a callback for learning events"""
        self.learning_callbacks.append(callback)
    
    def calculate_task_efficiency(self, task: Dict[str, Any]) -> float:
        """Calculate the efficiency of handling a specific task"""
        # Base efficiency calculation
        efficiency = 0.5  # Start with neutral efficiency
        
        # Adjust based on agent capabilities
        required_caps = task.get("required_capabilities", [])
        matching_caps = [cap for cap in required_caps if cap in [c.value for c in self.capabilities]]
        if required_caps:
            efficiency += (len(matching_caps) / len(required_caps)) * 0.3
        
        # Adjust based on experience
        similar_task_count = sum(1 for exp in self.learning_history 
                                if exp.task_result == task.get("type", "unknown"))
        efficiency += min(similar_task_count * 0.05, 0.2)  # Up to 20% boost from experience
        
        # Adjust based on adaptation level
        adaptation_bonus = {
            AdaptationLevel.BASIC: 0.0,
            AdaptationLevel.ADVANCED: 0.1,
            AdaptationLevel.EXPERT: 0.2,
            AdaptationLevel.MASTER: 0.3
        }[self.adaptation_level]
        efficiency += adaptation_bonus
        
        return min(1.0, max(0.0, efficiency))  # Clamp between 0 and 1


class LearningAgent(AdvancedBaseAgent):
    """Agent with advanced learning capabilities"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.add_capability(AgentCapability.ANALYSIS)
        self.add_capability(AgentCapability.DATA_PROCESSING)
        self.agent_type = "learning_agent"
        self.learning_model = None  # Would be a ML model in real implementation
        self.training_data = []
        self.validation_data = []
        self.model_accuracy = 0.0
        self.model_complexity = 0  # How complex the model is
        self.feature_importance = {}  # Importance of different features
        self.overfitting_detector = None
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        self.update_status("working")
        self.logger = setup_logger(f"LearningAgent_{self.id}")
        
        self.logger.info(f"Learning agent executing task: {task.get('title', 'Unknown task')}")
        
        # Store task in memory
        self.add_to_memory(AgentMemoryType.EPISODIC, task, importance=0.8)
        
        # Calculate task efficiency
        efficiency = self.calculate_task_efficiency(task)
        
        # Perform learning-specific operations
        result = await self._perform_learning_task(task)
        
        # Store learning experience
        reward = self._calculate_reward(result, task)
        self.store_learning_experience(task.get("id", "unknown"), "completed", reward, task)
        
        # Adapt behavior based on experience
        self.adapt_behavior()
        
        final_result = {
            "status": "completed",
            "agent_id": self.id,
            "task_id": task.get("id"),
            "result": result,
            "efficiency_score": efficiency,
            "learning_improvement": self.adaptation_metrics.performance_improvement,
            "adaptation_level": self.adaptation_level.value
        }
        
        self.update_status("idle")
        return final_result
    
    async def _perform_learning_task(self, task: Dict[str, Any]) -> Any:
        """Perform the specific learning task"""
        task_type = task.get("type", "unknown")
        
        if task_type == "train_model":
            return await self._train_model(task)
        elif task_type == "analyze_data":
            return await self._analyze_data(task)
        elif task_type == "predict":
            return await self._make_prediction(task)
        elif task_type == "validate_model":
            return await self._validate_model(task)
        else:
            # Default learning task
            return f"Learning task of type {task_type} completed"
    
    async def _train_model(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Train a learning model"""
        # In a real implementation, this would train an actual ML model
        # For now, we'll simulate the process
        
        training_data = task.get("training_data", [])
        validation_data = task.get("validation_data", [])
        
        # Store data
        self.training_data.extend(training_data)
        self.validation_data.extend(validation_data)
        
        # Simulate training process
        epochs = task.get("epochs", 10)
        for epoch in range(epochs):
            # Simulate training step
            time.sleep(0.01)  # Simulate processing time
            
            # Update progress
            progress = (epoch + 1) / epochs * 100
            self.update_progress(progress, f"Training epoch {epoch + 1}/{epochs}")
        
        # Calculate simulated accuracy
        self.model_accuracy = min(0.95, 0.5 + (len(training_data) * 0.001))
        
        return {
            "status": "trained",
            "model_accuracy": self.model_accuracy,
            "training_samples": len(training_data),
            "validation_samples": len(validation_data),
            "epochs_completed": epochs
        }
    
    async def _analyze_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data patterns"""
        data = task.get("data", [])
        
        # Store in memory
        self.add_to_memory(AgentMemoryType.SEMANTIC, {
            "analysis_type": "data_pattern",
            "data_summary": f"Analyzed {len(data)} data points",
            "timestamp": datetime.now().isoformat()
        }, importance=0.7)
        
        # Perform analysis (simulated)
        analysis_result = {
            "total_samples": len(data),
            "data_types": list(set(type(item).__name__ for item in data)),
            "patterns_identified": random.randint(1, 5),
            "anomalies_detected": random.randint(0, 2)
        }
        
        return analysis_result
    
    async def _make_prediction(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the trained model"""
        input_data = task.get("input_data", [])
        
        # Simulate prediction
        predictions = []
        for item in input_data:
            # Add some randomness to simulate prediction uncertainty
            prediction = {
                "input": item,
                "prediction": f"predicted_value_{random.randint(1, 100)}",
                "confidence": random.uniform(0.6, 0.95)
            }
            predictions.append(prediction)
        
        return {
            "predictions": predictions,
            "model_accuracy": self.model_accuracy,
            "prediction_count": len(predictions)
        }
    
    async def _validate_model(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the learning model"""
        validation_data = task.get("validation_data", [])
        
        # Simulate validation
        correct_predictions = 0
        for _ in validation_data:
            if random.random() < self.model_accuracy:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(validation_data) if validation_data else 0
        
        return {
            "validation_accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": len(validation_data),
            "model_status": "validated"
        }
    
    def _calculate_reward(self, result: Any, task: Dict[str, Any]) -> float:
        """Calculate reward for the task completion"""
        # Base reward calculation
        base_reward = 0.5
        
        # Task-specific adjustments
        task_type = task.get("type", "unknown")
        if task_type == "train_model":
            accuracy = result.get("model_accuracy", 0)
            base_reward = min(1.0, accuracy)  # Reward based on model accuracy
        elif task_type == "analyze_data":
            patterns = result.get("patterns_identified", 0)
            base_reward = min(1.0, 0.3 + (patterns * 0.1))
        elif task_type == "predict":
            avg_confidence = sum(p["confidence"] for p in result.get("predictions", [])) / len(result.get("predictions", [1])) if result.get("predictions") else 0
            base_reward = min(1.0, 0.4 + avg_confidence * 0.5)
        
        return base_reward


class AdaptiveAgent(AdvancedBaseAgent):
    """Agent with advanced adaptation capabilities"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.add_capability(AgentCapability.ANALYSIS)
        self.add_capability(AgentCapability.TROUBLESHOOTING)
        self.agent_type = "adaptive_agent"
        self.environment_context = {}  # Current environment context
        self.context_adaptation_rules = {}  # Rules for adapting to contexts
        self.performance_thresholds = {}  # Performance thresholds by context
        self.context_history = []  # History of contexts encountered
        self.adaptation_strategies = []  # Available adaptation strategies
        self.current_strategy = None  # Currently selected strategy
        self.strategy_effectiveness = {}  # Effectiveness of each strategy
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        self.update_status("working")
        self.logger = setup_logger(f"AdaptiveAgent_{self.id}")
        
        # Analyze current context
        context = self._analyze_context(task)
        
        # Select appropriate strategy based on context
        strategy = self._select_adaptation_strategy(context, task)
        self.current_strategy = strategy
        
        self.logger.info(f"Adaptive agent executing task with strategy '{strategy}': {task.get('title', 'Unknown task')}")
        
        # Store task in memory
        self.add_to_memory(AgentMemoryType.EPISODIC, {
            "task": task,
            "context": context,
            "strategy": strategy
        }, importance=0.9)
        
        # Execute task with selected strategy
        result = await self._execute_with_strategy(task, strategy)
        
        # Evaluate strategy effectiveness
        effectiveness = self._evaluate_strategy_effectiveness(result, task)
        self.strategy_effectiveness[strategy] = effectiveness
        
        # Store learning experience
        reward = self._calculate_adaptation_reward(result, task, effectiveness)
        self.store_learning_experience(task.get("id", "unknown"), "completed", reward, {
            "context": context,
            "strategy": strategy,
            "effectiveness": effectiveness
        })
        
        # Adapt behavior based on experience
        self.adapt_behavior()
        
        final_result = {
            "status": "completed",
            "agent_id": self.id,
            "task_id": task.get("id"),
            "result": result,
            "selected_strategy": strategy,
            "strategy_effectiveness": effectiveness,
            "context_adaptation_score": self.adaptation_metrics.adaptation_score
        }
        
        self.update_status("idle")
        return final_result
    
    def _analyze_context(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the current context for the task"""
        context = {
            "task_type": task.get("type", "unknown"),
            "task_complexity": task.get("complexity", "medium"),
            "required_capabilities": task.get("required_capabilities", []),
            "deadline": task.get("deadline"),
            "priority": task.get("priority", 1),
            "environment": task.get("environment", "default"),
            "resource_constraints": task.get("resource_constraints", {}),
            "stakeholders": task.get("stakeholders", [])
        }
        
        # Store context
        self.environment_context = context
        self.context_history.append({
            "context": context,
            "timestamp": datetime.now()
        })
        
        # Keep context history manageable
        if len(self.context_history) > 100:
            self.context_history = self.context_history[-50:]
        
        return context
    
    def _select_adaptation_strategy(self, context: Dict[str, Any], task: Dict[str, Any]) -> str:
        """Select the most appropriate adaptation strategy"""
        # Define available strategies
        if not self.adaptation_strategies:
            self.adaptation_strategies = [
                "optimization", "exploration", "conservative", 
                "aggressive", "collaborative", "independent"
            ]
        
        # Determine strategy based on context and past effectiveness
        task_type = context["task_type"]
        priority = context["priority"]
        
        # High priority tasks might need aggressive approach
        if priority >= 3:  # High or critical priority
            return "aggressive"
        
        # Complex tasks might need exploration
        if context["task_complexity"] == "high":
            return "exploration"
        
        # Try to use the most effective strategy for this task type
        best_strategy = None
        best_effectiveness = -1
        
        for strategy, effectiveness in self.strategy_effectiveness.items():
            if strategy not in self.strategy_effectiveness:
                continue
            if effectiveness > best_effectiveness:
                best_effectiveness = effectiveness
                best_strategy = strategy
        
        # If no strategy has proven effective, use default based on task type
        if best_strategy is None:
            strategy_mapping = {
                "computation": "optimization",
                "analysis": "exploration", 
                "communication": "collaborative",
                "validation": "conservative"
            }
            best_strategy = strategy_mapping.get(task_type, "optimization")
        
        return best_strategy
    
    async def _execute_with_strategy(self, task: Dict[str, Any], strategy: str) -> Any:
        """Execute task using the selected strategy"""
        if strategy == "optimization":
            return await self._execute_optimization_strategy(task)
        elif strategy == "exploration":
            return await self._execute_exploration_strategy(task)
        elif strategy == "conservative":
            return await self._execute_conservative_strategy(task)
        elif strategy == "aggressive":
            return await self._execute_aggressive_strategy(task)
        elif strategy == "collaborative":
            return await self._execute_collaborative_strategy(task)
        elif strategy == "independent":
            return await self._execute_independent_strategy(task)
        else:
            # Default strategy
            return await self._execute_default_strategy(task)
    
    async def _execute_optimization_strategy(self, task: Dict[str, Any]) -> Any:
        """Execute with optimization strategy"""
        # Focus on efficiency and resource optimization
        start_time = datetime.now()
        
        # Simulate optimized execution
        time.sleep(0.05)  # Simulate processing
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "result": f"Optimized execution of {task.get('title', 'task')}",
            "execution_time": execution_time,
            "resources_used": "minimal",
            "optimization_applied": True
        }
    
    async def _execute_exploration_strategy(self, task: Dict[str, Any]) -> Any:
        """Execute with exploration strategy"""
        # Try multiple approaches to find the best solution
        approaches_tried = []
        
        for i in range(3):  # Try 3 different approaches
            approach = f"approach_{i+1}"
            approaches_tried.append(approach)
            
            # Simulate approach execution
            time.sleep(0.02)
        
        return {
            "result": f"Exploration of {task.get('title', 'task')} with multiple approaches",
            "approaches_tried": approaches_tried,
            "best_approach": approaches_tried[0],
            "exploration_completed": True
        }
    
    async def _execute_conservative_strategy(self, task: Dict[str, Any]) -> Any:
        """Execute with conservative strategy"""
        # Use proven, safe methods
        return {
            "result": f"Conservative execution of {task.get('title', 'task')}",
            "method": "proven_approach",
            "risk_level": "low",
            "safety_measures": ["validation", "backup", "rollback"]
        }
    
    async def _execute_aggressive_strategy(self, task: Dict[str, Any]) -> Any:
        """Execute with aggressive strategy"""
        # Focus on speed and results, accept higher risk
        start_time = datetime.now()
        
        # Simulate fast execution
        time.sleep(0.01)  # Very fast execution
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "result": f"Aggressive execution of {task.get('title', 'task')}",
            "execution_time": execution_time,
            "speed_factor": "high",
            "risk_level": "medium"
        }
    
    async def _execute_collaborative_strategy(self, task: Dict[str, Any]) -> Any:
        """Execute with collaborative strategy"""
        # Coordinate with other agents
        collaborators = task.get("collaborators", [])
        
        collaboration_results = []
        for agent_id in collaborators:
            # Simulate collaboration
            collaboration_results.append({
                "agent_id": agent_id,
                "contribution": f"Collaboration with {agent_id}",
                "status": "completed"
            })
        
        return {
            "result": f"Collaborative execution of {task.get('title', 'task')}",
            "collaborators": collaborators,
            "collaboration_results": collaboration_results,
            "coordination_applied": True
        }
    
    async def _execute_independent_strategy(self, task: Dict[str, Any]) -> Any:
        """Execute with independent strategy"""
        # Work alone without coordination
        return {
            "result": f"Independent execution of {task.get('title', 'task')}",
            "method": "standalone",
            "coordination_required": False,
            "autonomy_level": "high"
        }
    
    async def _execute_default_strategy(self, task: Dict[str, Any]) -> Any:
        """Execute with default strategy"""
        # Standard execution approach
        return {
            "result": f"Default execution of {task.get('title', 'task')}",
            "method": "standard",
            "status": "completed"
        }
    
    def _evaluate_strategy_effectiveness(self, result: Any, task: Dict[str, Any]) -> float:
        """Evaluate how effective the strategy was"""
        # Calculate effectiveness based on various factors
        task_success = 1.0  # Assume success for now
        time_efficiency = 0.8  # Default efficiency
        resource_efficiency = 0.7  # Default efficiency
        
        # Calculate composite effectiveness score
        effectiveness = (task_success * 0.4 + time_efficiency * 0.3 + resource_efficiency * 0.3)
        
        return effectiveness
    
    def _calculate_adaptation_reward(self, result: Any, task: Dict[str, Any], effectiveness: float) -> float:
        """Calculate reward for adaptation"""
        base_reward = 0.5
        
        # Adjust based on effectiveness
        base_reward += effectiveness * 0.3
        
        # Task-specific adjustments
        priority = task.get("priority", 1)
        if priority >= 3:
            base_reward += 0.2  # Extra reward for high priority tasks
        
        return min(1.0, base_reward)  # Cap at 1.0


class SpecializedAgent(AdvancedBaseAgent):
    """Base class for specialized agents with domain-specific capabilities"""
    
    def __init__(self, name: str, description: str = "", domain: str = "general"):
        super().__init__(name, description)
        self.domain = domain
        self.domain_expertise = 0.0  # Level of expertise in the domain
        self.domain_knowledge = {}  # Knowledge specific to the domain
        self.domain_patterns = []  # Common patterns in the domain
        self.domain_best_practices = []  # Best practices for the domain
        self.domain_tools = []  # Tools specific to the domain
        self.domain_constraints = []  # Constraints specific to the domain
        self.specialization_level = 1  # Level of specialization
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        self.update_status("working")
        self.logger = setup_logger(f"SpecializedAgent_{self.id}")
        
        # Verify task is in our domain
        task_domain = task.get("domain", "general")
        if task_domain != self.domain and task_domain != "general":
            return {
                "status": "error",
                "message": f"Task domain '{task_domain}' does not match agent domain '{self.domain}'",
                "agent_id": self.id,
                "task_id": task.get("id")
            }
        
        self.logger.info(f"Specialized agent ({self.domain}) executing task: {task.get('title', 'Unknown task')}")
        
        # Store task in memory
        self.add_to_memory(AgentMemoryType.EPISODIC, task, importance=0.8)
        
        # Apply domain-specific processing
        result = await self._execute_domain_specific_task(task)
        
        # Calculate domain expertise improvement
        expertise_gain = self._calculate_expertise_gain(result, task)
        self.domain_expertise = min(1.0, self.domain_expertise + expertise_gain)
        
        # Store learning experience
        reward = self._calculate_domain_reward(result, task)
        self.store_learning_experience(task.get("id", "unknown"), "completed", reward, {
            "domain": self.domain,
            "expertise_level": self.domain_expertise
        })
        
        # Adapt behavior based on experience
        self.adapt_behavior()
        
        final_result = {
            "status": "completed",
            "agent_id": self.id,
            "task_id": task.get("id"),
            "result": result,
            "domain": self.domain,
            "domain_expertise": self.domain_expertise,
            "specialization_level": self.specialization_level
        }
        
        self.update_status("idle")
        return final_result
    
    async def _execute_domain_specific_task(self, task: Dict[str, Any]) -> Any:
        """Execute the task using domain-specific knowledge"""
        # This should be overridden by subclasses
        return f"Domain-specific execution of {task.get('title', 'task')} in {self.domain}"
    
    def _calculate_expertise_gain(self, result: Any, task: Dict[str, Any]) -> float:
        """Calculate the gain in domain expertise from this task"""
        # Base expertise gain
        gain = 0.01
        
        # Task complexity affects learning
        complexity = task.get("complexity", "medium")
        if complexity == "high":
            gain *= 2.0
        elif complexity == "low":
            gain *= 0.5
        
        # Successful completion gives bonus
        if isinstance(result, dict) and result.get("status") == "completed":
            gain *= 1.5
        
        return min(0.1, gain)  # Cap gain to prevent rapid expertise inflation
    
    def _calculate_domain_reward(self, result: Any, task: Dict[str, Any]) -> float:
        """Calculate reward for domain-specific task completion"""
        base_reward = 0.6  # Base reward for domain expertise
        
        # Task-specific adjustments
        if isinstance(result, dict) and result.get("status") == "completed":
            base_reward += 0.3  # Bonus for successful completion
        
        # Priority-based adjustment
        priority = task.get("priority", 1)
        if priority >= 3:
            base_reward += 0.1  # Extra for high priority tasks
        
        return min(1.0, base_reward)


class CodeSpecialist(SpecializedAgent):
    """Specialized agent for code-related tasks"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description, domain="code")
        self.add_capability(AgentCapability.CODE_WRITING)
        self.add_capability(AgentCapability.CODE_REVIEW)
        self.add_capability(AgentCapability.ANALYSIS)
        self.code_languages = ["python", "javascript", "java", "go"]  # Supported languages
        self.code_patterns = {}  # Known code patterns
        self.code_quality_metrics = {}  # Quality metrics
        self.testing_frameworks = ["pytest", "unittest", "junit"]  # Supported frameworks
        self.linting_tools = ["pylint", "eslint", "flake8"]  # Available tools
        self.code_complexity_threshold = 10  # Cyclomatic complexity threshold
    
    async def _execute_domain_specific_task(self, task: Dict[str, Any]) -> Any:
        """Execute code-specific task"""
        task_type = task.get("type", "unknown")
        
        if task_type == "write_code":
            return await self._write_code(task)
        elif task_type == "review_code":
            return await self._review_code(task)
        elif task_type == "debug_code":
            return await self._debug_code(task)
        elif task_type == "test_code":
            return await self._test_code(task)
        elif task_type == "optimize_code":
            return await self._optimize_code(task)
        else:
            return await super()._execute_domain_specific_task(task)
    
    async def _write_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Write code based on specifications"""
        language = task.get("language", "python")
        requirements = task.get("requirements", [])
        code_structure = task.get("structure", {})
        
        # Simulate code writing
        code_snippets = []
        for i, req in enumerate(requirements):
            snippet = {
                "id": f"snippet_{i+1}",
                "description": req,
                "code": f"def function_{i+1}():\n    # Implementation for {req}\n    pass",
                "language": language
            }
            code_snippets.append(snippet)
        
        return {
            "status": "completed",
            "code_snippets": code_snippets,
            "language": language,
            "total_snippets": len(code_snippets),
            "generated_by": self.name
        }
    
    async def _review_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Review code for quality and best practices"""
        code = task.get("code", "")
        language = task.get("language", "python")
        
        # Simulate code review
        issues = [
            {"type": "style", "severity": "low", "description": "Missing docstring"},
            {"type": "complexity", "severity": "medium", "description": "Function too complex"},
            {"type": "performance", "severity": "high", "description": "Inefficient algorithm"}
        ]
        
        # Store knowledge about code patterns
        self.domain_knowledge[f"pattern_{len(self.domain_knowledge)}"] = {
            "type": "code_issue",
            "language": language,
            "common_issues": [issue["type"] for issue in issues]
        }
        
        return {
            "status": "completed",
            "issues_found": issues,
            "code_quality_score": 7.5,  # Out of 10
            "suggestions": ["Add documentation", "Simplify complex functions", "Optimize algorithm"],
            "language": language
        }
    
    async def _debug_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Debug code and identify issues"""
        code = task.get("code", "")
        error_description = task.get("error", "")
        
        # Simulate debugging process
        debug_result = {
            "status": "completed",
            "issues_identified": [
                {"type": "logical_error", "location": "line_15", "description": "Potential null reference"},
                {"type": "syntax_error", "location": "line_23", "description": "Missing colon"}
            ],
            "suggested_fixes": [
                "Add null check before accessing object",
                "Add missing colon at end of if statement"
            ],
            "confidence_level": 0.85
        }
        
        return debug_result
    
    async def _test_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create and run tests for code"""
        code = task.get("code", "")
        framework = task.get("framework", "pytest")
        
        # Simulate test creation and execution
        test_results = {
            "framework": framework,
            "tests_created": 5,
            "tests_passed": 4,
            "tests_failed": 1,
            "coverage_percentage": 85.0,
            "test_results": [
                {"name": "test_function_1", "status": "passed"},
                {"name": "test_function_2", "status": "passed"},
                {"name": "test_function_3", "status": "passed"},
                {"name": "test_function_4", "status": "passed"},
                {"name": "test_function_5", "status": "failed", "error": "AssertionError"}
            ]
        }
        
        return test_results
    
    async def _optimize_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize code for performance"""
        code = task.get("code", "")
        
        # Simulate optimization
        optimization_result = {
            "status": "completed",
            "original_complexity": 15,
            "optimized_complexity": 8,
            "performance_improvement": 25.5,  # Percentage improvement
            "optimizations_applied": [
                "Algorithm optimization",
                "Memory usage reduction",
                "Loop optimization"
            ],
            "suggestions": ["Use more efficient data structures", "Cache computation results"]
        }
        
        return optimization_result


class DataSpecialist(SpecializedAgent):
    """Specialized agent for data-related tasks"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description, domain="data")
        self.add_capability(AgentCapability.DATA_PROCESSING)
        self.add_capability(AgentCapability.ANALYSIS)
        self.add_capability(AgentCapability.VALIDATION)
        self.data_formats = ["csv", "json", "xml", "parquet", "hdf5"]  # Supported formats
        self.data_sources = ["database", "api", "file", "stream"]  # Supported sources
        self.ml_frameworks = ["scikit-learn", "tensorflow", "pytorch"]  # ML frameworks
        self.visualization_tools = ["matplotlib", "seaborn", "plotly"]  # Visualization tools
        self.data_quality_metrics = {
            "completeness": 0.0,
            "accuracy": 0.0,
            "consistency": 0.0,
            "timeliness": 0.0
        }
    
    async def _execute_domain_specific_task(self, task: Dict[str, Any]) -> Any:
        """Execute data-specific task"""
        task_type = task.get("type", "unknown")
        
        if task_type == "process_data":
            return await self._process_data(task)
        elif task_type == "analyze_data":
            return await self._analyze_data(task)
        elif task_type == "clean_data":
            return await self._clean_data(task)
        elif task_type == "visualize_data":
            return await self._visualize_data(task)
        elif task_type == "validate_data":
            return await self._validate_data(task)
        else:
            return await super()._execute_domain_specific_task(task)
    
    async def _process_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process data according to specifications"""
        data = task.get("data", [])
        operations = task.get("operations", [])
        
        # Simulate data processing
        processed_count = len(data)
        result_data = data  # In a real implementation, this would actually process the data
        
        return {
            "status": "completed",
            "original_count": len(data),
            "processed_count": processed_count,
            "operations_applied": operations,
            "data_format": task.get("format", "unknown")
        }
    
    async def _analyze_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data to extract insights"""
        data = task.get("data", [])
        
        # Simulate data analysis
        analysis_result = {
            "total_records": len(data),
            "summary_statistics": {
                "mean": sum(data) / len(data) if data and all(isinstance(x, (int, float)) for x in data) else "N/A",
                "min": min(data) if data and all(isinstance(x, (int, float)) for x in data) else "N/A",
                "max": max(data) if data and all(isinstance(x, (int, float)) for x in data) else "N/A"
            },
            "data_types": list(set(type(item).__name__ for item in data)),
            "insights": [
                "Data distribution appears normal",
                "No significant outliers detected",
                "Correlation between variables: moderate"
            ]
        }
        
        return analysis_result
    
    async def _clean_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Clean data by removing or fixing inconsistencies"""
        data = task.get("data", [])
        
        # Simulate data cleaning
        original_count = len(data)
        cleaned_data = [item for item in data if item is not None]  # Remove None values
        removed_count = original_count - len(cleaned_data)
        
        return {
            "status": "completed",
            "original_count": original_count,
            "cleaned_count": len(cleaned_data),
            "removed_count": removed_count,
            "cleaning_operations": ["remove_null_values"],
            "data_quality_improvement": 15.0  # Percentage improvement
        }
    
    async def _visualize_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualizations for data"""
        data = task.get("data", [])
        chart_type = task.get("chart_type", "bar")
        
        # Simulate visualization creation
        visualization_result = {
            "status": "completed",
            "chart_type": chart_type,
            "data_points": len(data),
            "visualization_url": f"chart_{uuid.uuid4().hex[:8]}.png",
            "visualization_tool": "matplotlib",
            "insights_highlighted": ["Trend identification", "Outlier detection"]
        }
        
        return visualization_result
    
    async def _validate_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality and integrity"""
        data = task.get("data", [])
        
        # Simulate data validation
        validation_result = {
            "status": "completed",
            "total_records": len(data),
            "valid_records": len(data),  # In this simulation, all records are valid
            "invalid_records": 0,
            "quality_metrics": {
                "completeness": 0.95,
                "accuracy": 0.92,
                "consistency": 0.98,
                "timeliness": 0.90
            },
            "validation_rules_applied": ["format_check", "range_check", "consistency_check"],
            "data_quality_score": 93.75  # Average of quality metrics
        }
        
        return validation_result


class CommunicationSpecialist(SpecializedAgent):
    """Specialized agent for communication tasks"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description, domain="communication")
        self.add_capability(AgentCapability.COMMUNICATION)
        self.add_capability(AgentCapability.ANALYSIS)
        self.communication_channels = ["email", "chat", "api", "file", "database"]  # Supported channels
        self.communication_protocols = ["http", "https", "smtp", "websocket", "mqtt"]  # Protocols
        self.message_formats = ["json", "xml", "text", "binary"]  # Supported formats
        self.security_protocols = ["tls", "oauth", "basic_auth"]  # Security protocols
        self.language_models = ["gpt", "bert", "t5"]  # Available language models
        self.communication_metrics = {
            "response_time": 0.0,
            "success_rate": 0.0,
            "throughput": 0.0,
            "error_rate": 1.0
        }
    
    async def _execute_domain_specific_task(self, task: Dict[str, Any]) -> Any:
        """Execute communication-specific task"""
        task_type = task.get("type", "unknown")
        
        if task_type == "send_message":
            return await self._send_message(task)
        elif task_type == "receive_message":
            return await self._receive_message(task)
        elif task_type == "translate_message":
            return await self._translate_message(task)
        elif task_type == "format_message":
            return await self._format_message(task)
        elif task_type == "route_message":
            return await self._route_message(task)
        else:
            return await super()._execute_domain_specific_task(task)
    
    async def _send_message(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message through specified channel"""
        message = task.get("message", "")
        channel = task.get("channel", "chat")
        recipient = task.get("recipient", "unknown")
        
        # Simulate message sending
        send_result = {
            "status": "sent",
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "channel": channel,
            "recipient": recipient,
            "timestamp": datetime.now().isoformat(),
            "delivery_status": "delivered"
        }
        
        return send_result
    
    async def _receive_message(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Receive a message from specified source"""
        source = task.get("source", "chat")
        
        # Simulate message receiving
        received_message = {
            "status": "received",
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "source": source,
            "content": "Sample received message content",
            "timestamp": datetime.now().isoformat(),
            "sender": "system"
        }
        
        return received_message
    
    async def _translate_message(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Translate message from one language to another"""
        message = task.get("message", "")
        source_lang = task.get("source_language", "en")
        target_lang = task.get("target_language", "es")
        
        # Simulate translation
        translated_message = f"[TRANSLATED] {message}"  # In a real system, this would use actual translation
        
        return {
            "status": "translated",
            "original_message": message,
            "translated_message": translated_message,
            "source_language": source_lang,
            "target_language": target_lang,
            "translation_quality": 0.85
        }
    
    async def _format_message(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Format message according to specified format"""
        message = task.get("message", "")
        target_format = task.get("format", "json")
        
        # Simulate message formatting
        formatted_message = {
            "status": "formatted",
            "original_format": "text",
            "target_format": target_format,
            "formatted_content": f"{{\"content\": \"{message}\", \"format\": \"{target_format}\"}}",
            "validation_passed": True
        }
        
        return formatted_message
    
    async def _route_message(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Route message to appropriate destination"""
        message = task.get("message", "")
        routing_rules = task.get("routing_rules", [])
        
        # Simulate message routing
        routing_result = {
            "status": "routed",
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "routing_rules_applied": routing_rules,
            "destination": "appropriate_endpoint",
            "routing_time": 0.005,  # seconds
            "route_optimization_applied": True
        }
        
        return routing_result


# Factory function to create specialized agents
def create_specialized_agent(agent_type: str, name: str, description: str = "") -> SpecializedAgent:
    """Factory function to create specialized agents"""
    if agent_type == "code":
        return CodeSpecialist(name, description)
    elif agent_type == "data":
        return DataSpecialist(name, description)
    elif agent_type == "communication":
        return CommunicationSpecialist(name, description)
    else:
        raise ValueError(f"Unknown specialized agent type: {agent_type}")


# Utility functions for advanced agent operations

def create_agent_network(topology: str, agent_configs: List[Dict[str, Any]]) -> List[AdvancedBaseAgent]:
    """Create a network of interconnected agents"""
    agents = []
    
    for config in agent_configs:
        agent_type = config.get("type", "basic")
        name = config.get("name", f"Agent_{uuid.uuid4().hex[:8]}")
        description = config.get("description", "")
        
        if agent_type == "learning":
            agent = LearningAgent(name, description)
        elif agent_type == "adaptive":
            agent = AdaptiveAgent(name, description)
        elif agent_type in ["code", "data", "communication"]:
            agent = create_specialized_agent(agent_type, name, description)
        else:
            from src.agents.default_agent import DefaultAgent
            agent = DefaultAgent(name, description)
        
        # Add capabilities from config
        capabilities = config.get("capabilities", [])
        for cap_name in capabilities:
            try:
                capability = AgentCapability(cap_name)
                agent.add_capability(capability)
            except ValueError:
                pass  # Skip unknown capabilities
        
        agents.append(agent)
    
    # Connect agents based on topology
    if topology == "mesh":
        # Connect every agent to every other agent
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    if not hasattr(agent1, 'connections'):
                        agent1.connections = []
                    agent1.connections.append(agent2.id)
    elif topology == "star":
        # Connect all agents to a central agent
        if agents:
            center_agent = agents[0]
            for agent in agents[1:]:
                if not hasattr(agent, 'connections'):
                    agent.connections = []
                agent.connections.append(center_agent.id)
                if not hasattr(center_agent, 'connections'):
                    center_agent.connections = []
                center_agent.connections.append(agent.id)
    
    return agents


def evaluate_agent_performance(agent: AdvancedBaseAgent, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate the performance of an agent on a set of tasks"""
    results = []
    start_time = datetime.now()
    
    for task in tasks:
        # Record start time for this task
        task_start = datetime.now()
        
        # Execute task (simulated)
        result = {
            "task_id": task.get("id", "unknown"),
            "status": "completed",
            "execution_time": random.uniform(0.1, 2.0),  # Simulated execution time
            "success": random.choice([True, True, True, False])  # 75% success rate
        }
        results.append(result)
        
        # Update progress
        progress = (len(results) / len(tasks)) * 100
        agent.update_progress(progress, f"Processing task {len(results)}/{len(tasks)}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Calculate performance metrics
    successful_tasks = [r for r in results if r["success"]]
    success_rate = len(successful_tasks) / len(results) if results else 0
    
    avg_execution_time = sum(r["execution_time"] for r in results) / len(results) if results else 0
    
    performance_metrics = {
        "total_tasks": len(tasks),
        "successful_tasks": len(successful_tasks),
        "failed_tasks": len(results) - len(successful_tasks),
        "success_rate": success_rate,
        "total_execution_time": total_time,
        "average_task_time": avg_execution_time,
        "results": results
    }
    
    return performance_metrics


def optimize_agent_configuration(agent: AdvancedBaseAgent, 
                               performance_data: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize agent configuration based on performance data"""
    recommendations = {
        "parameter_adjustments": [],
        "capability_changes": [],
        "resource_allocations": [],
        "behavior_modifications": []
    }
    
    # Analyze performance data
    success_rate = performance_data.get("success_rate", 0)
    avg_time = performance_data.get("average_task_time", 1.0)
    
    # Make recommendations based on performance
    if success_rate < 0.7:
        recommendations["parameter_adjustments"].append({
            "parameter": "exploration_rate",
            "current_value": getattr(agent, 'exploration_rate', 0.1),
            "recommended_value": min(0.3, getattr(agent, 'exploration_rate', 0.1) * 1.5),
            "reason": "Low success rate indicates need for more exploration"
        })
    
    if avg_time > 1.0:
        recommendations["behavior_modifications"].append({
            "modification": "streamline_process",
            "reason": "High average execution time suggests optimization needed"
        })
    
    # Check adaptation level
    if hasattr(agent, 'adaptation_level'):
        if agent.adaptation_level == AdaptationLevel.BASIC and success_rate > 0.8:
            recommendations["behavior_modifications"].append({
                "modification": "upgrade_adaptation_level",
                "current_level": agent.adaptation_level.value,
                "recommended_level": AdaptationLevel.ADVANCED.value,
                "reason": "High performance suggests capability for advanced adaptation"
            })
    
    return recommendations


def backup_agent_state(agent: AdvancedBaseAgent, backup_path: str) -> bool:
    """Backup the state of an agent"""
    try:
        # Prepare agent state data
        state_data = {
            "agent_id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "capabilities": [cap.value for cap in agent.capabilities],
            "status": agent.status.value,
            "memory": {
                memory_type: [
                    {
                        "content": str(mem.content),  # Convert to string for JSON serialization
                        "timestamp": mem.timestamp.isoformat(),
                        "importance": mem.importance,
                        "access_count": mem.access_count,
                        "last_access": mem.last_access.isoformat()
                    }
                    for mem in memories
                ]
                for memory_type, memories in agent.memory.items()
            },
            "learning_history": [
                {
                    "task_id": exp.task_id,
                    "task_result": exp.task_result,
                    "reward": exp.reward,
                    "timestamp": exp.timestamp.isoformat(),
                    "context": exp.context
                }
                for exp in agent.learning_history
            ],
            "adaptation_metrics": asdict(agent.adaptation_metrics),
            "knowledge_base": agent.knowledge_base,
            "experience_points": agent.experience_points,
            "skill_level": agent.skill_level,
            "personality_traits": agent.personality_traits,
            "communication_style": agent.communication_style,
            "task_preferences": agent.task_preferences,
            "backup_timestamp": datetime.now().isoformat()
        }
        
        # Write to file
        backup_file = Path(backup_path) / f"agent_{agent.id}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        agent.logger.info(f"Agent state backed up to {backup_file}")
        return True
    except Exception as e:
        agent.logger.error(f"Failed to backup agent state: {e}")
        return False


def restore_agent_state(agent: AdvancedBaseAgent, backup_path: str) -> bool:
    """Restore the state of an agent from backup"""
    try:
        with open(backup_path, 'r') as f:
            state_data = json.load(f)
        
        # Restore basic properties
        agent.id = state_data.get("agent_id", agent.id)
        agent.name = state_data.get("name", agent.name)
        agent.description = state_data.get("description", agent.description)
        
        # Restore capabilities
        agent.capabilities = set()
        for cap_name in state_data.get("capabilities", []):
            try:
                capability = AgentCapability(cap_name)
                agent.add_capability(capability)
            except ValueError:
                pass  # Skip unknown capabilities
        
        # Restore memory
        agent.memory = defaultdict(list)
        for memory_type, memories in state_data.get("memory", {}).items():
            for mem_data in memories:
                memory_entry = AgentMemory(
                    memory_type=AgentMemoryType(memory_type),
                    content=mem_data["content"],
                    timestamp=datetime.fromisoformat(mem_data["timestamp"]),
                    importance=mem_data["importance"],
                    access_count=mem_data["access_count"],
                    last_access=datetime.fromisoformat(mem_data["last_access"])
                )
                agent.memory[memory_type].append(memory_entry)
        
        # Restore learning history
        agent.learning_history = []
        for exp_data in state_data.get("learning_history", []):
            experience = AgentLearningHistory(
                task_id=exp_data["task_id"],
                task_result=exp_data["task_result"],
                reward=exp_data["reward"],
                timestamp=datetime.fromisoformat(exp_data["timestamp"]),
                context=exp_data["context"]
            )
            agent.learning_history.append(experience)
        
        # Restore adaptation metrics
        metrics_data = state_data.get("adaptation_metrics", {})
        agent.adaptation_metrics = AgentAdaptationMetrics(
            learning_rate=metrics_data.get("learning_rate", 0.0),
            adaptation_score=metrics_data.get("adaptation_score", 0.0),
            performance_improvement=metrics_data.get("performance_improvement", 0.0),
            error_rate=metrics_data.get("error_rate", 1.0),
            efficiency_score=metrics_data.get("efficiency_score", 0.0),
            last_adaptation=datetime.fromisoformat(metrics_data.get("last_adaptation", datetime.now().isoformat()))
        )
        
        # Restore other properties
        agent.knowledge_base = state_data.get("knowledge_base", {})
        agent.experience_points = state_data.get("experience_points", 0)
        agent.skill_level = state_data.get("skill_level", 1)
        agent.personality_traits = state_data.get("personality_traits", {})
        agent.communication_style = state_data.get("communication_style", "formal")
        agent.task_preferences = state_data.get("task_preferences", [])
        
        agent.logger.info(f"Agent state restored from {backup_path}")
        return True
    except Exception as e:
        agent.logger.error(f"Failed to restore agent state: {e}")
        return False


def analyze_agent_collaboration(agents: List[AdvancedBaseAgent], 
                              collaboration_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze collaboration patterns between agents"""
    analysis = {
        "total_interactions": len(collaboration_data),
        "agent_interaction_counts": {},
        "collaboration_efficiency": {},
        "communication_patterns": {},
        "suggested_improvements": []
    }
    
    # Count interactions per agent
    for interaction in collaboration_data:
        sender = interaction.get("sender", "unknown")
        receiver = interaction.get("receiver", "unknown")
        
        analysis["agent_interaction_counts"][sender] = analysis["agent_interaction_counts"].get(sender, 0) + 1
        analysis["agent_interaction_counts"][receiver] = analysis["agent_interaction_counts"].get(receiver, 0) + 1
    
    # Analyze collaboration efficiency (simplified)
    for agent in agents:
        interaction_count = analysis["agent_interaction_counts"].get(agent.id, 0)
        # In a real system, this would use actual performance data
        efficiency = min(1.0, interaction_count * 0.1)  # Simple efficiency calculation
        analysis["collaboration_efficiency"][agent.id] = efficiency
    
    # Identify communication patterns
    pattern_counts = defaultdict(int)
    for interaction in collaboration_data:
        pattern = f"{interaction.get('sender', 'unknown')}->{interaction.get('receiver', 'unknown')}"
        pattern_counts[pattern] += 1
    
    analysis["communication_patterns"] = dict(pattern_counts)
    
    # Suggest improvements
    if len(agents) > 1:
        least_interacted = min(analysis["agent_interaction_counts"].items(), key=lambda x: x[1])
        analysis["suggested_improvements"].append({
            "type": "increase_collaboration",
            "agent_id": least_interacted[0],
            "current_interactions": least_interacted[1],
            "suggestion": "Increase collaboration with other agents"
        })
    
    return analysis


# Example usage and testing functions

def example_usage():
    """Example of how to use the advanced agents"""
    print("Creating advanced agents...")
    
    # Create a learning agent
    learning_agent = LearningAgent("LearningBot", "Agent that learns from tasks")
    print(f"Created learning agent: {learning_agent.name}")
    
    # Create an adaptive agent
    adaptive_agent = AdaptiveAgent("AdaptiBot", "Agent that adapts to contexts")
    print(f"Created adaptive agent: {adaptive_agent.name}")
    
    # Create specialized agents
    code_agent = CodeSpecialist("CodeMaster", "Specialized in code tasks")
    data_agent = DataSpecialist("DataWizard", "Specialized in data tasks")
    comm_agent = CommunicationSpecialist("CommPro", "Specialized in communication tasks")
    
    print(f"Created specialized agents: {code_agent.name}, {data_agent.name}, {comm_agent.name}")
    
    # Create a simple task
    task = {
        "id": "task_1",
        "title": "Sample Learning Task",
        "type": "analyze_data",
        "required_capabilities": ["analysis", "data_processing"],
        "priority": 2,
        "data": [1, 2, 3, 4, 5]
    }
    
    # Execute task with learning agent (simulated)
    print(f"Learning agent would execute task: {task['title']}")
    
    # Evaluate agent performance (simulated)
    tasks = [{"id": f"task_{i}", "type": "simple_task"} for i in range(5)]
    performance = evaluate_agent_performance(learning_agent, tasks)
    print(f"Agent performance: {performance['success_rate']:.2%} success rate")
    
    # Get optimization recommendations
    recommendations = optimize_agent_configuration(learning_agent, performance)
    print(f"Configuration recommendations: {len(recommendations['parameter_adjustments'])} adjustments suggested")


if __name__ == "__main__":
    example_usage()