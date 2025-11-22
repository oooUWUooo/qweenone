#!/usr/bin/env python3

from src.agents.base_agent import BaseAgent
from src.task_manager.task_decomposer import TaskDecomposer
from src.communication.a2a_manager import A2ACommunicationManager
from src.testing.test_runner import TestRunner
from src.builders.agent_builder import AgentBuilder
from src.utils.logger import setup_logger

import asyncio
import argparse
from typing import Dict, Any

class AgenticSystem:
    
    def __init__(self):
        self.logger = setup_logger("AgenticSystem")
        self.task_decomposer = TaskDecomposer()
        self.a2a_manager = A2ACommunicationManager()
        self.test_runner = TestRunner()
        self.agent_builder = AgentBuilder()
        self.agents = {}
        
    def build_agent(self, agent_config: Dict[str, Any]) -> BaseAgent:
        agent = self.agent_builder.create_agent(agent_config)
        self.agents[agent.id] = agent
        return agent
    
    def decompose_task(self, task_description: str, iterations: int = 3) -> Dict[str, Any]:
        return self.task_decomposer.decompose(task_description, iterations)
    
    def run_tests(self, code_path: str) -> Dict[str, Any]:
        return self.test_runner.run_tests(code_path)
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        # Find appropriate agent for the task
        agent_type = task.get('agent_type', 'default')
        agent = self.get_agent_by_type(agent_type)
        
        if not agent:
            self.logger.info(f"No specific agent found for type {agent_type}, using default agent")
            agent = self.agents.get('default')
        
        if not agent:
            self.logger.error("No agents available to execute task")
            return None
            
        return await agent.execute_task(task)
    
    def get_agent_by_type(self, agent_type: str) -> BaseAgent:
        for agent in self.agents.values():
            if hasattr(agent, 'agent_type') and agent.agent_type == agent_type:
                return agent
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task to decompose and execute")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for task decomposition")
    
    args = parser.parse_args()
    
    system = AgenticSystem()
    
    if args.task:
        task_plan = system.decompose_task(args.task, args.iterations)
        print(f"Decomposed task plan: {task_plan}")
        
        print("Task execution would happen here in a full implementation")

if __name__ == "__main__":
    main()