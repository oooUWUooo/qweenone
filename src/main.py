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
import json
from datetime import datetime


class AgenticSystem:
    
    def __init__(self):
        self.logger = setup_logger("AgenticSystem")
        self.task_decomposer = TaskDecomposer()
        self.a2a_manager = A2ACommunicationManager()
        self.test_runner = TestRunner()
        self.agent_builder = AgentBuilder()
        self.agents = {}
        self.execution_history = []
        self.task_queue = asyncio.Queue()
        
    def build_agent(self, agent_config: Dict[str, Any]) -> BaseAgent:
        agent = self.agent_builder.create_agent(agent_config)
        self.agents[agent.id] = agent
        
        # Register agent with communication manager
        asyncio.create_task(self.a2a_manager.register_agent(agent.id, agent.communicate_with_agent))
        
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
            agent = self.get_default_agent()
        
        if not agent:
            self.logger.error("No agents available to execute task")
            return None
            
        # Track execution start time
        start_time = datetime.now()
        
        try:
            result = await agent.execute_task(task)
            
            # Log execution in history
            execution_record = {
                "task_id": task.get("id"),
                "agent_id": agent.id,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration": (datetime.now() - start_time).total_seconds(),
                "status": "completed",
                "result": result
            }
            
            self.execution_history.append(execution_record)
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing task: {e}")
            
            # Log execution failure
            execution_record = {
                "task_id": task.get("id"),
                "agent_id": agent.id,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration": (datetime.now() - start_time).total_seconds(),
                "status": "failed",
                "error": str(e)
            }
            
            self.execution_history.append(execution_record)
            raise e
    
    def get_agent_by_type(self, agent_type: str) -> BaseAgent:
        for agent in self.agents.values():
            if hasattr(agent, 'agent_type') and agent.agent_type == agent_type:
                return agent
        return None

    def get_default_agent(self) -> BaseAgent:
        """Get or create a default agent if one doesn't exist"""
        for agent in self.agents.values():
            if hasattr(agent, 'agent_type') and agent.agent_type == 'default':
                return agent
        
        # Create a default agent if none exists
        default_config = {
            "type": "default",
            "name": "DefaultAgent",
            "description": "Default agent for general tasks"
        }
        default_agent = self.build_agent(default_config)
        return default_agent

    async def execute_task_plan(self, task_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a full task plan with multiple iterations and subtasks"""
        results = {
            "original_task": task_plan["original_task"],
            "iterations": [],
            "overall_status": "completed"
        }
        
        for iteration in task_plan["iterations"]:
            iteration_results = {
                "iteration_number": iteration["iteration_number"],
                "subtask_results": []
            }
            
            for subtask in iteration["subtasks"]:
                # Execute each subtask
                result = await self.execute_task(subtask)
                iteration_results["subtask_results"].append({
                    "subtask": subtask,
                    "result": result
                })
                
            results["iterations"].append(iteration_results)
        
        return results

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "active_agents": len(self.agents),
            "agent_types": list(set(agent.agent_type for agent in self.agents.values())),
            "execution_history_size": len(self.execution_history),
            "queued_tasks": self.task_queue.qsize(),
            "a2a_stats": self.a2a_manager.get_communication_stats()
        }

    async def build_agents_from_plan(self, task_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Build specialized agents based on task requirements"""
        agent_configs = []
        
        # Analyze task plan to determine required agent types
        for iteration in task_plan["iterations"]:
            for subtask in iteration["subtasks"]:
                # Determine required agent type based on subtask
                if "code" in subtask["title"].lower() or "implement" in subtask["title"].lower():
                    agent_configs.append({
                        "type": "code_writer",
                        "name": f"CodeAgent_{len(self.agents)}",
                        "description": f"Code writing agent for: {subtask['title']}"
                    })
                elif "test" in subtask["title"].lower() or "validate" in subtask["title"].lower():
                    agent_configs.append({
                        "type": "tester", 
                        "name": f"TesterAgent_{len(self.agents)}",
                        "description": f"Testing agent for: {subtask['title']}"
                    })
                elif "communication" in subtask["title"].lower() or "connect" in subtask["title"].lower():
                    agent_configs.append({
                        "type": "communication",
                        "name": f"CommAgent_{len(self.agents)}",
                        "description": f"Communication agent for: {subtask['title']}"
                    })
                else:
                    agent_configs.append({
                        "type": "task_manager",
                        "name": f"TaskAgent_{len(self.agents)}",
                        "description": f"Task management agent for: {subtask['title']}"
                    })
        
        # Build the agents
        built_agents = []
        for config in agent_configs:
            agent = self.build_agent(config)
            built_agents.append(agent.get_agent_info())
        
        return {
            "status": "agents_built",
            "count": len(built_agents),
            "agents": built_agents
        }

    async def run_full_pipeline(self, task_description: str, iterations: int = 3) -> Dict[str, Any]:
        """Run the complete agentic pipeline: decompose -> build agents -> execute -> test"""
        self.logger.info(f"Starting full pipeline for task: {task_description}")
        
        # 1. Decompose task
        task_plan = self.decompose_task(task_description, iterations)
        self.logger.info(f"Task decomposed into {task_plan['total_subtasks']} subtasks")
        
        # 2. Build specialized agents
        agent_build_result = await self.build_agents_from_plan(task_plan)
        self.logger.info(f"Built {agent_build_result['count']} specialized agents")
        
        # 3. Execute task plan
        execution_result = await self.execute_task_plan(task_plan)
        self.logger.info("Task plan execution completed")
        
        # 4. Run tests on results (if applicable)
        test_result = self.run_tests(".")  # Run tests on current project
        
        return {
            "task_decomposition": task_plan,
            "agent_building": agent_build_result,
            "execution": execution_result,
            "testing": test_result,
            "system_status": self.get_system_status()
        }


def main():
    parser = argparse.ArgumentParser(description="Agentic Architecture System")
    parser.add_argument("--task", type=str, help="Task to decompose and execute")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for task decomposition")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--status", action="store_true", help="Show system status")
    
    args = parser.parse_args()
    
    system = AgenticSystem()
    
    if args.status:
        # Show system status
        status = system.get_system_status()
        print(json.dumps(status, indent=2))
        
    elif args.interactive:
        # Interactive mode
        print("Agentic System - Interactive Mode")
        print("Commands: 'run <task>', 'status', 'quit'")
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'status':
                    status = system.get_system_status()
                    print(json.dumps(status, indent=2))
                elif user_input.startswith('run '):
                    task = user_input[4:]  # Remove 'run ' prefix
                    print(f"Running task: {task}")
                    
                    # Run the full pipeline
                    result = asyncio.run(system.run_full_pipeline(task, args.iterations))
                    print("Pipeline completed. Results:")
                    print(json.dumps(result, indent=2))
                else:
                    print("Unknown command. Use 'run <task>', 'status', or 'quit'")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.task:
        # Run the full pipeline for the specified task
        result = asyncio.run(system.run_full_pipeline(args.task, args.iterations))
        print("Pipeline completed. Results:")
        print(json.dumps(result, indent=2))
    
    else:
        print("Use --task to specify a task, --interactive for interactive mode, or --status for system status")


if __name__ == "__main__":
    main()