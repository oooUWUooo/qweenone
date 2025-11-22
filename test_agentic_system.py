#!/usr/bin/env python3
"""
Test script to demonstrate the Agentic Architecture System
"""

from src.main import AgenticSystem
from src.agents.base_agent import AgentCapability
import asyncio
import json

async def test_basic_functionality():
    """
    Test the basic functionality of the agentic system
    """
    print("=== Testing Agentic Architecture System ===\n")
    
    # Create the main system
    system = AgenticSystem()
    
    # Test 1: Task decomposition
    print("1. Testing Task Decomposition:")
    task_description = "Write a server that can handle HTTP requests"
    decomposition_plan = system.decompose_task(task_description, iterations=3)
    
    print(f"Original task: {task_description}")
    print(f"Task type: {decomposition_plan['task_type']}")
    print(f"Total subtasks: {decomposition_plan['total_subtasks']}")
    
    print("\nIterations:")
    for iteration in decomposition_plan['iterations']:
        print(f"  {iteration['iteration_number']}. {iteration['description']} (Focus: {iteration['focus']})")
        print(f"     Subtasks: {len(iteration['subtasks'])}")
        for i, subtask in enumerate(iteration['subtasks'], 1):
            print(f"       {i}. {subtask['title']}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Agent creation
    print("2. Testing Agent Creation:")
    
    # Create a code writing agent
    code_agent_config = {
        "type": "code_writer",
        "name": "ServerDeveloper",
        "description": "Agent specialized in writing server code"
    }
    code_agent = system.build_agent(code_agent_config)
    print(f"Created agent: {code_agent}")
    print(f"Agent capabilities: {[cap.value for cap in code_agent.capabilities]}")
    
    # Create a testing agent
    test_agent_config = {
        "type": "tester", 
        "name": "ServerTester",
        "description": "Agent specialized in testing server code"
    }
    test_agent = system.build_agent(test_agent_config)
    print(f"Created agent: {test_agent}")
    print(f"Agent capabilities: {[cap.value for cap in test_agent.capabilities]}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Task execution simulation
    print("3. Testing Task Execution:")
    
    # Create a simple task
    simple_task = {
        "id": "task_1",
        "title": "Write basic server",
        "description": "Write a basic HTTP server",
        "agent_type": "code_writer"
    }
    
    # Execute the task
    result = await system.execute_task(simple_task)
    print(f"Task execution result: {result}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 4: Instagram parser task decomposition (as per your example)
    print("4. Testing Instagram Parser Task Decomposition:")
    instagram_task = "Write an Instagram parser"
    instagram_decomposition = system.decompose_task(instagram_task, iterations=3)
    
    print(f"Original task: {instagram_task}")
    print(f"Task type: {instagram_decomposition['task_type']}")
    
    print("\nIterations:")
    for iteration in instagram_decomposition['iterations']:
        print(f"  {iteration['iteration_number']}. {iteration['description']} (Focus: {iteration['focus']})")
        print(f"     Subtasks: {len(iteration['subtasks'])}")
        for i, subtask in enumerate(iteration['subtasks'], 1):
            print(f"       {i}. {subtask['title']} - {subtask['description']}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 5: Communication system
    print("5. Testing Communication System:")
    print(f"Active agents in system: {list(system.agents.keys())}")
    
    # Register agents with communication manager
    await system.a2a_manager.register_agent(code_agent.id)
    await system.a2a_manager.register_agent(test_agent.id)
    
    print(f"Active agents after registration: {system.a2a_manager.get_active_agents()}")
    print(f"Communication stats: {system.a2a_manager.get_communication_stats()}")
    
    print("\n=== Test Complete ===")

def main():
    """
    Main function to run the tests
    """
    asyncio.run(test_basic_functionality())

if __name__ == "__main__":
    main()