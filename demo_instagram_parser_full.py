#!/usr/bin/env python3
"""
Full demonstration of building an Instagram parser using the Agentic Architecture System
This demonstrates the complete iterative development process with all three iterations.
"""

import asyncio
import json
import os
from src.main import AgenticSystem
from src.agents.base_agent import AgentCapability
from src.utils.logger import setup_logger

async def build_instagram_parser():
    """
    Full demonstration of building an Instagram parser through 3 iterations
    """
    logger = setup_logger("InstagramParserDemo")
    logger.info("Starting full Instagram Parser development demonstration")
    
    print("\n" + "="*70)
    print("FULL INSTAGRAM PARSER DEVELOPMENT DEMONSTRATION")
    print("Using Agentic Architecture System with 3 Iterations")
    print("="*70)
    
    # Create the main system
    system = AgenticSystem()
    
    # Create specialized agents for each iteration
    print("\n1. CREATING SPECIALIZED AGENTS FOR ITERATIVE DEVELOPMENT")
    print("-" * 55)
    
    # Iteration 1: Basic functionality
    iter1_dev = system.build_agent({
        "type": "code_writer",
        "name": "BasicParserDeveloper",
        "description": "Specialized in basic Instagram parser functionality"
    })
    
    # Iteration 2: Enhancement with containerization
    iter2_dev = system.build_agent({
        "type": "code_writer", 
        "name": "DockerIntegrationDeveloper",
        "description": "Specialized in Docker containerization"
    })
    
    # Iteration 3: Multi-component architecture
    iter3_dev = system.build_agent({
        "type": "code_writer",
        "name": "MultiComponentDeveloper",
        "description": "Specialized in multi-component system architecture"
    })
    
    # Testing agent for all iterations
    tester = system.build_agent({
        "type": "tester",
        "name": "IntegrationTester", 
        "description": "Specialized in testing all iterations"
    })
    
    # Task manager for coordination
    task_manager = system.build_agent({
        "type": "task_manager",
        "name": "ProjectCoordinator",
        "description": "Manages the entire development process"
    })
    
    # Register all agents with communication system
    for agent_id in system.agents.keys():
        await system.a2a_manager.register_agent(agent_id)
    
    print(f"Created {len(system.agents)} specialized agents:")
    for agent in system.agents.values():
        print(f"  - {agent.name} ({agent.agent_type}): {agent.description}")
    
    # Main task description
    main_task = "Write an Instagram parser that can extract public posts with authentication"
    
    print(f"\n2. DECOMPOSING MAIN TASK: {main_task}")
    print("-" * 55)
    
    # Decompose the task into 3 iterations
    decomposition_plan = system.decompose_task(main_task, iterations=3)
    
    print(f"Task type identified: {decomposition_plan['task_type']}")
    print(f"Total iterations: {len(decomposition_plan['iterations'])}")
    print(f"Total subtasks: {decomposition_plan['total_subtasks']}")
    
    # Create a comprehensive project plan
    print(f"\n3. ITERATIVE DEVELOPMENT PLAN")
    print("-" * 55)
    
    # Execute each iteration
    iteration_results = []
    
    for iteration in decomposition_plan['iterations']:
        print(f"\nITERATION {iteration['iteration_number']}: {iteration['description']}")
        print(f"  Focus: {iteration['focus']}")
        print(f"  Subtasks: {len(iteration['subtasks'])}")
        
        # Select the appropriate developer agent for this iteration
        if iteration['iteration_number'] == 1:
            dev_agent = iter1_dev
        elif iteration['iteration_number'] == 2:
            dev_agent = iter2_dev
        else:
            dev_agent = iter3_dev
            
        # Execute each subtask in the iteration
        iteration_subtasks_results = []
        
        for i, subtask in enumerate(iteration['subtasks'], 1):
            print(f"    {i}. {subtask['title']}")
            print(f"        {subtask['description']}")
            
            # Create task object
            task = {
                "id": f"iter{iteration['iteration_number']}_task{i}",
                "title": subtask['title'],
                "description": subtask['description'],
                "agent_type": dev_agent.agent_type,
                "iteration": iteration['iteration_number'],
                "original_task": main_task
            }
            
            # Execute the task with the appropriate agent
            result = await dev_agent.execute_task(task)
            iteration_subtasks_results.append({
                "task": subtask['title'],
                "result": result
            })
            
            print(f"        ✓ Completed: {result['result'][:60]}...")
        
        iteration_results.append({
            "iteration": iteration['iteration_number'],
            "description": iteration['description'],
            "results": iteration_subtasks_results
        })
        
        # Test the iteration results
        test_task = {
            "id": f"iter{iteration['iteration_number']}_test",
            "title": f"Test Iteration {iteration['iteration_number']} Results",
            "description": f"Testing the results from iteration {iteration['iteration_number']}",
            "agent_type": "tester"
        }
        
        test_result = await tester.execute_task(test_task)
        print(f"    Test result: {test_result['result'][:60]}...")
    
    # Generate task board for the entire project
    print(f"\n4. PROJECT TASK BOARD")
    print("-" * 55)
    
    task_board = system.task_decomposer.create_task_board(decomposition_plan)
    print(f"Generated task board with {len(task_board)} individual tasks")
    
    # Show task distribution by iteration
    iteration_tasks = {}
    for task in task_board:
        iteration_num = task.params.get('iteration', 1)
        if iteration_num not in iteration_tasks:
            iteration_tasks[iteration_num] = []
        iteration_tasks[iteration_num].append(task)
    
    for iter_num in sorted(iteration_tasks.keys()):
        print(f"  Iteration {iter_num}: {len(iteration_tasks[iter_num])} tasks")
        for i, task in enumerate(iteration_tasks[iter_num][:3]):  # Show first 3 tasks
            print(f"    {i+1}. {task.title}")
        if len(iteration_tasks[iter_num]) > 3:
            print(f"    ... and {len(iteration_tasks[iter_num]) - 3} more tasks")
    
    # Demonstrate agent-to-agent communication
    print(f"\n5. AGENT COMMUNICATION DEMONSTRATION")
    print("-" * 55)
    
    from src.communication.a2a_manager import Message, MessageType
    
    # Coordinator sends status update to tester
    status_message = Message(
        sender_id=task_manager.id,
        receiver_id=tester.id,
        msg_type=MessageType.NOTIFICATION,
        content={
            "notification_type": "iteration_complete",
            "iteration": 2,
            "status": "completed",
            "next_steps": ["proceed to iteration 3", "prepare integration tests"]
        }
    )
    
    success = await system.a2a_manager.send_message(status_message)
    print(f"  Coordinator → Tester: Iteration 2 complete notification: {'✓' if success else '✗'}")
    
    # Tester acknowledges and responds
    ack_message = Message(
        sender_id=tester.id,
        receiver_id=task_manager.id,
        msg_type=MessageType.RESPONSE,
        content={
            "response_type": "acknowledgment",
            "acknowledged_iteration": 2,
            "ready_for": "iteration_3_testing",
            "status": "ready"
        }
    )
    
    success = await system.a2a_manager.send_message(ack_message)
    print(f"  Tester → Coordinator: Acknowledgment: {'✓' if success else '✗'}")
    
    # Show communication statistics
    comm_stats = system.a2a_manager.get_communication_stats()
    print(f"  Communication system stats: {comm_stats}")
    
    # Final project summary
    print(f"\n6. PROJECT SUMMARY")
    print("-" * 55)
    
    print(f"✓ Main Task: {main_task}")
    print(f"✓ Iterations completed: {len(iteration_results)}")
    print(f"✓ Total subtasks executed: {sum(len(iter['results']) for iter in iteration_results)}")
    print(f"✓ Agents utilized: {len(system.agents)}")
    print(f"✓ Communication events: {comm_stats['total_messages_processed']}")
    
    print(f"\nITERATION BREAKDOWN:")
    for iteration in iteration_results:
        print(f"  Iteration {iteration['iteration']} ({iteration['description']}):")
        print(f"    - {len(iteration['results'])} subtasks completed")
        print(f"    - Agent: {iter1_dev.name if iteration['iteration'] == 1 else iter2_dev.name if iteration['iteration'] == 2 else iter3_dev.name}")
    
    # Show the complete task decomposition
    print(f"\nCOMPLETE TASK DECOMPOSITION TREE:")
    for iteration in decomposition_plan['iterations']:
        print(f"  └── Iteration {iteration['iteration_number']}: {iteration['description']}")
        for i, subtask in enumerate(iteration['subtasks']):
            indent = "      ├──" if i < len(iteration['subtasks']) - 1 else "      └──"
            print(f"{indent} {subtask['title']}")
    
    print(f"\n" + "="*70)
    print("INSTAGRAM PARSER DEVELOPMENT COMPLETE")
    print("All three iterations successfully executed with agent collaboration")
    print("="*70)
    
    return {
        "system": system,
        "decomposition_plan": decomposition_plan,
        "iteration_results": iteration_results,
        "task_board": task_board
    }

async def demonstrate_recursive_decomposition():
    """
    Demonstrate how each subtask can be recursively decomposed
    """
    print(f"\n7. RECURSIVE DECOMPOSITION DEMONSTRATION")
    print("-" * 55)
    
    system = AgenticSystem()
    
    # Take a complex subtask and show how it could be further decomposed
    complex_subtask = "Implement basic parsing logic"
    
    print(f"Original subtask: '{complex_subtask}'")
    print(f"This could be recursively decomposed into:")
    
    # Show recursive decomposition
    sub_decomposition = [
        "Parse Instagram post structure",
        "Extract media URLs from posts", 
        "Extract captions and hashtags",
        "Handle pagination for multiple posts",
        "Implement rate limiting",
        "Add error handling for failed requests"
    ]
    
    for i, sub_subtask in enumerate(sub_decomposition, 1):
        print(f"  {i}. {sub_subtask}")
    
    print(f"\nThis demonstrates the recursive nature of the task decomposition system.")
    print(f"Each task can be broken down further until reaching implementation-level steps.")

async def main():
    """
    Main function to run the complete demonstration
    """
    print("AGENTIC ARCHITECTURE SYSTEM")
    print("Full Instagram Parser Development Demonstration")
    
    # Run the main demonstration
    results = await build_instagram_parser()
    
    # Run recursive decomposition demo
    await demonstrate_recursive_decomposition()
    
    print(f"\nDEMONSTRATION COMPLETE")
    print("The Agentic Architecture System has successfully demonstrated:")
    print("- Complete iterative development (3 iterations)")
    print("- Task decomposition and subtask management")
    print("- Agent specialization and assignment")
    print("- Agent-to-agent communication")
    print("- Task board generation and project tracking")
    print("- Recursive task decomposition capabilities")
    print("- Comprehensive project management")

if __name__ == "__main__":
    asyncio.run(main())