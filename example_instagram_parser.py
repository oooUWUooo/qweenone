#!/usr/bin/env python3
"""
Example implementation of the Instagram Parser using the Agentic Architecture System
This demonstrates the iterative development process with 3 iterations as requested.
"""

import asyncio
import json
from src.main import AgenticSystem
from src.agents.base_agent import AgentCapability
from src.agents.base_agent import Task
from src.utils.logger import setup_logger

async def create_instagram_parser_system():
    """
    Create an Instagram parser using the agentic system with 3 iterations
    """
    logger = setup_logger("InstagramParserExample")
    logger.info("Starting Instagram Parser creation with Agentic Architecture System")
    
    # Create the main system
    system = AgenticSystem()
    
    # Define the main task
    main_task = "Write an Instagram parser that can extract public posts"
    
    # Decompose the task into 3 iterations
    logger.info(f"Decomposing task: {main_task}")
    decomposition_plan = system.decompose_task(main_task, iterations=3)
    
    print("\n=== Instagram Parser Development Plan ===")
    print(f"Original task: {main_task}")
    print(f"Task type: {decomposition_plan['task_type']}")
    print(f"Total iterations: {len(decomposition_plan['iterations'])}")
    print(f"Total subtasks: {decomposition_plan['total_subtasks']}")
    
    # Create agents for each iteration
    agents = {}
    
    # Create specialized agents for different tasks
    agents['code_writer'] = system.build_agent({
        "type": "code_writer",
        "name": "InstagramParserDeveloper",
        "description": "Specialized in developing Instagram parsing functionality"
    })
    
    agents['tester'] = system.build_agent({
        "type": "tester",
        "name": "InstagramParserTester", 
        "description": "Specialized in testing Instagram parser functionality"
    })
    
    agents['task_manager'] = system.build_agent({
        "type": "task_manager",
        "name": "IterationManager",
        "description": "Manages iterative development process"
    })
    
    # Register agents with communication system
    for agent_id in system.agents.keys():
        await system.a2a_manager.register_agent(agent_id)
    
    print(f"\nCreated {len(agents)} specialized agents for the project")
    
    # Execute each iteration
    for iteration in decomposition_plan['iterations']:
        print(f"\n--- Executing {iteration['description']} (Focus: {iteration['focus']}) ---")
        
        # Execute each subtask in the iteration
        for i, subtask in enumerate(iteration['subtasks'], 1):
            print(f"  Executing subtask {i}: {subtask['title']}")
            
            # Determine which agent is best suited for this subtask
            if any(keyword in subtask['title'].lower() for keyword in ['test', 'verify', 'validate']):
                agent_type = 'tester'
            elif any(keyword in subtask['title'].lower() for keyword in ['implement', 'write', 'create', 'design']):
                agent_type = 'code_writer'
            else:
                agent_type = 'task_manager'
            
            # Create task object
            task = {
                "id": f"iter{iteration['iteration_number']}_task{i}",
                "title": subtask['title'],
                "description": subtask['description'],
                "agent_type": agent_type,
                "iteration": iteration['iteration_number'],
                "original_task": main_task
            }
            
            # Execute the task
            result = await system.execute_task(task)
            print(f"    Result: {result['result']}")
    
    print(f"\n=== Instagram Parser Development Complete ===")
    
    # Show communication stats
    print(f"Communication stats: {system.a2a_manager.get_communication_stats()}")
    
    # Create task board from the decomposition plan
    task_board = system.task_decomposer.create_task_board(decomposition_plan)
    print(f"Generated {len(task_board)} tasks for the task board")
    
    # Show first few tasks in the task board
    print("\nFirst few tasks in the task board:")
    for i, task in enumerate(task_board[:6]):  # Show first 6 tasks
        print(f"  {i+1}. {task.title}")
    
    return system, decomposition_plan, task_board

async def create_server_system():
    """
    Create a server using the agentic system with 3 iterations
    """
    logger = setup_logger("ServerExample")
    logger.info("Starting Server creation with Agentic Architecture System")
    
    # Create the main system
    system = AgenticSystem()
    
    # Define the main task
    main_task = "Write a server that can handle HTTP requests"
    
    # Decompose the task into 3 iterations
    logger.info(f"Decomposing task: {main_task}")
    decomposition_plan = system.decompose_task(main_task, iterations=3)
    
    print("\n=== Server Development Plan ===")
    print(f"Original task: {main_task}")
    print(f"Task type: {decomposition_plan['task_type']}")
    print(f"Total iterations: {len(decomposition_plan['iterations'])}")
    print(f"Total subtasks: {decomposition_plan['total_subtasks']}")
    
    # Create agents for each iteration
    agents = {}
    
    # Create specialized agents for different tasks
    agents['code_writer'] = system.build_agent({
        "type": "code_writer",
        "name": "ServerDeveloper",
        "description": "Specialized in developing server functionality"
    })
    
    agents['tester'] = system.build_agent({
        "type": "tester",
        "name": "ServerTester", 
        "description": "Specialized in testing server functionality"
    })
    
    agents['task_manager'] = system.build_agent({
        "type": "task_manager",
        "name": "ServerIterationManager",
        "description": "Manages server iterative development process"
    })
    
    # Register agents with communication system
    for agent_id in system.agents.keys():
        await system.a2a_manager.register_agent(agent_id)
    
    print(f"\nCreated {len(agents)} specialized agents for the server project")
    
    # Execute each iteration
    for iteration in decomposition_plan['iterations']:
        print(f"\n--- Executing {iteration['description']} (Focus: {iteration['focus']}) ---")
        
        # Execute each subtask in the iteration
        for i, subtask in enumerate(iteration['subtasks'], 1):
            print(f"  Executing subtask {i}: {subtask['title']}")
            
            # Determine which agent is best suited for this subtask
            if any(keyword in subtask['title'].lower() for keyword in ['test', 'verify', 'validate']):
                agent_type = 'tester'
            elif any(keyword in subtask['title'].lower() for keyword in ['implement', 'write', 'create', 'design']):
                agent_type = 'code_writer'
            else:
                agent_type = 'task_manager'
            
            # Create task object
            task = {
                "id": f"server_iter{iteration['iteration_number']}_task{i}",
                "title": subtask['title'],
                "description": subtask['description'],
                "agent_type": agent_type,
                "iteration": iteration['iteration_number'],
                "original_task": main_task
            }
            
            # Execute the task
            result = await system.execute_task(task)
            print(f"    Result: {result['result']}")
    
    print(f"\n=== Server Development Complete ===")
    
    # Show communication stats
    print(f"Communication stats: {system.a2a_manager.get_communication_stats()}")
    
    # Create task board from the decomposition plan
    task_board = system.task_decomposer.create_task_board(decomposition_plan)
    print(f"Generated {len(task_board)} tasks for the task board")
    
    return system, decomposition_plan, task_board

async def demonstrate_agent_communication():
    """
    Demonstrate agent-to-agent communication capabilities
    """
    logger = setup_logger("A2ACommunicationDemo")
    logger.info("Demonstrating Agent-to-Agent Communication")
    
    # Create system
    system = AgenticSystem()
    
    # Create two agents for communication
    agent1 = system.build_agent({
        "type": "communication",
        "name": "AgentAlpha",
        "description": "First communication agent"
    })
    
    agent2 = system.build_agent({
        "type": "communication", 
        "name": "AgentBeta",
        "description": "Second communication agent"
    })
    
    # Register agents with communication manager
    await system.a2a_manager.register_agent(agent1.id)
    await system.a2a_manager.register_agent(agent2.id)
    
    print("\n=== Agent-to-Agent Communication Demo ===")
    print(f"Registered agents: {system.a2a_manager.get_active_agents()}")
    
    # Send a message from agent1 to agent2
    from src.communication.a2a_manager import Message, MessageType
    
    message = Message(
        sender_id=agent1.id,
        receiver_id=agent2.id,
        msg_type=MessageType.REQUEST,
        content={
            "request_type": "data_query",
            "query": "Can you help with parsing Instagram data?",
            "context": "Instagram parser development project"
        }
    )
    
    success = await system.a2a_manager.send_message(message)
    print(f"Message sent from {agent1.name} to {agent2.name}: {success}")
    
    # Agent2 responds to agent1
    response = Message(
        sender_id=agent2.id,
        receiver_id=agent1.id,
        msg_type=MessageType.RESPONSE,
        content={
            "response_type": "data_response",
            "answer": "Yes, I can help with Instagram parsing. I recommend using the public API.",
            "suggestions": ["Use Instagram Basic Display API", "Consider rate limiting", "Handle authentication properly"]
        }
    )
    
    success = await system.a2a_manager.send_message(response)
    print(f"Response sent from {agent2.name} to {agent1.name}: {success}")
    
    # Get messages for agent1
    messages = await system.a2a_manager.get_messages_for_agent(agent1.id)
    print(f"Agent1 received {len(messages)} messages")
    
    # Show communication statistics
    stats = system.a2a_manager.get_communication_stats()
    print(f"Final communication stats: {stats}")
    
    return system

async def main():
    """
    Main function to run all examples
    """
    print("Agentic Architecture System - Full Demonstration")
    print("=" * 60)
    
    # Run Instagram parser example
    print("\n1. Instagram Parser Development Example")
    print("-" * 40)
    await create_instagram_parser_system()
    
    # Run server example
    print("\n\n2. Server Development Example") 
    print("-" * 40)
    await create_server_system()
    
    # Run communication example
    print("\n\n3. Agent-to-Agent Communication Example")
    print("-" * 40)
    await demonstrate_agent_communication()
    
    print("\n\n=== All Demonstrations Complete ===")
    print("The Agentic Architecture System successfully demonstrated:")
    print("- Iterative task decomposition (3 iterations)")
    print("- Agent creation and specialization")
    print("- Task execution with appropriate agents")
    print("- Agent-to-agent communication")
    print("- Task board generation")
    print("- Comprehensive project management")

if __name__ == "__main__":
    asyncio.run(main())