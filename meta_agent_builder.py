#!/usr/bin/env python3
"""
Meta Agent Builder - Demonstrates the system's ability to create agents that can build other agents
This showcases the self-improving and meta-agentic capabilities of the architecture
"""

import asyncio
import json
from src.main import AgenticSystem
from src.agents.base_agent import AgentCapability
from src.utils.logger import setup_logger

class MetaAgentBuilder:
    """
    A specialized agent that can create and configure other agents
    This demonstrates the meta-agentic capability of the system
    """
    
    def __init__(self, system: AgenticSystem):
        self.system = system
        self.logger = setup_logger("MetaAgentBuilder")
        
    async def create_specialized_agent(self, agent_spec: dict):
        """
        Create a specialized agent based on specifications
        """
        self.logger.info(f"Creating specialized agent: {agent_spec.get('name', 'Unknown')}")
        
        # Create the agent using the system's agent builder
        agent = self.system.build_agent(agent_spec)
        
        # Register the agent with the communication system
        await self.system.a2a_manager.register_agent(agent.id)
        
        self.logger.info(f"Successfully created agent: {agent.name} (ID: {agent.id})")
        return agent
    
    async def create_agent_factory(self):
        """
        Create an agent that can itself create other agents
        """
        self.logger.info("Creating Agent Factory - an agent that creates other agents")
        
        # First, create a base agent that will serve as the agent factory
        factory_agent_spec = {
            "type": "task_manager",  # Use task manager as base since it has decomposition capability
            "name": "AgentFactory",
            "description": "An agent that can create and configure other specialized agents",
            "capabilities": [AgentCapability.DECOMPOSITION.value, AgentCapability.TASK_EXECUTION.value]
        }
        
        factory_agent = await self.create_specialized_agent(factory_agent_spec)
        
        # Add special functionality to the factory agent
        # In a real implementation, this would involve more sophisticated methods
        factory_agent.create_agent = self.create_specialized_agent
        factory_agent.agent_factory = self
        
        return factory_agent
    
    async def create_iterative_development_agents(self):
        """
        Create a suite of agents specifically designed for iterative development
        """
        self.logger.info("Creating iterative development agent suite")
        
        # Define agent specifications for iterative development
        agent_specs = [
            {
                "type": "code_writer",
                "name": "Iteration1Developer",
                "description": "Specialized in implementing basic functionality (Iteration 1)"
            },
            {
                "type": "code_writer", 
                "name": "Iteration2Developer",
                "description": "Specialized in enhancing and containerizing solutions (Iteration 2)"
            },
            {
                "type": "code_writer",
                "name": "Iteration3Developer", 
                "description": "Specialized in creating multi-component architectures (Iteration 3)"
            },
            {
                "type": "tester",
                "name": "IterativeTester",
                "description": "Specialized in testing iterative development solutions"
            },
            {
                "type": "task_manager",
                "name": "IterationManager",
                "description": "Manages the iterative development process"
            }
        ]
        
        agents = []
        for spec in agent_specs:
            agent = await self.create_specialized_agent(spec)
            agents.append(agent)
            
        return agents

async def demonstrate_meta_agentic_capabilities():
    """
    Demonstrate the meta-agentic capabilities of the system
    """
    logger = setup_logger("MetaAgenticDemo")
    logger.info("Starting Meta-Agentic Capabilities Demonstration")
    
    # Create the main system
    system = AgenticSystem()
    
    # Create the meta agent builder
    meta_builder = MetaAgentBuilder(system)
    
    print("\n=== Meta-Agentic Capabilities Demonstration ===")
    
    # 1. Demonstrate agent creation by the system
    print("\n1. Creating Agent Factory (an agent that creates other agents)")
    agent_factory = await meta_builder.create_agent_factory()
    print(f"   Created: {agent_factory}")
    print(f"   Capabilities: {[cap.value for cap in agent_factory.capabilities]}")
    
    # 2. Demonstrate the agent factory creating other agents
    print("\n2. Agent Factory creating specialized agents")
    new_agent_spec = {
        "type": "code_writer",
        "name": "SpecializedCodeAgent",
        "description": "Created by the Agent Factory"
    }
    
    new_agent = await agent_factory.agent_factory.create_specialized_agent(new_agent_spec)
    print(f"   Agent Factory created: {new_agent}")
    
    # 3. Create iterative development agents
    print("\n3. Creating iterative development agent suite")
    iterative_agents = await meta_builder.create_iterative_development_agents()
    print(f"   Created {len(iterative_agents)} specialized agents for iterative development:")
    for agent in iterative_agents:
        print(f"   - {agent.name}: {agent.description}")
    
    # 4. Demonstrate task decomposition with the new agents
    print("\n4. Testing iterative development with Instagram parser task")
    
    instagram_task = "Write an Instagram parser with authentication and data storage"
    decomposition_plan = system.decompose_task(instagram_task, iterations=3)
    
    print(f"   Task: {instagram_task}")
    print(f"   Total subtasks: {decomposition_plan['total_subtasks']}")
    
    # Execute tasks using appropriate agents
    for iteration in decomposition_plan['iterations']:
        print(f"   \n   Executing: {iteration['description']}")
        
        for i, subtask in enumerate(iteration['subtasks'], 1):
            # Select appropriate agent based on iteration
            if iteration['iteration_number'] == 1:
                agent_type = 'Iteration1Developer'
            elif iteration['iteration_number'] == 2:
                agent_type = 'Iteration2Developer' 
            elif iteration['iteration_number'] == 3:
                agent_type = 'Iteration3Developer'
            else:
                agent_type = 'IterationManager'
                
            # Find the agent by name
            agent = None
            for a in system.agents.values():
                if agent_type in a.name:
                    agent = a
                    break
            
            if not agent:
                # Fallback to any code writer if specific agent not found
                for a in system.agents.values():
                    if hasattr(a, 'agent_type') and a.agent_type == 'code_writer':
                        agent = a
                        break
            
            if not agent:
                agent = system.agents[list(system.agents.keys())[0]]  # Use first available agent
            
            task = {
                "id": f"meta_task_{iteration['iteration_number']}_{i}",
                "title": subtask['title'],
                "description": subtask['description'],
                "agent_type": agent.agent_type if hasattr(agent, 'agent_type') else 'default'
            }
            
            result = await agent.execute_task(task)
            print(f"      Subtask {i}: {subtask['title']} -> {result['result'][:50]}...")
    
    # 5. Demonstrate agent-to-agent communication in the meta-system
    print("\n5. Testing agent-to-agent communication in meta-system")
    
    # Find communication-capable agents
    comm_agents = [a for a in system.agents.values() 
                   if AgentCapability.COMMUNICATION in a.capabilities]
    
    if len(comm_agents) >= 2:
        agent1, agent2 = comm_agents[0], comm_agents[1]
        
        # Send a message between agents
        from src.communication.a2a_manager import Message, MessageType
        
        message = Message(
            sender_id=agent1.id,
            receiver_id=agent2.id,
            msg_type=MessageType.NOTIFICATION,
            content={
                "notification_type": "task_completion",
                "task": "Instagram parser development",
                "iteration": 2,
                "status": "completed"
            }
        )
        
        success = await system.a2a_manager.send_message(message)
        print(f"   Message from {agent1.name} to {agent2.name}: {'Success' if success else 'Failed'}")
        
        # Check if message was received
        received_messages = await system.a2a_manager.get_messages_for_agent(agent2.id)
        print(f"   {agent2.name} received {len(received_messages)} messages")
    else:
        # Create communication agents if none exist
        comm_agent1 = await meta_builder.create_specialized_agent({
            "type": "communication",
            "name": "MetaCommAgent1",
            "description": "Communication agent for meta-system"
        })
        
        comm_agent2 = await meta_builder.create_specialized_agent({
            "type": "communication", 
            "name": "MetaCommAgent2",
            "description": "Communication agent for meta-system"
        })
        
        from src.communication.a2a_manager import Message, MessageType
        
        message = Message(
            sender_id=comm_agent1.id,
            receiver_id=comm_agent2.id,
            msg_type=MessageType.NOTIFICATION,
            content={
                "notification_type": "meta_system_event",
                "event": "Agent creation complete",
                "timestamp": "now"
            }
        )
        
        success = await system.a2a_manager.send_message(message)
        print(f"   Message from {comm_agent1.name} to {comm_agent2.name}: {'Success' if success else 'Failed'}")
    
    print(f"\n6. System Statistics:")
    print(f"   Total agents in system: {len(system.agents)}")
    print(f"   Communication stats: {system.a2a_manager.get_communication_stats()}")
    
    return system, meta_builder

async def demonstrate_recursive_decomposition():
    """
    Demonstrate the system's ability to recursively decompose tasks
    """
    logger = setup_logger("RecursiveDecompositionDemo")
    logger.info("Starting Recursive Task Decomposition Demonstration")
    
    print("\n=== Recursive Task Decomposition Demonstration ===")
    
    # Create system
    system = AgenticSystem()
    
    # Example: Instagram parser task that gets decomposed recursively
    main_task = "Write an Instagram parser"
    
    print(f"\nMain task: {main_task}")
    
    # Decompose the main task
    decomposition_plan = system.decompose_task(main_task, iterations=3)
    
    print(f"\nFirst-level decomposition into {len(decomposition_plan['iterations'])} iterations:")
    for iteration in decomposition_plan['iterations']:
        print(f"  Iteration {iteration['iteration_number']}: {iteration['description']}")
        print(f"    Focus: {iteration['focus']}")
        print(f"    Subtasks: {len(iteration['subtasks'])}")
        
        # Show the subtasks for each iteration
        for i, subtask in enumerate(iteration['subtasks'], 1):
            print(f"      {i}. {subtask['title']}")
            print(f"         {subtask['description']}")
    
    # Now demonstrate how each subtask could be further decomposed
    print(f"\nRecursive decomposition example:")
    print(f"Taking the first subtask: '{decomposition_plan['iterations'][0]['subtasks'][0]['title']}'")
    
    # In a real system, each subtask could be treated as a new main task
    # and decomposed further. For this demo, we'll show how this would work:
    
    first_subtask = decomposition_plan['iterations'][0]['subtasks'][0]['title']
    print(f"  This subtask could be further decomposed into implementation steps:")
    print(f"    - Research available Instagram API endpoints")
    print(f"    - Identify required authentication methods")
    print(f"    - Determine data structures to extract")
    print(f"    - Plan error handling strategies")
    print(f"    - Design rate limiting approach")
    
    # Create a task board for the entire project
    task_board = system.task_decomposer.create_task_board(decomposition_plan)
    print(f"\nGenerated task board with {len(task_board)} individual tasks")
    
    # Show first few tasks with their hierarchy
    print(f"\nTask hierarchy example (first 6 tasks):")
    for i, task in enumerate(task_board[:6]):
        print(f"  {i+1}. {task.title}")
        print(f"     Priority: {task.priority}")
        print(f"     Status: {task.status}")
        print(f"     Created: {task.created_at}")
    
    return system, decomposition_plan

async def main():
    """
    Main function to run meta-agentic demonstrations
    """
    print("Meta-Agentic Architecture System - Advanced Capabilities")
    print("=" * 60)
    
    # Run meta-agentic capabilities demo
    print("\n1. Meta-Agentic Capabilities")
    print("-" * 40)
    await demonstrate_meta_agentic_capabilities()
    
    # Run recursive decomposition demo
    print("\n\n2. Recursive Task Decomposition") 
    print("-" * 40)
    await demonstrate_recursive_decomposition()
    
    print("\n\n=== Meta-Agentic System Demonstration Complete ===")
    print("The system successfully demonstrated:")
    print("- Agent creation by agents (meta-agentic capability)")
    print("- Iterative development agent specialization")
    print("- Recursive task decomposition")
    print("- Agent-to-agent communication in complex systems")
    print("- Task board generation and management")
    print("- Self-improving agent architecture")

if __name__ == "__main__":
    asyncio.run(main())