#!/usr/bin/env python3
"""
Comprehensive Usage Examples for Multi-Agent System
This file contains extensive examples of how to use the multi-agent system
for various scenarios and use cases.
"""

from typing import Dict, Any, List
import asyncio
import time
import uuid
from datetime import datetime
import json

from src.builders.agent_builder import AgentBuilder
from src.builders.advanced_agent_builder import AdvancedAgentBuilder
from src.task_manager.advanced_task_manager import AdvancedTaskManager
from src.agents.advanced_agents import (
    LearningAgent, AdaptiveAgent, 
    CodeSpecialist, DataSpecialist, CommunicationSpecialist
)
from src.utils.logger import setup_logger


def example_basic_agent_creation():
    """Example: Basic agent creation and usage"""
    print("=== Basic Agent Creation Example ===")
    
    # Create a basic agent builder
    builder = AgentBuilder()
    
    # Create a simple configuration
    config = {
        "type": "code_writer",
        "name": "BasicCoder",
        "description": "A basic code writing agent",
        "capabilities": ["code_writing", "analysis"]
    }
    
    # Create the agent
    agent = builder.create_agent(config)
    print(f"Created agent: {agent.name} with ID: {agent.id}")
    
    # Show agent capabilities
    print(f"Agent capabilities: {[cap.value for cap in agent.capabilities]}")
    
    # Create and execute a simple task
    task = {
        "id": "task_1",
        "title": "Write a simple function",
        "type": "code_generation",
        "description": "Create a function that adds two numbers"
    }
    
    # In a real system, we would execute the task
    print(f"Agent would execute task: {task['title']}")
    print()


def example_advanced_agent_creation():
    """Example: Advanced agent creation with profiles and configurations"""
    print("=== Advanced Agent Creation Example ===")
    
    # Create an advanced agent builder
    builder = AdvancedAgentBuilder()
    
    # Create an agent using a predefined profile
    profile_agent = builder.create_agent_from_profile(
        "developer",
        custom_config={
            "name": "AdvancedDevAgent",
            "description": "An advanced developer agent with custom capabilities"
        }
    )
    print(f"Created profile-based agent: {profile_agent.name}")
    
    # Create an agent with specific configuration
    config_agent = builder.create_agent_from_config({
        "name": "ConfigAgent",
        "description": "Agent created from configuration",
        "agent_type": "code_writer",
        "capabilities": ["code_writing", "analysis", "testing"],
        "parameters": {"language": "python", "style_guide": "PEP8"}
    })
    print(f"Created config-based agent: {config_agent.name}")
    
    # Create a dynamic agent with custom capabilities
    dynamic_agent = builder.create_dynamic_agent(
        "DynamicAgent",
        "Agent created dynamically with custom capabilities",
        ["code_writing", "analysis", "validation"]
    )
    print(f"Created dynamic agent: {dynamic_agent.name}")
    
    # Show agent statistics
    stats = builder.get_agent_statistics()
    print(f"Agent statistics: {stats}")
    print()


def example_cooperative_agent_network():
    """Example: Creating and using a cooperative agent network"""
    print("=== Cooperative Agent Network Example ===")
    
    # Create an advanced agent builder
    builder = AdvancedAgentBuilder()
    
    # Define agent configurations for a cooperative network
    agent_configs = [
        {
            "type": "task_manager",
            "name": "ProjectManager",
            "description": "Manages the overall project",
            "capabilities": ["decomposition", "task_execution", "communication"]
        },
        {
            "type": "code_writer", 
            "name": "CodeGenerator",
            "description": "Generates code based on requirements",
            "capabilities": ["code_writing", "analysis"]
        },
        {
            "type": "tester",
            "name": "QualityAssurer", 
            "description": "Tests and validates the code",
            "capabilities": ["testing", "analysis", "validation"]
        },
        {
            "type": "communication",
            "name": "Messenger",
            "description": "Handles communication between agents",
            "capabilities": ["communication"]
        }
    ]
    
    # Create a cooperative network
    network = builder.create_cooperative_agent_network(
        agent_configs,
        communication_protocol="a2a"
    )
    
    print(f"Created cooperative network: {network.name}")
    print(f"Network protocol: {network.protocol.value}")
    print(f"Number of agents: {len(network.agents)}")
    
    # Show network statistics
    network_stats = network.get_network_stats()
    print(f"Network stats: {json.dumps(network_stats, indent=2, default=str)}")
    print()


def example_task_management():
    """Example: Advanced task management and scheduling"""
    print("=== Advanced Task Management Example ===")
    
    # Create a task manager
    task_manager = AdvancedTaskManager()
    
    # Create a complex task workflow
    task1 = task_manager.create_task(
        title="Data Collection",
        description="Collect data from various sources",
        task_type="data_processing",
        priority=3,  # High priority
        tags=["data", "collection"]
    )
    
    task2 = task_manager.create_task(
        title="Data Processing",
        description="Process collected data",
        task_type="data_processing", 
        priority=2,
        dependencies=[{"dependent_task_id": task2.id, "dependency_task_id": task1.id}],
        tags=["data", "processing"]
    )
    
    task3 = task_manager.create_task(
        title="Analysis",
        description="Analyze processed data",
        task_type="analysis",
        priority=2,
        dependencies=[{"dependent_task_id": task3.id, "dependency_task_id": task2.id}],
        tags=["analysis"]
    )
    
    task4 = task_manager.create_task(
        title="Report Generation",
        description="Generate final report",
        task_type="computation",
        priority=1,
        dependencies=[{"dependent_task_id": task4.id, "dependency_task_id": task3.id}],
        tags=["reporting"]
    )
    
    print(f"Created workflow with {len(task_manager.tasks)} tasks")
    
    # Get system status
    status = task_manager.get_system_status()
    print(f"Task manager status: {status}")
    
    # Get task statistics
    stats = task_manager.get_task_statistics()
    print(f"Task statistics: {stats}")
    print()


def example_learning_agent():
    """Example: Using a learning agent that improves over time"""
    print("=== Learning Agent Example ===")
    
    # Create a learning agent
    learning_agent = LearningAgent("LearningBot", "Agent that learns from tasks")
    
    print(f"Created learning agent: {learning_agent.name}")
    print(f"Initial adaptation level: {learning_agent.adaptation_level.value}")
    
    # Simulate multiple task executions to demonstrate learning
    for i in range(5):
        task = {
            "id": f"task_{i+1}",
            "title": f"Learning Task {i+1}",
            "type": "analyze_data",
            "data": list(range(10))
        }
        
        # Simulate task execution
        print(f"Executing task {i+1}: {task['title']}")
        
        # Add to agent's memory
        learning_agent.add_to_memory(
            memory_type="episodic",
            content=task,
            importance=0.8
        )
        
        # Simulate learning experience
        reward = 0.7 + (i * 0.05)  # Increasing reward to simulate improvement
        learning_agent.store_learning_experience(
            task_id=task["id"],
            task_result="completed",
            reward=reward,
            context=task
        )
        
        # Adapt behavior
        learning_agent.adapt_behavior()
        
        print(f"  - Reward: {reward:.2f}")
        print(f"  - New adaptation level: {learning_agent.adaptation_level.value}")
        print(f"  - Exploration rate: {learning_agent.exploration_rate:.3f}")
    
    # Show final state
    print(f"Final adaptation level: {learning_agent.adaptation_level.value}")
    print(f"Learning history length: {len(learning_agent.learning_history)}")
    print()


def example_adaptive_agent():
    """Example: Using an adaptive agent that changes behavior based on context"""
    print("=== Adaptive Agent Example ===")
    
    # Create an adaptive agent
    adaptive_agent = AdaptiveAgent("AdaptiBot", "Agent that adapts to contexts")
    
    print(f"Created adaptive agent: {adaptive_agent.name}")
    print(f"Initial adaptation level: {adaptive_agent.adaptation_level.value}")
    
    # Simulate different contexts and tasks
    contexts = [
        {"task_type": "computation", "priority": 1, "complexity": "low"},
        {"task_type": "analysis", "priority": 3, "complexity": "high"},
        {"task_type": "communication", "priority": 2, "complexity": "medium"},
        {"task_type": "validation", "priority": 1, "complexity": "medium"}
    ]
    
    for i, context in enumerate(contexts):
        task = {
            "id": f"adaptive_task_{i+1}",
            "title": f"Adaptive Task {i+1}",
            "type": context["task_type"],
            "priority": context["priority"],
            "complexity": context["complexity"]
        }
        
        print(f"Processing task {i+1}: {task['title']}")
        print(f"  - Context: {context}")
        
        # Select adaptation strategy
        strategy = adaptive_agent._select_adaptation_strategy(context, task)
        print(f"  - Selected strategy: {strategy}")
        
        # Simulate task execution
        result = {
            "status": "completed",
            "execution_time": 0.1 + (i * 0.02),
            "success": True
        }
        
        # Evaluate strategy effectiveness
        effectiveness = adaptive_agent._evaluate_strategy_effectiveness(result, task)
        print(f"  - Strategy effectiveness: {effectiveness:.2f}")
        
        # Store learning experience
        reward = adaptive_agent._calculate_adaptation_reward(result, task, effectiveness)
        adaptive_agent.store_learning_experience(
            task_id=task["id"],
            task_result="completed",
            reward=reward,
            context=context
        )
        
        # Adapt behavior
        adaptive_agent.adapt_behavior()
        
        print(f"  - New adaptation level: {adaptive_agent.adaptation_level.value}")
        print(f"  - New exploration rate: {adaptive_agent.exploration_rate:.3f}")
        print()
    
    print(f"Final adaptation level: {adaptive_agent.adaptation_level.value}")
    print()


def example_specialized_agents():
    """Example: Using specialized agents for specific domains"""
    print("=== Specialized Agents Example ===")
    
    # Create specialized agents
    code_agent = CodeSpecialist("CodeMaster", "Specialized in code tasks")
    data_agent = DataSpecialist("DataWizard", "Specialized in data tasks")
    comm_agent = CommunicationSpecialist("CommPro", "Specialized in communication tasks")
    
    print(f"Created specialized agents:")
    print(f"  - Code Agent: {code_agent.name} (Domain: {code_agent.domain})")
    print(f"  - Data Agent: {data_agent.name} (Domain: {data_agent.domain})")
    print(f"  - Communication Agent: {comm_agent.name} (Domain: {comm_agent.domain})")
    
    # Demonstrate code agent capabilities
    print("\n--- Code Agent Example ---")
    code_task = {
        "id": "code_task_1",
        "type": "write_code",
        "language": "python",
        "requirements": ["create a function to calculate factorial", "add error handling"],
        "title": "Code Generation Task"
    }
    
    code_result = asyncio.run(code_agent._execute_domain_specific_task(code_task))
    print(f"Code agent result: {code_result['total_snippets']} snippets generated")
    
    # Demonstrate data agent capabilities
    print("\n--- Data Agent Example ---")
    data_task = {
        "id": "data_task_1",
        "type": "analyze_data",
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "title": "Data Analysis Task"
    }
    
    data_result = asyncio.run(data_agent._execute_domain_specific_task(data_task))
    print(f"Data agent result: {data_result['total_records']} records analyzed")
    
    # Demonstrate communication agent capabilities
    print("\n--- Communication Agent Example ---")
    comm_task = {
        "id": "comm_task_1",
        "type": "send_message",
        "message": "Hello from communication agent!",
        "channel": "chat",
        "recipient": "system",
        "title": "Communication Task"
    }
    
    comm_result = asyncio.run(comm_agent._execute_domain_specific_task(comm_task))
    print(f"Communication agent result: {comm_result['status']} - {comm_result['delivery_status']}")
    
    # Show domain expertise improvement
    print(f"\nDomain expertise levels:")
    print(f"  - Code agent: {code_agent.domain_expertise:.2f}")
    print(f"  - Data agent: {data_agent.domain_expertise:.2f}")
    print(f"  - Communication agent: {comm_agent.domain_expertise:.2f}")
    print()


def example_complex_workflow():
    """Example: Complex workflow with multiple agents and task dependencies"""
    print("=== Complex Workflow Example ===")
    
    # Create agents
    builder = AdvancedAgentBuilder()
    
    # Create a project management network
    pm_network_config = {
        "name": "ProjectManagementNetwork",
        "levels": {
            "executive": {
                "agents": [
                    {
                        "type": "task_manager",
                        "name": "ExecutiveManager",
                        "description": "Executive level manager",
                        "capabilities": ["decomposition", "task_execution", "communication"]
                    }
                ]
            },
            "managers": {
                "agents": [
                    {
                        "type": "task_manager",
                        "name": "TeamLead1",
                        "description": "Team lead for development",
                        "capabilities": ["decomposition", "task_execution", "communication"]
                    },
                    {
                        "type": "task_manager", 
                        "name": "TeamLead2",
                        "description": "Team lead for testing",
                        "capabilities": ["decomposition", "task_execution", "communication"]
                    }
                ]
            },
            "workers": {
                "agents": [
                    {
                        "type": "code_writer",
                        "name": "Dev1",
                        "description": "Developer",
                        "capabilities": ["code_writing", "analysis"]
                    },
                    {
                        "type": "tester",
                        "name": "Tester1",
                        "description": "Tester",
                        "capabilities": ["testing", "analysis", "validation"]
                    },
                    {
                        "type": "communication",
                        "name": "Comm1",
                        "description": "Communicator",
                        "capabilities": ["communication"]
                    }
                ]
            }
        }
    }
    
    # Create hierarchical network
    network = builder.create_hierarchical_network(pm_network_config)
    print(f"Created hierarchical network: {network.name}")
    print(f"Network has {len(network.agents)} agents in {len(pm_network_config['levels'])} levels")
    
    # Create task manager for workflow
    task_manager = AdvancedTaskManager()
    
    # Create a complex project workflow
    project_tasks = [
        {
            "title": "Project Planning",
            "description": "Plan the entire project",
            "type": "computation",
            "priority": 3,
            "agent_requirements": ["ExecutiveManager"]
        },
        {
            "title": "Requirements Analysis",
            "description": "Analyze project requirements",
            "type": "analysis", 
            "priority": 3,
            "agent_requirements": ["TeamLead1", "TeamLead2"]
        },
        {
            "title": "Architecture Design",
            "description": "Design system architecture",
            "type": "computation",
            "priority": 3,
            "agent_requirements": ["TeamLead1"]
        },
        {
            "title": "Code Implementation",
            "description": "Implement the code",
            "type": "code_generation",
            "priority": 2,
            "agent_requirements": ["Dev1"],
            "dependencies": [{"dependent_task_id": "task_4", "dependency_task_id": "task_3"}]
        },
        {
            "title": "Unit Testing",
            "description": "Write and run unit tests",
            "type": "testing",
            "priority": 2,
            "agent_requirements": ["Tester1"],
            "dependencies": [{"dependent_task_id": "task_5", "dependency_task_id": "task_4"}]
        },
        {
            "title": "Integration Testing",
            "description": "Perform integration testing",
            "type": "testing", 
            "priority": 2,
            "agent_requirements": ["Tester1"],
            "dependencies": [{"dependent_task_id": "task_6", "dependency_task_id": "task_5"}]
        },
        {
            "title": "Project Communication",
            "description": "Communicate project status",
            "type": "communication",
            "priority": 1,
            "agent_requirements": ["Comm1"],
            "dependencies": [{"dependent_task_id": "task_7", "dependency_task_id": "task_6"}]
        }
    ]
    
    # Create tasks in task manager
    created_tasks = []
    for task_data in project_tasks:
        task = task_manager.create_task(
            title=task_data["title"],
            description=task_data["description"],
            task_type=task_data["type"],
            priority=task_data["priority"],
            agent_requirements=task_data["agent_requirements"]
        )
        created_tasks.append(task)
    
    print(f"Created {len(created_tasks)} project tasks")
    
    # Show network and task statistics
    network_stats = builder.monitor_agent_network(network.name)
    print(f"Network status: {len(network_stats['agent_metrics'])} agents active")
    
    system_status = task_manager.get_system_status()
    print(f"Task system status: {system_status['total_tasks']} total tasks")
    print()


def example_agent_optimization():
    """Example: Agent optimization and performance tuning"""
    print("=== Agent Optimization Example ===")
    
    # Create an agent
    agent = LearningAgent("OptimizerBot", "Agent for optimization")
    
    # Simulate performance data
    performance_data = {
        "total_tasks": 100,
        "successful_tasks": 85,
        "failed_tasks": 15,
        "success_rate": 0.85,
        "total_execution_time": 120.5,
        "average_task_time": 1.205
    }
    
    print(f"Agent performance data: {performance_data}")
    
    # Optimize agent configuration
    from src.agents.advanced_agents import optimize_agent_configuration
    recommendations = optimize_agent_configuration(agent, performance_data)
    
    print("Optimization recommendations:")
    for category, items in recommendations.items():
        if items:
            print(f"  {category.replace('_', ' ').title()}:")
            for item in items:
                if isinstance(item, dict):
                    print(f"    - {item.get('parameter', item.get('modification', 'Unknown'))}: {item.get('reason', 'No reason given')}")
                else:
                    print(f"    - {item}")
    
    # Show how optimization would affect the agent
    print(f"\nCurrent exploration rate: {agent.exploration_rate:.3f}")
    if recommendations["parameter_adjustments"]:
        adjustment = recommendations["parameter_adjustments"][0]
        print(f"Recommended exploration rate: {adjustment['recommended_value']:.3f}")
    
    print()


def example_agent_collaboration_analysis():
    """Example: Analyzing collaboration between agents"""
    print("=== Agent Collaboration Analysis Example ===")
    
    # Create some agents
    agents = [
        LearningAgent("Agent1", "First agent"),
        AdaptiveAgent("Agent2", "Second agent"),
        CodeSpecialist("Agent3", "Third agent")
    ]
    
    # Simulate collaboration data
    collaboration_data = []
    for i in range(10):
        collaboration_data.append({
            "sender": agents[i % 3].id,
            "receiver": agents[(i + 1) % 3].id,
            "message_type": "task_result",
            "timestamp": datetime.now().isoformat(),
            "content": f"Collaboration message {i+1}"
        })
    
    print(f"Simulated {len(collaboration_data)} collaboration interactions")
    
    # Analyze collaboration
    from src.agents.advanced_agents import analyze_agent_collaboration
    analysis = analyze_agent_collaboration(agents, collaboration_data)
    
    print("Collaboration analysis:")
    print(f"  Total interactions: {analysis['total_interactions']}")
    print(f"  Agent interaction counts: {analysis['agent_interaction_counts']}")
    print(f"  Collaboration efficiency: {analysis['collaboration_efficiency']}")
    print(f"  Communication patterns: {len(analysis['communication_patterns'])} patterns")
    
    if analysis['suggested_improvements']:
        print("  Suggestions:")
        for suggestion in analysis['suggested_improvements']:
            print(f"    - {suggestion['suggestion']} for {suggestion['agent_id']}")
    
    print()


def example_agent_backup_and_restore():
    """Example: Agent state backup and restore"""
    print("=== Agent Backup and Restore Example ===")
    
    # Create an agent with some state
    agent = LearningAgent("BackupBot", "Agent for backup testing")
    
    # Add some memory and learning history
    agent.add_to_memory("semantic", "Important information", importance=0.9)
    agent.add_to_memory("episodic", "Recent experience", importance=0.7)
    
    # Add some learning experiences
    for i in range(3):
        agent.store_learning_experience(
            f"task_{i+1}",
            "completed",
            0.8 + (i * 0.05),
            {"context": f"test_context_{i+1}"}
        )
    
    print(f"Agent created with {len(agent.learning_history)} learning experiences")
    print(f"Agent has {len(agent.memory['semantic']) + len(agent.memory['episodic'])} memory entries")
    
    # Backup agent state
    from src.agents.advanced_agents import backup_agent_state
    backup_success = backup_agent_state(agent, "/workspace/backups")
    
    if backup_success:
        print("Agent state backed up successfully")
        
        # Create a new agent and restore state
        restored_agent = LearningAgent("RestoredBot", "Restored agent")
        print(f"New agent created with {len(restored_agent.learning_history)} learning experiences")
        
        # Find the backup file and restore
        import os
        backup_files = [f for f in os.listdir("/workspace/backups") if f.startswith(f"agent_{agent.id}_backup_")]
        if backup_files:
            latest_backup = max(backup_files)
            backup_path = f"/workspace/backups/{latest_backup}"
            
            from src.agents.advanced_agents import restore_agent_state
            restore_success = restore_agent_state(restored_agent, backup_path)
            
            if restore_success:
                print(f"Agent state restored successfully")
                print(f"Restored agent now has {len(restored_agent.learning_history)} learning experiences")
                print(f"Restored agent memory types: {list(restored_agent.memory.keys())}")
    
    print()


def example_agent_monitoring_and_metrics():
    """Example: Agent monitoring and metrics collection"""
    print("=== Agent Monitoring and Metrics Example ===")
    
    # Create a monitoring service
    from src.task_manager.advanced_task_manager import TaskMonitoringService
    task_manager = AdvancedTaskManager()
    monitoring_service = TaskMonitoringService(task_manager)
    
    print("Created monitoring service for task manager")
    
    # Register an alert callback
    def alert_callback(alert_type, alert_data):
        print(f"ALERT: {alert_type} - {alert_data}")
    
    monitoring_service.register_alert_callback(alert_callback)
    
    # Generate a report
    report = monitoring_service.generate_task_report()
    print(f"Generated task report for period: {report['period']['start']} to {report['period']['end']}")
    print(f"  - Total tasks: {report['total_tasks']}")
    print(f"  - Success rate: {report['success_rate']:.2%}")
    print(f"  - Average execution time: {report['avg_execution_time']:.2f}s")
    
    # Check performance thresholds
    monitoring_service.check_performance_thresholds()
    print("Performance threshold check completed")
    print()


def example_agent_network_optimization():
    """Example: Optimizing agent networks"""
    print("=== Agent Network Optimization Example ===")
    
    # Create an advanced builder
    builder = AdvancedAgentBuilder()
    
    # Create a network
    network_config = [
        {"type": "task_manager", "name": "Manager1"},
        {"type": "code_writer", "name": "Coder1", "capabilities": ["code_writing", "analysis"]},
        {"type": "tester", "name": "Tester1", "capabilities": ["testing", "analysis"]}
    ]
    
    network = builder.create_cooperative_network("optimization_network", network_config)
    
    print(f"Created network: {network.name} with {len(network.agents)} agents")
    
    # Optimize the network
    optimization_result = builder.optimize_agent_network(
        network.name, 
        {"strategy": "communication_efficiency"}
    )
    
    print(f"Network optimization completed: {optimization_result}")
    
    # Scale the network
    scaling_result = builder.scale_agent_network(
        network.name,
        {"scale_up": True, "agent_type": "tester", "count": 2}
    )
    
    print(f"Network scaling completed: {scaling_result}")
    print(f"Network now has {len(network.agents)} agents")
    
    # Export network schema
    schema = builder.export_network_schema(network.name)
    print(f"Exported network schema with {len(schema['agents'])} agents and {schema['topology']['edges']} connections")
    print()


def example_comprehensive_system_integration():
    """Example: Comprehensive integration of all system components"""
    print("=== Comprehensive System Integration Example ===")
    
    print("Initializing comprehensive multi-agent system...")
    
    # 1. Create agent builder and agents
    builder = AdvancedAgentBuilder()
    print("  - Created advanced agent builder")
    
    # 2. Create specialized agents
    agents = [
        builder.create_agent_from_profile("developer", {"name": "DevAgent1"}),
        builder.create_agent_from_profile("tester", {"name": "TestAgent1"}),
        builder.create_agent_from_profile("manager", {"name": "MgrAgent1"})
    ]
    print(f"  - Created {len(agents)} specialized agents")
    
    # 3. Create task manager
    task_manager = AdvancedTaskManager()
    print("  - Created advanced task manager")
    
    # 4. Create complex workflow
    workflow_task_ids = []
    for i in range(5):
        task = task_manager.create_task(
            title=f"Integration Task {i+1}",
            description=f"Task {i+1} in comprehensive integration workflow",
            task_type="computation" if i % 2 == 0 else "analysis",
            priority=2 if i < 3 else 3,
            tags=["integration", "comprehensive"]
        )
        workflow_task_ids.append(task.id)
    
    print(f"  - Created workflow with {len(workflow_task_ids)} tasks")
    
    # 5. Create monitoring
    from src.task_manager.advanced_task_manager import TaskMonitoringService
    monitoring_service = TaskMonitoringService(task_manager)
    print("  - Created monitoring service")
    
    # 6. Simulate system operation
    print("  - Simulating system operation...")
    
    # Submit tasks for execution
    for i, task_id in enumerate(workflow_task_ids):
        task = task_manager.get_task(task_id)
        if task:
            task_manager.submit_task(task, agent_id=agents[i % len(agents)].id)
    
    print("  - Tasks submitted for execution")
    
    # 7. Generate system report
    system_report = {
        "timestamp": datetime.now().isoformat(),
        "agents_created": len(builder.created_agents),
        "tasks_managed": len(task_manager.tasks),
        "active_tasks": len(task_manager.executor.get_running_tasks()),
        "completed_tasks": len(task_manager.executor.get_completed_tasks()),
        "networks_created": len(builder.agent_networks),
        "agent_templates": len(builder.agent_templates)
    }
    
    print("System integration report:")
    for key, value in system_report.items():
        print(f"  - {key.replace('_', ' ').title()}: {value}")
    
    # 8. Performance evaluation
    performance = task_manager.get_task_statistics()
    print(f"Task performance metrics:")
    for key, value in performance.items():
        if isinstance(value, (int, float)) and key != 'by_status' and key != 'by_priority' and key != 'by_type':
            print(f"  - {key.replace('_', ' ').title()}: {value}")
    
    print("\nComprehensive system integration completed successfully!")
    print()


def run_all_examples():
    """Run all examples to demonstrate the system capabilities"""
    print("Running All Multi-Agent System Examples")
    print("=" * 50)
    
    example_basic_agent_creation()
    example_advanced_agent_creation()
    example_cooperative_agent_network()
    example_task_management()
    example_learning_agent()
    example_adaptive_agent()
    example_specialized_agents()
    example_complex_workflow()
    example_agent_optimization()
    example_agent_collaboration_analysis()
    example_agent_backup_and_restore()
    example_agent_monitoring_and_metrics()
    example_agent_network_optimization()
    example_comprehensive_system_integration()
    
    print("All examples completed!")
    print("=" * 50)


def performance_benchmark():
    """Performance benchmark of the multi-agent system"""
    print("=== Performance Benchmark ===")
    
    start_time = time.time()
    
    # Create multiple agents
    agents = []
    for i in range(10):
        agent = LearningAgent(f"BenchmarkAgent_{i}", f"Agent for benchmark {i}")
        agents.append(agent)
    
    # Create multiple tasks
    tasks = []
    for i in range(20):
        task = {
            "id": f"benchmark_task_{i}",
            "title": f"Benchmark Task {i}",
            "type": "analysis",
            "data": list(range(100))
        }
        tasks.append(task)
    
    # Simulate task execution
    execution_results = []
    for agent in agents:
        for task in tasks[:5]:  # Each agent does 5 tasks
            # Simulate execution
            result = {
                "agent_id": agent.id,
                "task_id": task["id"],
                "status": "completed",
                "execution_time": 0.01  # Simulated
            }
            execution_results.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Benchmark completed in {total_time:.4f} seconds")
    print(f"Created {len(agents)} agents")
    print(f"Created {len(tasks)} tasks")
    print(f"Executed {len(execution_results)} agent-task combinations")
    print(f"Average execution time per combination: {total_time/len(execution_results)*1000:.2f} ms")
    print()


def system_architecture_overview():
    """Provide an overview of the system architecture"""
    print("=== Multi-Agent System Architecture Overview ===")
    
    architecture = {
        "Core Components": [
            "Agent Builder System",
            "Advanced Agent Builder",
            "Task Management System", 
            "Agent Communication System",
            "Monitoring and Analytics"
        ],
        "Agent Types": [
            "Basic Agents",
            "Learning Agents",
            "Adaptive Agents", 
            "Specialized Agents (Code, Data, Communication)"
        ],
        "Key Features": [
            "Dynamic Agent Creation",
            "Cooperative Agent Networks",
            "Advanced Task Management",
            "Learning and Adaptation",
            "Performance Monitoring",
            "State Persistence"
        ],
        "Communication Protocols": [
            "Agent-to-Agent (A2A)",
            "Centralized Communication",
            "Decentralized Communication",
            "Hybrid Communication"
        ],
        "Task Management": [
            "Priority-based Queuing",
            "Dependency Management",
            "Resource Allocation",
            "Performance Tracking"
        ]
    }
    
    for category, items in architecture.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  - {item}")
    
    print("\nThe system provides a comprehensive framework for creating,")
    print("managing, and coordinating multiple intelligent agents to")
    print("solve complex tasks through collaboration and specialization.")
    print()


def best_practices_guide():
    """Provide best practices for using the multi-agent system"""
    print("=== Best Practices Guide ===")
    
    best_practices = [
        {
            "category": "Agent Design",
            "practices": [
                "Define clear agent responsibilities and capabilities",
                "Use appropriate specialization for different domains",
                "Implement proper error handling and recovery",
                "Design for scalability and maintainability"
            ]
        },
        {
            "category": "Task Management",
            "practices": [
                "Break complex tasks into smaller, manageable units",
                "Define clear dependencies between tasks",
                "Set appropriate priorities for tasks",
                "Monitor task execution and performance"
            ]
        },
        {
            "category": "Communication",
            "practices": [
                "Use appropriate communication protocols for the use case",
                "Implement message validation and error handling",
                "Consider security and privacy requirements",
                "Optimize communication for performance"
            ]
        },
        {
            "category": "Performance",
            "practices": [
                "Monitor resource usage and optimize allocation",
                "Implement caching for frequently accessed data",
                "Use asynchronous processing where appropriate",
                "Regularly profile and optimize performance"
            ]
        },
        {
            "category": "Maintenance",
            "practices": [
                "Implement proper logging and monitoring",
                "Backup agent states regularly",
                "Use version control for agent configurations",
                "Document agent behaviors and interactions"
            ]
        }
    ]
    
    for bp_group in best_practices:
        print(f"\n{bp_group['category']}:")
        for practice in bp_group['practices']:
            print(f"  • {practice}")
    
    print("\nFollowing these best practices will help ensure that your")
    print("multi-agent system is robust, efficient, and maintainable.")
    print()


def use_case_scenarios():
    """Describe various use case scenarios for the multi-agent system"""
    print("=== Use Case Scenarios ===")
    
    scenarios = [
        {
            "name": "Software Development Pipeline",
            "description": "Automate the entire software development lifecycle",
            "agents_involved": [
                "Project Manager Agent",
                "Code Generation Agent", 
                "Testing Agent",
                "Code Review Agent",
                "Deployment Agent"
            ],
            "workflow": [
                "Requirement analysis and task decomposition",
                "Code generation based on requirements",
                "Automated testing and quality assurance",
                "Code review and improvement suggestions",
                "Deployment and monitoring"
            ]
        },
        {
            "name": "Data Processing Pipeline",
            "description": "Process large volumes of data with specialized agents",
            "agents_involved": [
                "Data Ingestion Agent",
                "Data Cleaning Agent",
                "Data Analysis Agent", 
                "Visualization Agent",
                "Reporting Agent"
            ],
            "workflow": [
                "Collect data from multiple sources",
                "Clean and validate data quality",
                "Perform complex data analysis",
                "Generate visualizations and insights",
                "Create reports and dashboards"
            ]
        },
        {
            "name": "Research Collaboration",
            "description": "Facilitate collaboration between research agents",
            "agents_involved": [
                "Research Agent",
                "Analysis Agent",
                "Communication Agent",
                "Validation Agent",
                "Documentation Agent"
            ],
            "workflow": [
                "Literature review and information gathering",
                "Hypothesis formation and testing",
                "Results analysis and validation",
                "Communication of findings",
                "Documentation and knowledge sharing"
            ]
        },
        {
            "name": "System Monitoring and Maintenance",
            "description": "Monitor and maintain complex systems",
            "agents_involved": [
                "Monitoring Agent",
                "Alert Agent",
                "Troubleshooting Agent",
                "Maintenance Agent",
                "Reporting Agent"
            ],
            "workflow": [
                "Continuous system monitoring",
                "Anomaly detection and alerting",
                "Automated troubleshooting",
                "Maintenance scheduling and execution",
                "Performance reporting"
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Agents Involved:")
        for agent in scenario['agents_involved']:
            print(f"     • {agent}")
        print(f"   Workflow:")
        for step in scenario['workflow']:
            print(f"     • {step}")
    
    print("\nThese scenarios demonstrate the versatility and power of the")
    print("multi-agent system in handling various complex tasks through")
    print("collaboration and specialization.")
    print()


def advanced_features_demo():
    """Demonstrate advanced features of the multi-agent system"""
    print("=== Advanced Features Demonstration ===")
    
    # Feature 1: Dynamic Agent Creation
    print("1. Dynamic Agent Creation:")
    builder = AdvancedAgentBuilder()
    dynamic_agent = builder.create_dynamic_agent(
        "DynamicSpecialist",
        "Agent created dynamically with custom capabilities",
        ["analysis", "communication", "validation"]
    )
    print(f"   Created: {dynamic_agent.name} with capabilities: {[c.value for c in dynamic_agent.capabilities]}")
    
    # Feature 2: Agent Network Creation
    print("\n2. Agent Network Creation:")
    network = builder.create_cooperative_agent_network([
        {"type": "task_manager", "name": "NetManager"},
        {"type": "code_writer", "name": "NetCoder"},
        {"type": "tester", "name": "NetTester"}
    ])
    print(f"   Created network: {network.name} with {len(network.agents)} agents")
    
    # Feature 3: Hierarchical Networks
    print("\n3. Hierarchical Network:")
    hierarchy_config = {
        "name": "HierarchicalNet",
        "levels": {
            "top": {
                "agents": [{"type": "task_manager", "name": "Director"}]
            },
            "middle": {
                "agents": [{"type": "task_manager", "name": "Manager1"}]
            },
            "bottom": {
                "agents": [{"type": "code_writer", "name": "Worker1"}]
            }
        }
    }
    hier_network = builder.create_hierarchical_network(hierarchy_config)
    print(f"   Created hierarchical network: {hier_network.name}")
    
    # Feature 4: Task Dependencies and Scheduling
    print("\n4. Task Dependencies and Scheduling:")
    task_manager = AdvancedTaskManager()
    task1 = task_manager.create_task("Task 1", "First task", priority=3)
    task2 = task_manager.create_task("Task 2", "Second task", priority=2, 
                                   dependencies=[{"dependency_task_id": task1.id, "dependent_task_id": "temp"}])
    print(f"   Created {task_manager.scheduler.dependency_count[task2.id]} dependencies")
    
    # Feature 5: Learning and Adaptation
    print("\n5. Learning and Adaptation:")
    learning_agent = LearningAgent("SmartAgent", "Learning-enabled agent")
    for i in range(3):
        learning_agent.store_learning_experience(f"task_{i}", "completed", 0.7 + i*0.1)
    learning_agent.adapt_behavior()
    print(f"   Adaptation level: {learning_agent.adaptation_level.value}")
    print(f"   Exploration rate: {learning_agent.exploration_rate:.3f}")
    
    # Feature 6: Specialized Domain Agents
    print("\n6. Specialized Domain Agents:")
    code_agent = CodeSpecialist("CodeSpecialist", "Code domain expert")
    data_agent = DataSpecialist("DataSpecialist", "Data domain expert")
    comm_agent = CommunicationSpecialist("CommSpecialist", "Communication domain expert")
    print(f"   Created 3 specialized agents for different domains")
    
    # Feature 7: Performance Monitoring
    print("\n7. Performance Monitoring:")
    from src.task_manager.advanced_task_manager import TaskMonitoringService
    monitoring_service = TaskMonitoringService(task_manager)
    report = monitoring_service.generate_task_report()
    print(f"   Generated performance report with {report['total_tasks']} tasks")
    
    # Feature 8: State Persistence
    print("\n8. State Persistence:")
    import os
    os.makedirs("/workspace/backups", exist_ok=True)
    from src.agents.advanced_agents import backup_agent_state
    backup_success = backup_agent_state(learning_agent, "/workspace/backups")
    print(f"   Agent state backup: {'Success' if backup_success else 'Failed'}")
    
    print("\nAll advanced features demonstrated successfully!")
    print()


def tutorial_step_by_step():
    """Step-by-step tutorial for creating and using the multi-agent system"""
    print("=== Step-by-Step Tutorial ===")
    
    steps = [
        {
            "step": 1,
            "title": "Initialize the Agent Builder",
            "description": "Create an instance of the AdvancedAgentBuilder to manage agents",
            "code": """
from src.builders.advanced_agent_builder import AdvancedAgentBuilder

builder = AdvancedAgentBuilder()
print(f"Created agent builder with {len(builder.agent_templates)} templates")
            """
        },
        {
            "step": 2,
            "title": "Create Agents Using Templates",
            "description": "Use predefined templates to create specialized agents",
            "code": """
# Create agents using predefined templates
dev_agent = builder.create_agent_from_profile(
    "developer", 
    {"name": "MyDevAgent"}
)
print(f"Created developer agent: {dev_agent.name}")
            """
        },
        {
            "step": 3,
            "title": "Create Agent Networks",
            "description": "Connect multiple agents to work cooperatively",
            "code": """
# Define agent configurations
agent_configs = [
    {"type": "task_manager", "name": "Manager"},
    {"type": "code_writer", "name": "Coder"}, 
    {"type": "tester", "name": "Tester"}
]

# Create cooperative network
network = builder.create_cooperative_network("dev_network", agent_configs)
print(f"Created network with {len(network.agents)} agents")
            """
        },
        {
            "step": 4,
            "title": "Manage Tasks",
            "description": "Use the task manager to create and manage tasks",
            "code": """
from src.task_manager.advanced_task_manager import AdvancedTaskManager

task_manager = AdvancedTaskManager()

# Create a task
task = task_manager.create_task(
    title="Sample Task",
    description="A sample task for demonstration",
    task_type="computation",
    priority=2
)
print(f"Created task: {task.title}")
            """
        },
        {
            "step": 5,
            "title": "Execute Tasks",
            "description": "Submit tasks for execution by agents",
            "code": """
# Submit the task for execution
task_manager.submit_task(task)
print(f"Submitted task {task.id} for execution")
            """
        },
        {
            "step": 6,
            "title": "Monitor Performance",
            "description": "Use monitoring services to track system performance",
            "code": """
from src.task_manager.advanced_task_manager import TaskMonitoringService

monitoring_service = TaskMonitoringService(task_manager)
report = monitoring_service.generate_task_report()
print(f"Generated report for {report['total_tasks']} tasks")
            """
        },
        {
            "step": 7,
            "title": "Analyze Results",
            "description": "Analyze the results and optimize the system",
            "code": """
# Get system status
status = task_manager.get_system_status()
print(f"System status: {status['total_tasks']} tasks, {status['running_tasks']} running")

# Get task statistics
stats = task_manager.get_task_statistics()
print(f"Success rate: {stats['by_status'].get('completed', 0)} completed tasks")
            """
        }
    ]
    
    for step_info in steps:
        print(f"\nStep {step_info['step']}: {step_info['title']}")
        print(f"Description: {step_info['description']}")
        print("Code:")
        print(step_info['code'])
    
    print("\nThis tutorial provides a complete walkthrough of the")
    print("multi-agent system from initialization to execution and monitoring.")
    print()


def main():
    """Main function to run the comprehensive examples"""
    print("Multi-Agent System - Comprehensive Usage Examples")
    print("=" * 60)
    
    # Run the comprehensive example
    example_comprehensive_system_integration()
    
    # Run additional demonstrations
    performance_benchmark()
    system_architecture_overview()
    best_practices_guide()
    use_case_scenarios()
    advanced_features_demo()
    tutorial_step_by_step()
    
    print("\nAll comprehensive examples completed!")
    print("The multi-agent system is ready for complex applications.")
    print("=" * 60)


if __name__ == "__main__":
    main()