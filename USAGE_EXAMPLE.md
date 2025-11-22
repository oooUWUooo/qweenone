# Usage Examples for Agentic Architecture System

## Quick Start

Here's how to use the Agentic Architecture System:

### Basic Task Decomposition
```python
from src.main import AgenticSystem

# Create the system
system = AgenticSystem()

# Decompose a task into 3 iterations
task_plan = system.decompose_task("Write a web server", iterations=3)

print(f"Task: {task_plan['original_task']}")
print(f"Iterations: {len(task_plan['iterations'])}")
print(f"Total subtasks: {task_plan['total_subtasks']}")
```

### Creating Specialized Agents
```python
# Create a code writing agent
code_agent = system.build_agent({
    "type": "code_writer",
    "name": "ServerDeveloper",
    "description": "Specialized in server development"
})

# Create a testing agent
tester_agent = system.build_agent({
    "type": "tester", 
    "name": "ServerTester",
    "description": "Specialized in testing server code"
})
```

### Executing Tasks
```python
import asyncio

async def execute_task_example():
    system = AgenticSystem()
    
    # Create agents
    agent = system.build_agent({
        "type": "code_writer",
        "name": "TaskExecutor",
        "description": "General purpose task executor"
    })
    
    # Create a task
    task = {
        "id": "example_task_1",
        "title": "Write basic server",
        "description": "Implement a basic HTTP server",
        "agent_type": "code_writer"
    }
    
    # Execute the task
    result = await agent.execute_task(task)
    print(f"Task result: {result}")

# Run the example
asyncio.run(execute_task_example())
```

## Complete Example: Building an Instagram Parser

```python
import asyncio
from src.main import AgenticSystem

async def build_instagram_parser_example():
    # Create the system
    system = AgenticSystem()
    
    # Define the main task
    main_task = "Write an Instagram parser"
    
    # Decompose into 3 iterations
    plan = system.decompose_task(main_task, iterations=3)
    
    print(f"Task: {main_task}")
    print(f"Type: {plan['task_type']}")
    
    # Create specialized agents
    parser_dev = system.build_agent({
        "type": "code_writer",
        "name": "ParserDeveloper",
        "description": "Specialized in parsing development"
    })
    
    auth_dev = system.build_agent({
        "type": "code_writer", 
        "name": "AuthDeveloper",
        "description": "Specialized in authentication systems"
    })
    
    integrator = system.build_agent({
        "type": "code_writer",
        "name": "SystemIntegrator", 
        "description": "Specialized in system integration"
    })
    
    tester = system.build_agent({
        "type": "tester",
        "name": "IntegrationTester",
        "description": "Specialized in integration testing"
    })
    
    # Execute each iteration
    agents = [parser_dev, auth_dev, integrator]
    
    for i, iteration in enumerate(plan['iterations']):
        print(f"\nExecuting: {iteration['description']}")
        
        current_agent = agents[i]
        
        for j, subtask in enumerate(iteration['subtasks']):
            task = {
                "title": subtask['title'],
                "description": subtask['description'],
                "agent_type": current_agent.agent_type
            }
            
            result = await current_agent.execute_task(task)
            print(f"  ✓ {subtask['title']}")
    
    # Test the complete system
    test_result = await tester.execute_task({
        "title": "Test Complete Instagram Parser",
        "description": "Test the fully integrated Instagram parser"
    })
    
    print(f"\nFinal test result: {test_result['result']}")

# Run the complete example
asyncio.run(build_instagram_parser_example())
```

## Available Agent Types

- `code_writer`: Specialized in writing and reviewing code
- `tester`: Specialized in testing code and validating solutions  
- `task_manager`: Specialized in managing and decomposing tasks
- `communication`: Specialized in agent-to-agent communication

## Communication System

```python
from src.communication.a2a_manager import Message, MessageType

# Send a message between agents
message = Message(
    sender_id=agent1.id,
    receiver_id=agent2.id,
    msg_type=MessageType.REQUEST,
    content={"request": "help with parsing", "context": "instagram project"}
)

success = await system.a2a_manager.send_message(message)
```

## Task Board Generation

```python
# Generate a task board from decomposition plan
task_board = system.task_decomposer.create_task_board(plan)

print(f"Created task board with {len(task_board)} tasks")
for task in task_board[:5]:  # Show first 5 tasks
    print(f"- {task.title}")
```

The system provides a complete framework for building and managing autonomous agents that can collaborate on complex projects with iterative development, task decomposition, and communication capabilities.