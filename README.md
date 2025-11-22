# Agentic Architecture System

A comprehensive, production-ready agentic system with multi-agent communication, task decomposition, and iterative development capabilities.

## Features

- **Multi-Agent Architecture**: Dynamic agent creation and management
- **A2A Communication**: Agent-to-agent messaging with individual queues
- **Task Decomposition**: 3-iteration development process (basic → enhanced → advanced)
- **Recursive Task Breakdown**: Decompose tasks to primitive operations
- **Testing Framework**: Integrated testing with pytest/unittest support
- **Containerization**: Docker and Docker Compose support
- **Monitoring**: Prometheus integration for metrics
- **Documentation**: Scalar-based API documentation

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Agentic       │    │   A2A           │    │   Task          │
│   System        │    │   Manager       │    │   Decomposer    │
│                 │    │                 │    │                 │
│ • Agent         │    │ • Message       │    │ • Task          │
│   Management    │    │   Queues        │    │   Planning      │
│ • Task          │◄──►│ • Communication │◄──►│ • Iterative     │
│   Execution     │    │   Protocols     │    │   Development   │
│ • System        │    │ • Heartbeat     │    │   Process       │
│   Monitoring    │    │   Management    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Components

### Core Components

- **AgenticSystem**: Main orchestrator managing agents, tasks, and execution
- **BaseAgent**: Abstract base class for all agent types
- **A2ACommunicationManager**: Handles agent-to-agent communication
- **TaskDecomposer**: Breaks down tasks into iterative subtasks
- **TestRunner**: Executes tests with multiple frameworks
- **AgentBuilder**: Creates specialized agents based on requirements

### Agent Types

- **CodeAgent**: Specialized in code generation and review
- **TestingAgent**: Executes and validates tests
- **TaskAgent**: Manages and decomposes complex tasks
- **CommunicationAgent**: Handles inter-agent communication

## Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd agentic-architecture

# Install dependencies
pip install -r requirements.txt

# Run the system
python src/main.py --task "Create a web server" --iterations 3
```

### Docker Deployment

```bash
# Build and start services
docker-compose up --build

# Run a specific task
docker-compose exec agentic-system python src/main.py --task "Create a web server" --iterations 3

# View system status
docker-compose exec agentic-system python src/main.py --status
```

## Usage

### Command Line Interface

```bash
# Run a task with 3 iterations
python src/main.py --task "Build an Instagram parser" --iterations 3

# Interactive mode
python src/main.py --interactive

# View system status
python src/main.py --status
```

### Example Tasks

#### Server Development
```bash
python src/main.py --task "Create a web server" --iterations 3
```
- Iteration 1: Basic server with core functionality
- Iteration 2: Dockerized server with containerization
- Iteration 3: Multi-component architecture with server-to-server communication

#### Parser Development
```bash
python src/main.py --task "Build an Instagram parser" --iterations 3
```
- Iteration 1: Basic parsing module with core logic
- Iteration 2: Authentication module with security
- Iteration 3: Integrated system with monitoring

### API Documentation

Documentation is available via Scalar at `http://localhost:5050` when running with Docker Compose.

## Development

### Adding New Agent Types

1. Create a new agent class inheriting from `BaseAgent`
2. Register the agent type in the `AgentBuilder`
3. Implement the `execute_task` method with specific logic

### Task Decomposition

The system supports recursive task decomposition. Complex tasks are automatically broken down into:
- High-level iterations (3 iterations by default)
- Subtasks within each iteration
- Primitive operations at the lowest level

### Communication Patterns

Agents communicate using the A2A protocol with:
- Individual message queues per agent
- Request/Response patterns
- Broadcast messaging
- Heartbeat monitoring
- Message history tracking

## Configuration

### Environment Variables

- `LOG_LEVEL`: Set logging level (default: INFO)
- `PYTHONPATH`: Python module search path

### Docker Compose Services

- `agentic-system`: Main application service
- `redis`: In-memory data store for caching and session management
- `postgres`: Persistent storage for agent states and task history
- `scalar`: API documentation service
- `prometheus`: Metrics collection and monitoring

## Monitoring and Observability

- **Metrics**: Prometheus endpoint at `http://localhost:9090`
- **Logging**: Structured logs available through Docker logs
- **Communication Stats**: Available via `--status` command
- **Message History**: Tracked and accessible through A2A manager

## Testing

The system supports multiple testing frameworks:

```bash
# Run tests with pytest
python src/main.py --task "Test system components" --iterations 1

# Direct test execution
python -m pytest tests/
python -m unittest discover tests/
```

## Security

- Containerized execution with non-root users
- Isolated agent communication channels
- Input validation for tasks and messages
- Environment-based configuration

## Scaling

The architecture supports horizontal scaling through:
- Independent agent containers
- Distributed message queues
- Shared Redis and PostgreSQL backends
- Load balancing capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License.