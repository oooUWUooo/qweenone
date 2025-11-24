# Qweenone - Modern Agentic Architecture System v2.0

A comprehensive, production-ready agentic system with **modern frameworks integration**: Prefect workflows, ROMA recursive task decomposition, OmniParser desktop automation, Playwright browser control, LiteLLM universal API routing, and enterprise A2A communication.

> **🆕 NEW:** Fully modernized with industry-leading tools - see [MODERN_ARCHITECTURE.md](./MODERN_ARCHITECTURE.md) for details

## ✨ Features

### Modern v2.0 Features
- ⚡ **Prefect Workflow Orchestration**: Production-ready task management with auto-retries
- 🧠 **ROMA Recursive Decomposition**: AI-guided task breakdown up to 5 levels deep
- 👁️ **OmniParser Desktop Automation**: Vision-based GUI control with natural language
- 🌐 **Playwright Browser Automation**: Modern web automation with auto-waiting
- 🔀 **LiteLLM Universal Router**: Access 100+ LLM providers through one API
- 📡 **Redis/RabbitMQ A2A Communication**: Enterprise messaging with persistence

### Core Features (v1.0)
- **Multi-Agent Architecture**: Dynamic agent creation and management
- **A2A Communication**: Agent-to-agent messaging with individual queues
- **Task Decomposition**: Iterative development process
- **Testing Framework**: Integrated testing with pytest/unittest support
- **Containerization**: Docker and Docker Compose support
- **Monitoring**: Prometheus + Grafana integration

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

### Quick Start (Modern v2.0)

```bash
# Clone the repository
git clone <repository-url>
cd qweenone

# Install modern dependencies
pip install -r requirements_modern.txt

# Install Playwright browsers
playwright install chromium

# Run the modern system
python src/modern_main.py --task "Create a web scraper" --iterations 3

# View system status
python src/modern_main.py --status
```

### Legacy v1.0

```bash
# Install legacy dependencies
pip install -r requirements.txt

# Run legacy system
python src/main.py --task "Create a web server" --iterations 3
```

### Docker Deployment (Modern)

```bash
# Build and start modern stack with all services
docker-compose -f docker-compose.modern.yml up --build

# Access services:
# - Prefect UI: http://localhost:4200
# - RabbitMQ Management: http://localhost:15672
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000

# Run a task in container
docker-compose -f docker-compose.modern.yml exec qweenone-modern \
  python src/modern_main.py --task "Create a web server"
```

### Legacy Docker Deployment

```bash
# Use legacy docker-compose
docker-compose up --build
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












```
Исследовать ключевые инструменты для интеграции 

Спроектировать новую архитектуру с готовыми компонентами 

Заменить AdvancedTaskManager на Prefect 

Заменить TaskDecomposer на AgentOrchestra/ROMA

Добавить OmniParser + PyAutoGUI для desktop automation

Интегрировать Playwright для web automation

Модернизировать систему агентов с использованием найденных фреймворков

Упростить API роутинг с LiteLLM  <- now

Улучшить A2A коммуникации с современными инструментами

Добавить тестирование новых компонентов

Обновить развертывание без Kubernetes

Обновить документацию для новой архитектуры
```
