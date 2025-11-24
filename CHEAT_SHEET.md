# 📋 Qweenone v2.0 - Cheat Sheet

Quick reference for common operations.

---

## 🚀 Installation

```bash
pip install -r requirements_modern.txt
playwright install chromium
```

---

## 🎯 Common Commands

### System Operations
```bash
# View status
python src/modern_main.py --status

# Run demo
python src/modern_main.py --demo

# Execute task
python src/modern_main.py --task "Your task here"

# Run tests
pytest tests/test_modern_integration.py -v
```

### Docker Operations
```bash
# Start all services
docker-compose -f docker-compose.modern.yml up -d

# View logs
docker-compose -f docker-compose.modern.yml logs -f

# Stop all
docker-compose -f docker-compose.modern.yml down

# Restart service
docker-compose -f docker-compose.modern.yml restart qweenone-modern
```

---

## 📦 Imports

### Workflow Orchestration
```python
from src.workflow_engine.prefect_manager import PrefectWorkflowManager, ModernTask
```

### Task Decomposition
```python
from src.task_decomposition.roma_decomposer import ROMAAugmentedTaskDecomposer
```

### Desktop Automation
```python
from src.desktop_automation.omni_automation import OmniDesktopAutomation
```

### Browser Automation
```python
from src.browser_automation.playwright_automation import PlaywrightAutomation, BrowserConfig
```

### LLM Routing
```python
from src.api_router.litellm_router import LiteLLMUnifiedRouter
```

### A2A Communication
```python
from src.communication.modern_a2a_manager import ModernA2ACommunicationManager, A2AConfig
```

### Full System
```python
from src.modern_main import ModernAgenticSystem
```

---

## 💡 Code Snippets

### Create & Execute Workflow
```python
manager = PrefectWorkflowManager()
tasks = [ModernTask(id="1", title="Task 1", description="Do something")]
result = await manager.execute_workflow(tasks)
```

### Decompose Task
```python
decomposer = ROMAAugmentedTaskDecomposer()
plan = await decomposer.decompose_with_roma("Create API", iterations=3)
print(f"Subtasks: {plan['total_subtasks']}")
```

### Browser Automation
```python
async with PlaywrightAutomation(BrowserConfig(headless=True)) as browser:
    result = await browser.execute_natural_language_task("Go to Google")
```

### LLM Query
```python
router = LiteLLMUnifiedRouter()
result = await router.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-3.5-turbo"
)
```

### Full Pipeline
```python
system = ModernAgenticSystem()
await system.initialize()
result = await system.full_pipeline("Create scraper", iterations=3)
await system.close()
```

---

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
REDIS_HOST=localhost
RABBITMQ_HOST=localhost
USE_PREFECT=true
A2A_BACKEND=redis
```

### Feature Flags
```python
ModernAgenticSystem(
    use_prefect=True,              # Use Prefect workflows
    use_roma=True,                 # Use ROMA decomposition
    enable_desktop_automation=True, # Enable desktop automation
    enable_browser_automation=True, # Enable browser automation
    enable_litellm=True,           # Use LiteLLM routing
    a2a_backend="redis"            # Use Redis for A2A
)
```

---

## 🎨 Deployment Patterns

### Development
```bash
python src/modern_main.py --task "..." --a2a-backend memory
```

### Staging
```bash
docker-compose -f docker-compose.modern.yml up -d
```

### Production
```bash
docker stack deploy -c docker-compose.modern.yml qweenone
```

---

## 📊 URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Prefect UI | http://localhost:4200 | None |
| RabbitMQ Management | http://localhost:15672 | qweenone/qweenone_secret |
| Prometheus | http://localhost:9090 | None |
| Grafana | http://localhost:3000 | admin/admin |

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | `pip install -r requirements_modern.txt` |
| Playwright error | `playwright install chromium` |
| Redis connection failed | `docker run -d -p 6379:6379 redis:7-alpine` |
| No API key | `export OPENAI_API_KEY="..."` or use `--no-litellm` |
| Async error | Wrap in `asyncio.run(...)` |

---

## 📄 File Locations

| What | Where |
|------|-------|
| Modern entry point | `src/modern_main.py` |
| Legacy entry point | `src/main.py` |
| Full demo | `demo_modern_full.py` |
| Tests | `tests/test_modern_integration.py` |
| Docker Compose | `docker-compose.modern.yml` |
| Architecture docs | `MODERN_ARCHITECTURE.md` |

---

## 🎓 Learn More

- Quick Start: [QUICKSTART.md](./QUICKSTART.md)
- Full Docs: [MODERN_ARCHITECTURE.md](./MODERN_ARCHITECTURE.md)
- Migration: [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)
- Deployment: [DEPLOYMENT.md](./DEPLOYMENT.md)

---

**Keep this cheat sheet handy! 📌**
