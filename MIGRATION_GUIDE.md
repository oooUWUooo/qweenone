# 🔄 Migration Guide: Legacy v1.0 → Modern v2.0

Complete guide for migrating from legacy Qweenone to modern architecture with minimal disruption.

---

## 📊 Migration Overview

### What's Changing

| Component | Legacy (v1.0) | Modern (v2.0) | Migration Difficulty |
|-----------|---------------|---------------|---------------------|
| Task Manager | AdvancedTaskManager | Prefect Workflows | 🟡 Medium |
| Task Decomposer | TaskDecomposer | ROMA Recursive Planner | 🟢 Easy |
| Desktop Automation | ❌ None | OmniParser + PyAutoGUI | 🟢 Easy (new feature) |
| Browser Automation | ❌ None | Playwright | 🟢 Easy (new feature) |
| API Router | Custom APIRouter | LiteLLM Router | 🟡 Medium |
| A2A Communication | In-Memory Manager | Redis/RabbitMQ Manager | 🟡 Medium |
| Main Entry Point | src/main.py | src/modern_main.py | 🟢 Easy |

---

## 🚦 Migration Strategies

### Strategy 1: Gradual Migration (Recommended)
**Best for:** Production systems, minimal risk

1. Install modern dependencies alongside legacy
2. Use modern components for new features
3. Gradually migrate existing workflows
4. Run both systems in parallel
5. Full cutover when ready

### Strategy 2: Complete Migration
**Best for:** New deployments, greenfield projects

1. Install modern dependencies
2. Update all imports and configurations
3. Test thoroughly
4. Deploy modern stack

### Strategy 3: Hybrid Approach
**Best for:** Complex systems with specific constraints

1. Identify components to modernize
2. Keep legacy for critical paths
3. Use modern for new capabilities
4. Migrate incrementally

---

## 📝 Step-by-Step Migration

### Phase 1: Setup & Dependencies

#### 1.1 Install Modern Dependencies

```bash
# Backup current dependencies
cp requirements.txt requirements.backup.txt

# Install modern dependencies
pip install -r requirements_modern.txt

# Install Playwright browsers
playwright install chromium

# Verify installations
python -c "import prefect; print(f'Prefect: {prefect.__version__}')"
python -c "import playwright; print('Playwright: OK')"
python -c "import litellm; print('LiteLLM: OK')"
```

#### 1.2 Setup Infrastructure Services

```bash
# Start Redis (for A2A and caching)
docker run -d --name qweenone-redis -p 6379:6379 redis:7-alpine

# Start RabbitMQ (for enterprise messaging)
docker run -d --name qweenone-rabbitmq \
  -p 5672:5672 -p 15672:15672 \
  -e RABBITMQ_DEFAULT_USER=qweenone \
  -e RABBITMQ_DEFAULT_PASS=qweenone_secret \
  rabbitmq:3-management-alpine

# Start PostgreSQL (for Prefect)
docker run -d --name qweenone-postgres \
  -p 5432:5432 \
  -e POSTGRES_USER=prefect \
  -e POSTGRES_PASSWORD=prefect_secret \
  -e POSTGRES_DB=prefect \
  postgres:15-alpine

# Or use docker-compose
docker-compose -f docker-compose.modern.yml up -d redis rabbitmq postgres
```

#### 1.3 Configure Environment Variables

Create `.env` file:

```bash
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
OPENROUTER_API_KEY=your_openrouter_key

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# RabbitMQ
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=qweenone
RABBITMQ_PASS=qweenone_secret

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=prefect
POSTGRES_USER=prefect
POSTGRES_PASSWORD=prefect_secret

# Feature Flags
USE_PREFECT=true
USE_ROMA=true
ENABLE_DESKTOP_AUTOMATION=false
ENABLE_BROWSER_AUTOMATION=true
ENABLE_LITELLM=true
A2A_BACKEND=redis
```

---

### Phase 2: Code Migration

#### 2.1 Task Manager → Prefect Workflows

**Before (Legacy):**
```python
from src.task_manager.advanced_task_manager import AdvancedTaskManager, Task, TaskType

manager = AdvancedTaskManager()

task = manager.create_task(
    title="Process Data",
    description="Process incoming data",
    task_type=TaskType.DATA_PROCESSING
)

manager.submit_task(task)
```

**After (Modern):**
```python
from src.workflow_engine.prefect_manager import PrefectWorkflowManager, ModernTask

manager = PrefectWorkflowManager()

task = ModernTask(
    id="process_data",
    title="Process Data",
    description="Process incoming data",
    agent_type="data"
)

result = await manager.execute_workflow([task])
```

**Migration Notes:**
- Replace `create_task()` with `ModernTask()` constructor
- Use `execute_workflow()` instead of `submit_task()`
- Tasks now execute asynchronously
- Built-in retry and monitoring

---

#### 2.2 TaskDecomposer → ROMA

**Before (Legacy):**
```python
from src.task_manager.task_decomposer import TaskDecomposer

decomposer = TaskDecomposer()
plan = decomposer.decompose("Create API server", iterations=3)
```

**After (Modern):**
```python
from src.task_decomposition.roma_decomposer import ROMAAugmentedTaskDecomposer

decomposer = ROMAAugmentedTaskDecomposer()
plan = await decomposer.decompose_with_roma(
    "Create API server",
    iterations=3,
    use_recursive=True
)

# Access ROMA enhancements
roma_data = plan["roma_enhanced"]["recursive_plan"]
print(f"Automation potential: {roma_data['automation_score']:.1%}")
```

**Migration Notes:**
- Replace sync `decompose()` with async `decompose_with_roma()`
- Access enhanced data through `roma_enhanced` key
- Benefits: deeper decomposition, automation scoring, confidence metrics

---

#### 2.3 API Router → LiteLLM

**Before (Legacy):**
```python
from src.api_router.api_router import APIRouter
from src.api_integrations import APIConfig, ProviderType

router = APIRouter()
router.register_provider(APIConfig(provider=ProviderType.OPENAI))

response = await router.route_request(
    messages=[{"role": "user", "content": "Hello"}]
)
```

**After (Modern):**
```python
from src.api_router.litellm_router import LiteLLMUnifiedRouter

router = LiteLLMUnifiedRouter()

result = await router.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-3.5-turbo"
)

print(f"Response: {result['content']}")
print(f"Cost: ${result['cost']:.6f}")
```

**Migration Notes:**
- LiteLLM handles provider registration automatically
- Model names are universal (no provider-specific config)
- Automatic cost tracking and retry logic
- Supports streaming: use `stream_completion()` method

---

#### 2.4 A2A Communication → Modern Backend

**Before (Legacy):**
```python
from src.communication.a2a_manager import A2ACommunicationManager, Message

manager = A2ACommunicationManager()

await manager.register_agent("agent_1")
await manager.send_message(message)
```

**After (Modern):**
```python
from src.communication.modern_a2a_manager import ModernA2ACommunicationManager, A2AConfig, CommunicationBackend

config = A2AConfig(backend=CommunicationBackend.REDIS)
manager = ModernA2ACommunicationManager(config)

await manager.initialize()
await manager.register_agent("agent_1")
await manager.send_message(message)
```

**Migration Notes:**
- Must call `initialize()` before use
- Choose backend: MEMORY (legacy compatible), REDIS, or RABBITMQ
- Messages are automatically persisted with Redis/RabbitMQ
- Backward compatible with legacy Message format

---

#### 2.5 Main System → ModernAgenticSystem

**Before (Legacy):**
```python
from src.main import AgenticSystem

system = AgenticSystem()
plan = system.decompose_task("Create parser", 3)
result = await system.execute_task_plan(plan)
```

**After (Modern):**
```python
from src.modern_main import ModernAgenticSystem

system = ModernAgenticSystem(
    use_prefect=True,
    use_roma=True,
    enable_browser_automation=True,
    a2a_backend="redis"
)

await system.initialize()
result = await system.full_pipeline("Create parser", iterations=3)
await system.close()
```

**Migration Notes:**
- Feature flags allow gradual adoption
- Single `full_pipeline()` call replaces multiple steps
- Must call `initialize()` and `close()`
- Provides unified interface to all modern components

---

### Phase 3: New Capabilities

#### 3.1 Desktop Automation (New)

```python
from src.modern_main import ModernAgenticSystem

system = ModernAgenticSystem(enable_desktop_automation=True)
await system.initialize()

result = await system.execute_desktop_automation(
    "Open calculator and compute 42 * 13"
)

print(f"Success: {result['overall_success']}")
await system.close()
```

#### 3.2 Browser Automation (New)

```python
from src.modern_main import ModernAgenticSystem

system = ModernAgenticSystem(enable_browser_automation=True)
await system.initialize()

result = await system.execute_browser_automation(
    "Navigate to GitHub and search for Python automation tools"
)

print(f"Success: {result['overall_success']}")
print(f"Final URL: {result.get('current_url')}")
await system.close()
```

---

## ✅ Migration Checklist

### Pre-Migration
- [ ] Backup current codebase and data
- [ ] Review current system dependencies
- [ ] Document custom configurations
- [ ] Test current system functionality
- [ ] Set up staging environment

### Infrastructure Setup
- [ ] Install modern Python dependencies
- [ ] Setup Redis (if using Redis backend)
- [ ] Setup RabbitMQ (if using RabbitMQ backend)
- [ ] Setup PostgreSQL for Prefect
- [ ] Configure environment variables
- [ ] Install Playwright browsers

### Code Migration
- [ ] Update task manager calls to Prefect
- [ ] Update task decomposer to ROMA
- [ ] Update API router to LiteLLM
- [ ] Update A2A manager to modern backend
- [ ] Update main entry point
- [ ] Add async/await where needed

### Testing & Validation
- [ ] Run modern integration tests
- [ ] Test each component individually
- [ ] Test full pipeline end-to-end
- [ ] Verify backward compatibility
- [ ] Performance testing
- [ ] Load testing (if applicable)

### Deployment
- [ ] Update Docker configurations
- [ ] Update docker-compose files
- [ ] Deploy to staging
- [ ] Monitor for issues
- [ ] Deploy to production
- [ ] Monitor metrics

### Post-Migration
- [ ] Update documentation
- [ ] Train team on new components
- [ ] Set up monitoring dashboards
- [ ] Plan for legacy code removal
- [ ] Document lessons learned

---

## 🐛 Common Migration Issues

### Issue 1: Async/Await Required

**Problem:** Modern components require async/await
```python
# This will fail
result = decomposer.decompose_with_roma(task)
```

**Solution:** Use asyncio
```python
# This works
result = await decomposer.decompose_with_roma(task)

# Or if calling from sync code:
result = asyncio.run(decomposer.decompose_with_roma(task))
```

---

### Issue 2: Redis Connection Failed

**Problem:** A2A manager can't connect to Redis

**Solution:** Check Redis is running and accessible
```bash
# Test connection
redis-cli ping

# Check connection from Python
python -c "import redis; r=redis.Redis(); print(r.ping())"

# Fallback to memory backend if needed
config = A2AConfig(backend=CommunicationBackend.MEMORY)
```

---

### Issue 3: Prefect Database Not Found

**Problem:** Prefect can't connect to database

**Solution:** Initialize Prefect database
```bash
# Set database URL
export PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://prefect:prefect_secret@localhost:5432/prefect"

# Create database
prefect database reset -y
```

---

### Issue 4: LiteLLM API Keys

**Problem:** LiteLLM can't authenticate with providers

**Solution:** Set API keys in environment
```bash
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"

# Or in code
import os
os.environ["OPENAI_API_KEY"] = "your_key"
```

---

### Issue 5: Playwright Browsers Not Installed

**Problem:** Browser automation fails with "browser not found"

**Solution:** Install browsers
```bash
# Install all browsers
playwright install

# Install specific browser
playwright install chromium

# Install with system dependencies
playwright install --with-deps chromium
```

---

## 📈 Performance Optimization

### Optimize Prefect Workflows

```python
# Use concurrent execution for independent tasks
manager = PrefectWorkflowManager(concurrent_limit=10)

# Enable caching for repeated tasks
@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
async def cached_task(params):
    ...
```

### Optimize LiteLLM Routing

```python
# Enable Redis caching for LiteLLM
config = LiteLLMConfig(
    enable_caching=True,
    redis_host="localhost",
    redis_port=6379
)

router = LiteLLMUnifiedRouter(config)
```

### Optimize A2A Communication

```python
# Use Redis for fast distributed messaging
config = A2AConfig(
    backend=CommunicationBackend.REDIS,
    redis_host="localhost",
    message_ttl=3600
)

# Or RabbitMQ for guaranteed delivery
config = A2AConfig(
    backend=CommunicationBackend.RABBITMQ,
    rabbitmq_host="localhost"
)
```

---

## 🔄 Backward Compatibility

### Using Legacy Components in Modern System

```python
from src.modern_main import ModernAgenticSystem

# Disable modern features as needed
system = ModernAgenticSystem(
    use_prefect=False,      # Use legacy task manager
    use_roma=False,         # Use legacy decomposer
    enable_litellm=False,   # Use legacy API router
    a2a_backend="memory"    # Use legacy in-memory A2A
)
```

### Using Modern Components in Legacy System

```python
from src.main import AgenticSystem
from src.task_decomposition.roma_decomposer import ROMAAugmentedTaskDecomposer

system = AgenticSystem()

# Replace legacy decomposer with ROMA
system.task_decomposer = ROMAAugmentedTaskDecomposer()

# Use modern decomposition
plan = await system.task_decomposer.decompose_with_roma(
    task_description="Create parser",
    iterations=3,
    use_recursive=False  # Disable recursion for legacy compatibility
)
```

---

## 🧪 Testing Migration

### Test Modern Components Individually

```bash
# Test Prefect workflow manager
python -m pytest tests/test_modern_integration.py::TestPrefectWorkflowManager -v

# Test ROMA decomposer
python -m pytest tests/test_modern_integration.py::TestROMATaskDecomposer -v

# Test A2A communication
python -m pytest tests/test_modern_integration.py::TestModernA2ACommunication -v

# Test full integration
python -m pytest tests/test_modern_integration.py::TestModernAgenticSystem -v
```

### Compare Legacy vs Modern Performance

```python
import asyncio
import time

from src.main import AgenticSystem
from src.modern_main import ModernAgenticSystem

async def benchmark():
    task = "Create a simple calculator script"
    
    # Legacy system
    legacy_system = AgenticSystem()
    start = time.time()
    legacy_result = await legacy_system.run_full_pipeline(task, 2)
    legacy_time = time.time() - start
    
    # Modern system
    modern_system = ModernAgenticSystem()
    await modern_system.initialize()
    start = time.time()
    modern_result = await modern_system.full_pipeline(task, 2)
    modern_time = time.time() - start
    await modern_system.close()
    
    print(f"Legacy: {legacy_time:.2f}s")
    print(f"Modern: {modern_time:.2f}s")
    print(f"Improvement: {((legacy_time - modern_time) / legacy_time * 100):.1f}%")

asyncio.run(benchmark())
```

---

## 🎯 Migration Timeline

### Week 1: Setup & Testing
- Day 1-2: Install dependencies and infrastructure
- Day 3-4: Run integration tests
- Day 5-7: Test in staging environment

### Week 2: Component Migration
- Day 8-9: Migrate task decomposition to ROMA
- Day 10-11: Migrate API routing to LiteLLM
- Day 12-14: Migrate A2A to Redis/RabbitMQ

### Week 3: Workflow Migration
- Day 15-17: Migrate workflows to Prefect
- Day 18-19: Add automation capabilities
- Day 20-21: End-to-end testing

### Week 4: Deployment & Monitoring
- Day 22-24: Deploy to production
- Day 25-26: Monitor and optimize
- Day 27-28: Documentation and training

---

## 📚 Additional Resources

- [Modern Architecture Documentation](./MODERN_ARCHITECTURE.md)
- [Modernization Summary](./MODERNIZATION_SUMMARY.md)
- [Prefect Documentation](https://docs.prefect.io/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Playwright Python Docs](https://playwright.dev/python/)

---

## ❓ FAQ

### Q: Can I run both legacy and modern systems simultaneously?
**A:** Yes! They use different entry points (`main.py` vs `modern_main.py`) and can run in parallel.

### Q: Do I need all infrastructure services (Redis, RabbitMQ, PostgreSQL)?
**A:** No. You can use:
- **Memory backend** for A2A (no Redis/RabbitMQ needed)
- **Prefect without persistence** (no PostgreSQL needed)
- Configure via feature flags

### Q: What if I don't have API keys for LiteLLM?
**A:** LiteLLM will gracefully fail for unavailable providers. You can:
- Use only providers you have keys for
- Disable LiteLLM: `enable_litellm=False`
- Use mock/local models

### Q: Will my existing agents still work?
**A:** Yes! Modern system is backward compatible with legacy agent types.

### Q: How do I rollback if migration fails?
**A:** 
1. Keep legacy code intact
2. Use git branches for migration work
3. Modern components are additive (don't delete legacy)
4. Can switch back to `main.py` anytime

---

## 🆘 Support

For issues or questions:
1. Check logs in `/app/logs`
2. Review test output: `pytest tests/test_modern_integration.py -v`
3. Check service status: `docker-compose -f docker-compose.modern.yml ps`
4. Review component documentation in source files

---

**Good luck with your migration! 🚀**
