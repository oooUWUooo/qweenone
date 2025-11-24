# ⚡ Qweenone v2.0 - 5-Minute Quickstart

Get up and running with modern Qweenone in 5 minutes!

---

## 🚀 Option 1: Run Full Demo (Fastest)

```bash
# 1. Install dependencies (2 min)
pip install -r requirements_modern.txt
playwright install chromium

# 2. Run comprehensive demo (2 min)
python demo_modern_full.py

# 3. Done! ✅
```

This will show you ALL modern features in action.

---

## 🐳 Option 2: Docker Deployment (Production-Like)

```bash
# 1. Create environment file (30 sec)
cat > .env << EOF
OPENAI_API_KEY=your_key_here
USE_PREFECT=true
USE_ROMA=true
A2A_BACKEND=redis
EOF

# 2. Start all services (1 min)
docker-compose -f docker-compose.modern.yml up -d

# 3. Check status (30 sec)
docker-compose -f docker-compose.modern.yml exec qweenone-modern \
  python src/modern_main.py --status

# 4. Access UIs
# - Prefect: http://localhost:4200
# - RabbitMQ: http://localhost:15672 (qweenone/qweenone_secret)
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

---

## 💻 Option 3: Local Development

```bash
# 1. Install (2 min)
pip install -r requirements_modern.txt
playwright install chromium

# 2. Start infrastructure (optional, 1 min)
docker run -d --name qweenone-redis -p 6379:6379 redis:7-alpine

# 3. Run a task (1 min)
python src/modern_main.py --task "Create a simple Python calculator"

# 4. View results
python src/modern_main.py --status
```

---

## 🎯 Quick Examples

### Task Decomposition

```python
from src.task_decomposition.roma_decomposer import ROMAAugmentedTaskDecomposer
import asyncio

async def decompose():
    decomposer = ROMAAugmentedTaskDecomposer()
    
    plan = await decomposer.decompose_with_roma(
        "Create Instagram scraper",
        iterations=3,
        use_recursive=True
    )
    
    print(f"Subtasks: {plan['total_subtasks']}")
    
    roma = plan["roma_enhanced"]["recursive_plan"]
    print(f"Automation: {roma['automation_score']:.1%}")
    print(f"Agents needed: {', '.join(roma['required_agents'])}")

asyncio.run(decompose())
```

### Browser Automation

```python
from src.browser_automation.playwright_automation import PlaywrightAutomation, BrowserConfig
import asyncio

async def browse():
    config = BrowserConfig(headless=True)
    
    async with PlaywrightAutomation(config) as browser:
        result = await browser.execute_natural_language_task(
            "Go to example.com and take a screenshot"
        )
        
        print(f"Success: {result['overall_success']}")
        print(f"URL: {result.get('current_url')}")

asyncio.run(browse())
```

### LLM Routing

```python
from src.api_router.litellm_router import LiteLLMUnifiedRouter
import asyncio

async def query_llm():
    router = LiteLLMUnifiedRouter()
    
    result = await router.chat_completion(
        messages=[{"role": "user", "content": "Explain AI in one sentence"}],
        model="gpt-3.5-turbo"
    )
    
    print(f"Response: {result['content']}")
    print(f"Cost: ${result['cost']:.6f}")

asyncio.run(query_llm())
```

### Full Pipeline

```python
from src.modern_main import ModernAgenticSystem
import asyncio

async def full_pipeline():
    system = ModernAgenticSystem(
        use_prefect=True,
        use_roma=True,
        a2a_backend="memory"
    )
    
    await system.initialize()
    
    result = await system.full_pipeline(
        task_description="Create a web scraper",
        iterations=2
    )
    
    print(f"Status: {result['status']}")
    
    await system.close()

asyncio.run(full_pipeline())
```

---

## 🧪 Run Tests

```bash
# All tests
pytest tests/test_modern_integration.py -v

# Specific component
pytest tests/test_modern_integration.py::TestROMATaskDecomposer -v

# With output
pytest tests/test_modern_integration.py -v -s
```

---

## 📚 Learn More

- **Architecture:** [MODERN_ARCHITECTURE.md](./MODERN_ARCHITECTURE.md)
- **Migration:** [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)
- **Deployment:** [DEPLOYMENT.md](./DEPLOYMENT.md)
- **Full Summary:** [INTEGRATION_COMPLETE.md](./INTEGRATION_COMPLETE.md)

---

## 🆘 Common Issues

### Issue: "Module not found"
```bash
# Solution: Install dependencies
pip install -r requirements_modern.txt
```

### Issue: "Playwright not found"
```bash
# Solution: Install browsers
playwright install chromium
```

### Issue: "Redis connection failed"
```bash
# Solution: Start Redis or use memory backend
docker run -d -p 6379:6379 redis:7-alpine

# Or disable Redis
python src/modern_main.py --a2a-backend memory --task "Your task"
```

### Issue: "LiteLLM API key missing"
```bash
# Solution: Set environment variable
export OPENAI_API_KEY="your_key"

# Or disable LiteLLM
python src/modern_main.py --no-litellm --task "Your task"
```

---

## ✅ Success Checklist

After quickstart, you should be able to:

- [ ] Run `demo_modern_full.py` successfully
- [ ] See system status with `--status` flag
- [ ] Execute a simple task
- [ ] View component statistics
- [ ] Access documentation files

If all checked, you're ready to build! 🎉

---

**Time to first task: < 5 minutes ⚡**
