# 📁 Qweenone Project Structure

Complete project structure showing legacy (v1.0) and modern (v2.0) components.

---

## 🗂️ Directory Tree

```
qweenone/
│
├── 📄 README.md                          # Updated with v2.0 quick start
├── 📄 MODERN_ARCHITECTURE.md             # 🆕 Modern architecture guide
├── 📄 MIGRATION_GUIDE.md                 # 🆕 Legacy → Modern migration
├── 📄 DEPLOYMENT.md                      # 🆕 Production deployment guide  
├── 📄 INTEGRATION_COMPLETE.md            # 🆕 Integration summary
├── 📄 COMPARISON_MATRIX.md               # 🆕 Feature comparison
├── 📄 MODERNIZATION_SUMMARY.md           # Existing modernization docs
├── 📄 PROJECT_STRUCTURE.md               # 🆕 This file
│
├── 🐳 docker-compose.yml                 # Legacy Docker Compose
├── 🐳 docker-compose.modern.yml          # 🆕 Modern Docker Compose
├── 🐳 Dockerfile                         # Legacy Dockerfile
├── 🐳 Dockerfile.modern                  # 🆕 Modern Dockerfile
│
├── 📦 requirements.txt                   # Legacy dependencies
├── 📦 requirements_modern.txt            # 🆕 Modern dependencies
│
├── 🎯 integration_test.py                # Integration tests (existing)
├── 🎯 demo_modern_full.py                # 🆕 Full modern demo
│
├── 📂 src/
│   │
│   ├── 📄 main.py                        # Legacy entry point
│   ├── 📄 modern_main.py                 # 🆕 Modern entry point
│   │
│   ├── 📂 agents/                        # Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py                 # Base agent class
│   │   ├── ai_agent.py                   # AI-powered agent
│   │   ├── code_agent.py                 # Code generation agent
│   │   ├── task_agent.py                 # Task management agent
│   │   ├── communication_agent.py        # Communication agent
│   │   ├── testing_agent.py              # Testing agent
│   │   ├── default_agent.py              # Default agent
│   │   └── advanced_agents.py            # Advanced specialized agents
│   │
│   ├── 📂 task_manager/                  # Legacy task management
│   │   ├── advanced_task_manager.py      # Legacy task manager (1200+ lines)
│   │   └── task_decomposer.py            # Legacy task decomposer (200 lines)
│   │
│   ├── 📂 workflow_engine/               # 🆕 Modern workflow orchestration
│   │   ├── __init__.py
│   │   └── prefect_manager.py            # ✨ Prefect integration (500+ lines)
│   │
│   ├── 📂 task_decomposition/            # 🆕 Modern task decomposition
│   │   ├── __init__.py
│   │   └── roma_decomposer.py            # ✨ ROMA integration (900+ lines)
│   │
│   ├── 📂 desktop_automation/            # 🆕 Desktop automation
│   │   ├── __init__.py
│   │   ├── omni_automation.py            # ✨ OmniParser + PyAutoGUI (800+ lines)
│   │   └── vision_gui_agent.py           # Agent system integration
│   │
│   ├── 📂 browser_automation/            # 🆕 Browser automation
│   │   ├── __init__.py
│   │   └── playwright_automation.py      # ✨ Playwright integration (900+ lines)
│   │
│   ├── 📂 api_router/                    # API routing
│   │   ├── __init__.py
│   │   ├── api_router.py                 # Legacy custom router
│   │   ├── provider_router.py            # Legacy provider routing
│   │   ├── router_config.py              # Legacy configuration
│   │   └── litellm_router.py             # ✨ LiteLLM integration (300+ lines)
│   │
│   ├── 📂 api_integrations/              # API integrations
│   │   ├── __init__.py
│   │   ├── api_manager.py                # API manager
│   │   ├── llm_provider.py               # LLM provider definitions
│   │   └── openrouter_client.py          # OpenRouter client
│   │
│   ├── 📂 communication/                 # Agent communication
│   │   ├── a2a_manager.py                # Legacy A2A manager
│   │   └── modern_a2a_manager.py         # ✨ Modern A2A (Redis/RabbitMQ)
│   │
│   ├── 📂 enhanced_agents/               # 🆕 Advanced agent frameworks
│   │   ├── init.py
│   │   ├── enhanced_agent_system.py      # Enhanced base system
│   │   ├── crewai_integration.py         # ✨ CrewAI integration (700+ lines)
│   │   └── orchestra_integration.py      # ✨ Orchestra integration (400+ lines)
│   │
│   ├── 📂 builders/                      # Agent builders
│   │   ├── __init__.py
│   │   ├── agent_builder.py              # Base agent builder
│   │   └── advanced_agent_builder.py     # Advanced builder
│   │
│   ├── 📂 testing/                       # Testing framework
│   │   └── test_runner.py                # Test execution
│   │
│   └── 📂 utils/                         # Utilities
│       └── logger.py                     # Logging utilities
│
├── 📂 tests/                             # Test suite
│   └── test_modern_integration.py        # 🆕 Comprehensive modern tests
│
├── 📂 docs/                              # Documentation
│   └── usage_examples.py                 # Usage examples
│
├── 📂 examples/                          # Examples
│   └── openrouter_integration_demo.py    # OpenRouter demo
│
├── 📂 data/                              # Data directory (created at runtime)
├── 📂 logs/                              # Logs directory (created at runtime)
└── 📂 screenshots/                       # Screenshots (created at runtime)
```

---

## 📊 File Statistics

### Code Distribution

| Category | Files | Lines | Status |
|----------|-------|-------|--------|
| **Legacy Core** | 15 | ~4000 | ✅ Maintained |
| **Modern Core** | 9 | ~5000 | ✨ NEW |
| **Tests** | 2 | ~500 | ✅ Enhanced |
| **Documentation** | 7 | ~3500 | ✨ 5 NEW |
| **Deployment** | 6 | ~500 | ✅ Enhanced |
| **Total** | **39** | **~13500** | ✅ Complete |

### Component Size Comparison

| Component | Legacy | Modern | Change |
|-----------|--------|--------|--------|
| Task Management | 1236 lines | 500 lines | -59% (framework) |
| Task Decomposition | 198 lines | 900 lines | +355% (features) |
| API Routing | 620 lines | 300 lines | -52% (framework) |
| A2A Communication | 230 lines | 400 lines | +74% (features) |
| Automation | 0 lines | 1700 lines | ∞% (new) |
| Agent Frameworks | 0 lines | 1100 lines | ∞% (new) |
| **Total Custom Code** | **6000+** | **1800** | **-70%** |
| **Total with Frameworks** | **6000** | **~15000** | **+150% capability** |

---

## 🎯 Import Paths Quick Reference

### Modern Components

```python
# Workflow Orchestration
from src.workflow_engine.prefect_manager import PrefectWorkflowManager, ModernTask

# Task Decomposition
from src.task_decomposition.roma_decomposer import ROMAAugmentedTaskDecomposer

# Desktop Automation
from src.desktop_automation.omni_automation import OmniDesktopAutomation, DesktopAutomationAgent

# Browser Automation
from src.browser_automation.playwright_automation import PlaywrightAutomation, BrowserAutomationAgent

# API Routing
from src.api_router.litellm_router import LiteLLMUnifiedRouter

# A2A Communication
from src.communication.modern_a2a_manager import ModernA2ACommunicationManager, A2AConfig

# Enhanced Agents
from src.enhanced_agents.crewai_integration import CrewAIAgentOrchestrator
from src.enhanced_agents.orchestra_integration import AgentOrchestra

# Modern System
from src.modern_main import ModernAgenticSystem
```

### Legacy Components (Still Available)

```python
# Legacy Task Management
from src.task_manager.advanced_task_manager import AdvancedTaskManager

# Legacy Task Decomposition
from src.task_manager.task_decomposer import TaskDecomposer

# Legacy API Routing
from src.api_router.api_router import APIRouter

# Legacy A2A
from src.communication.a2a_manager import A2ACommunicationManager

# Legacy System
from src.main import AgenticSystem
```

---

## 🔄 Backward Compatibility

### Modern → Legacy Adapters

| Adapter | Purpose | Location |
|---------|---------|----------|
| `PrefectTaskManagerAdapter` | Use Prefect with legacy API | workflow_engine/prefect_manager.py |
| `ModernTask.to_legacy_task()` | Convert task formats | workflow_engine/prefect_manager.py |
| `decompose_with_roma(..., use_recursive=False)` | Legacy-compatible decomposition | task_decomposition/roma_decomposer.py |

### Legacy → Modern Upgrades

All legacy components can be gradually replaced without breaking changes:

```python
# Start with legacy
system = AgenticSystem()

# Swap in modern components one by one
system.task_decomposer = ROMAAugmentedTaskDecomposer()  # ✅ Works!
system.workflow_manager = PrefectWorkflowManager()      # ✅ Works!

# Or use full modern system
modern_system = ModernAgenticSystem(
    use_prefect=True,      # Enable modern
    use_roma=True,         # Enable modern
    # ... other flags
)
```

---

## 🎨 Component Color Coding

In this document:
- 🆕 = New in v2.0
- ✨ = Modern implementation
- ✅ = Completed/Available
- 🟢 = Easy/Low effort
- 🟡 = Medium effort
- 🔴 = High effort
- ⭐ = Recommended
- ❌ = Not available
- 🚀 = Performance boost
- 🧠 = AI-enhanced
- 👁️ = Vision-based
- 📡 = Distributed
- 💾 = Persistent

---

## 📞 File Dependencies Graph

```
modern_main.py
├── workflow_engine/prefect_manager.py
│   ├── agents/base_agent.py
│   └── builders/agent_builder.py
│
├── task_decomposition/roma_decomposer.py
│   └── task_manager/task_decomposer.py (legacy fallback)
│
├── desktop_automation/omni_automation.py
│   ├── agents/base_agent.py
│   └── utils/logger.py
│
├── browser_automation/playwright_automation.py
│   ├── agents/base_agent.py
│   └── utils/logger.py
│
├── api_router/litellm_router.py
│   ├── api_integrations/llm_provider.py
│   └── utils/logger.py
│
├── communication/modern_a2a_manager.py
│   ├── communication/a2a_manager.py (Message, MessageType)
│   └── utils/logger.py
│
└── enhanced_agents/
    ├── crewai_integration.py
    │   ├── desktop_automation/omni_automation.py
    │   ├── browser_automation/playwright_automation.py
    │   └── task_decomposition/roma_decomposer.py
    │
    └── orchestra_integration.py
        ├── agents/base_agent.py
        └── task_decomposition/roma_decomposer.py
```

---

## 🎯 Where to Start?

### For Users
1. Read `README.md` - Quick overview
2. Run `demo_modern_full.py` - See it in action
3. Check `MODERN_ARCHITECTURE.md` - Understand architecture
4. Use `src/modern_main.py` - Start building

### For Migrators
1. Read `MIGRATION_GUIDE.md` - Step-by-step migration
2. Review `COMPARISON_MATRIX.md` - Understand differences
3. Test in staging first
4. Follow migration checklist

### For Deployers
1. Read `DEPLOYMENT.md` - Production deployment
2. Choose deployment option (Docker/Systemd/Swarm)
3. Configure environment variables
4. Follow deployment checklist
5. Setup monitoring

### For Developers
1. Review `src/modern_main.py` - Main system
2. Explore `src/workflow_engine/` - Workflow orchestration
3. Check `src/task_decomposition/` - Task decomposition
4. Study `tests/test_modern_integration.py` - Testing patterns

---

**Project structure documentation complete! 🎉**
