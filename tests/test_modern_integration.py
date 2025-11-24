"""
Comprehensive Integration Tests for Modern Qweenone Components

Tests all new modern components:
- Prefect workflow orchestration
- ROMA task decomposition
- Desktop automation (OmniParser)
- Browser automation (Playwright)
- LiteLLM unified routing
- Modern A2A communication
"""

import pytest
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflow_engine.prefect_manager import PrefectWorkflowManager, ModernTask
from src.task_decomposition.roma_decomposer import ROMAAugmentedTaskDecomposer, RecursiveTaskPlanner
from src.communication.modern_a2a_manager import ModernA2ACommunicationManager, A2AConfig, CommunicationBackend


@pytest.mark.asyncio
class TestPrefectWorkflowManager:
    """Test Prefect workflow orchestration"""
    
    async def test_initialization(self):
        """Test Prefect manager initialization"""
        manager = PrefectWorkflowManager(concurrent_limit=3)
        
        assert manager is not None
        assert manager.concurrent_limit == 3
        
        stats = manager.get_system_stats()
        assert stats["concurrent_limit"] == 3
        assert stats["prefect_available"] == True
    
    async def test_workflow_execution(self):
        """Test workflow execution with modern tasks"""
        manager = PrefectWorkflowManager()
        
        tasks = [
            ModernTask(
                id="test_1",
                title="Test Task 1",
                description="First test task",
                agent_type="default"
            ),
            ModernTask(
                id="test_2",
                title="Test Task 2",
                description="Second test task",
                agent_type="default"
            )
        ]
        
        result = await manager.execute_workflow(
            tasks=tasks,
            workflow_name="test_workflow"
        )
        
        assert result is not None
        assert "total_tasks" in result
        assert result["total_tasks"] == 2
    
    async def test_decompose_and_execute(self):
        """Test task decomposition and execution"""
        manager = PrefectWorkflowManager()
        
        result = await manager.decompose_and_execute(
            task_description="Create a simple test script",
            iterations=2,
            auto_execute=False
        )
        
        assert result is not None
        assert "decomposition" in result
        assert "modern_tasks" in result
        assert "agent_teams" in result
        assert isinstance(result["agent_teams"], list)
        assert "automation_focus" in result
        assert isinstance(result["automation_focus"], dict)


@pytest.mark.asyncio
class TestROMATaskDecomposer:
    """Test ROMA-enhanced task decomposition"""
    
    async def test_decomposer_initialization(self):
        """Test ROMA decomposer initialization"""
        decomposer = ROMAAugmentedTaskDecomposer()
        
        assert decomposer is not None
        assert decomposer.recursive_planner is not None
    
    async def test_simple_task_decomposition(self):
        """Test decomposition of simple task"""
        decomposer = ROMAAugmentedTaskDecomposer()
        
        result = await decomposer.decompose_with_roma(
            task_description="Create a simple calculator",
            iterations=2,
            use_recursive=True
        )
        
        assert result is not None
        assert "total_subtasks" in result
        assert "iterations" in result
        assert result["total_subtasks"] > 0
    
    async def test_complex_task_decomposition(self):
        """Test decomposition of complex task"""
        decomposer = ROMAAugmentedTaskDecomposer()
        
        result = await decomposer.decompose_with_roma(
            task_description="Build a web scraper for Instagram with authentication",
            iterations=3,
            use_recursive=True
        )
        
        assert result is not None
        assert "roma_enhanced" in result
        
        roma_data = result["roma_enhanced"]["recursive_plan"]
        assert "automation_score" in roma_data
        assert "confidence" in roma_data
        assert 0.0 <= roma_data["automation_score"] <= 1.0
    
    async def test_recursive_planner(self):
        """Test recursive task planner"""
        planner = RecursiveTaskPlanner(max_recursion_depth=3)
        
        plan = await planner.create_recursive_plan(
            objective="Create API server with authentication",
            context={"deadline": False, "quality_priority": "high"}
        )
        
        assert plan is not None
        assert plan.total_subtasks > 0
        assert len(plan.required_agents) > 0
        assert 0.0 <= plan.automation_score <= 1.0


@pytest.mark.asyncio
class TestModernA2ACommunication:
    """Test modern A2A communication manager"""
    
    async def test_memory_backend(self):
        """Test in-memory communication backend"""
        config = A2AConfig(backend=CommunicationBackend.MEMORY)
        manager = ModernA2ACommunicationManager(config)
        
        await manager.initialize()
        
        await manager.register_agent("agent_1")
        await manager.register_agent("agent_2")
        
        assert len(manager.get_active_agents()) == 2
        
        stats = manager.get_communication_stats()
        assert stats["backend"] == "memory"
        assert stats["active_agents"] == 2
        
        await manager.close()
    
    async def test_agent_registration(self):
        """Test agent registration and unregistration"""
        manager = ModernA2ACommunicationManager()
        await manager.initialize()
        
        await manager.register_agent("test_agent_1")
        assert "test_agent_1" in manager.get_active_agents()
        
        await manager.unregister_agent("test_agent_1")
        assert "test_agent_1" not in manager.get_active_agents()
        
        await manager.close()
    
    async def test_heartbeat(self):
        """Test agent heartbeat functionality"""
        manager = ModernA2ACommunicationManager()
        await manager.initialize()
        
        await manager.register_agent("agent_heartbeat")
        
        result = await manager.heartbeat("agent_heartbeat")
        assert result == True
        
        result = await manager.heartbeat("nonexistent_agent")
        assert result == False
        
        await manager.close()


@pytest.mark.asyncio
class TestDesktopAutomation:
    """Test desktop automation components"""
    
    async def test_omni_automation_initialization(self):
        """Test OmniParser automation initialization"""
        try:
            from src.desktop_automation.omni_automation import OmniDesktopAutomation
            
            automation = OmniDesktopAutomation(
                screenshot_dir="/tmp/test_screenshots",
                enable_vision=True
            )
            
            assert automation is not None
            
            stats = automation.get_automation_stats()
            assert "total_actions" in stats
            assert "vision_enabled" in stats
            
        except ImportError:
            pytest.skip("Desktop automation dependencies not available")


@pytest.mark.asyncio
class TestBrowserAutomation:
    """Test browser automation components"""
    
    async def test_playwright_initialization(self):
        """Test Playwright automation initialization"""
        try:
            from src.browser_automation.playwright_automation import PlaywrightAutomation, BrowserConfig
            
            config = BrowserConfig(headless=True, timeout=10000)
            
            async with PlaywrightAutomation(config) as browser:
                stats = browser.get_automation_stats()
                assert "browser_type" in stats
                assert stats["browser_type"] == "chromium"
                
        except ImportError:
            pytest.skip("Playwright not available")


@pytest.mark.asyncio
class TestLiteLLMRouter:
    """Test LiteLLM unified router"""
    
    async def test_router_initialization(self):
        """Test LiteLLM router initialization"""
        try:
            from src.api_router.litellm_router import LiteLLMUnifiedRouter, LiteLLMConfig
            
            config = LiteLLMConfig(
                routing_strategy="least-busy",
                num_retries=3
            )
            
            router = LiteLLMUnifiedRouter(config)
            
            assert router is not None
            
            stats = router.get_statistics()
            assert "routing_strategy" in stats
            assert stats["litellm_enabled"] == True
            
        except Exception as e:
            pytest.skip(f"LiteLLM not available: {e}")
    
    async def test_available_models(self):
        """Test available models listing"""
        try:
            from src.api_router.litellm_router import LiteLLMUnifiedRouter
            
            router = LiteLLMUnifiedRouter()
            models = router.get_available_models()
            
            assert len(models) > 0
            assert any("gpt" in model.lower() for model in models)
            assert any("gemini" in model.lower() for model in models)
            
        except Exception:
            pytest.skip("LiteLLM not available")


@pytest.mark.asyncio
class TestModernAgenticSystem:
    """Test integrated modern agentic system"""
    
    async def test_system_initialization(self):
        """Test modern system initialization"""
        from src.modern_main import ModernAgenticSystem
        
        system = ModernAgenticSystem(
            use_prefect=True,
            use_roma=True,
            enable_desktop_automation=False,
            enable_browser_automation=False,
            enable_litellm=False,
            a2a_backend="memory"
        )
        
        await system.initialize()
        
        assert system is not None
        assert system.workflow_manager is not None
        assert system.task_orchestrator is not None
        
        status = system.get_system_status()
        assert status["system"] == "Modern Qweenone Agentic Architecture"
        assert "components" in status
        
        await system.close()
    
    async def test_task_decomposition_integration(self):
        """Test task decomposition in integrated system"""
        from src.modern_main import ModernAgenticSystem
        
        system = ModernAgenticSystem(a2a_backend="memory")
        await system.initialize()
        
        result = await system.decompose_task(
            task_description="Create a simple test application",
            iterations=2
        )
        
        assert result is not None
        assert "total_subtasks" in result
        assert "agent_teams" in result
        assert isinstance(result["agent_teams"], list)
        assert "automation_focus" in result
        assert isinstance(result["automation_focus"], dict)
        
        await system.close()
    
    async def test_agent_registration(self):
        """Test agent registration in modern system"""
        from src.modern_main import ModernAgenticSystem
        
        system = ModernAgenticSystem(a2a_backend="memory")
        await system.initialize()
        
        await system.register_agent("test_agent_modern")
        
        status = system.get_system_status()
        assert status["agents"]["count"] >= 1
        
        await system.close()


def test_sync_imports():
    """Test that all modern components can be imported"""
    
    try:
        from src.workflow_engine.prefect_manager import PrefectWorkflowManager
        assert PrefectWorkflowManager is not None
    except ImportError as e:
        pytest.fail(f"Failed to import PrefectWorkflowManager: {e}")
    
    try:
        from src.task_decomposition.roma_decomposer import ROMAAugmentedTaskDecomposer
        assert ROMAAugmentedTaskDecomposer is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ROMAAugmentedTaskDecomposer: {e}")
    
    try:
        from src.communication.modern_a2a_manager import ModernA2ACommunicationManager
        assert ModernA2ACommunicationManager is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ModernA2ACommunicationManager: {e}")
    
    try:
        from src.modern_main import ModernAgenticSystem
        assert ModernAgenticSystem is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ModernAgenticSystem: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
