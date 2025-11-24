#!/usr/bin/env python3
"""
Modern Qweenone Agentic System

Fully modernized agentic architecture integrating:
- Prefect workflow orchestration
- ROMA recursive task decomposition
- OmniParser desktop automation
- Playwright browser automation
- LiteLLM universal API routing
- Redis/RabbitMQ A2A communication
"""

import asyncio
import argparse
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from src.utils.logger import setup_logger

from src.workflow_engine.prefect_manager import PrefectWorkflowManager, ModernTask
from src.task_decomposition.orchestrator import TaskDecompositionOrchestrator
from src.desktop_automation.omni_automation import OmniDesktopAutomation, DesktopAutomationAgent
from src.browser_automation.playwright_automation import PlaywrightAutomation, BrowserAutomationAgent
from src.api_router.litellm_router import LiteLLMUnifiedRouter, LiteLLMConfig
from src.communication.modern_a2a_manager import ModernA2ACommunicationManager, A2AConfig, CommunicationBackend

from src.builders.agent_builder import AgentBuilder
from src.testing.test_runner import TestRunner


class ModernAgenticSystem:
    """
    Modernized agentic system with production-ready components
    
    Features:
    - Prefect-based workflow orchestration (replaces AdvancedTaskManager)
    - ROMA recursive task decomposition (replaces TaskDecomposer)
    - Vision-based desktop automation (OmniParser + PyAutoGUI)
    - Modern browser automation (Playwright)
    - Universal LLM routing (LiteLLM)
    - Enterprise A2A communication (Redis/RabbitMQ)
    """
    
    def __init__(self, 
                 use_prefect: bool = True,
                 use_roma: bool = True,
                 enable_desktop_automation: bool = True,
                 enable_browser_automation: bool = True,
                 enable_litellm: bool = True,
                 a2a_backend: str = "memory"):
        """Initialize modern agentic system"""
        
        self.logger = setup_logger("ModernAgenticSystem")
        
        self.use_prefect = use_prefect
        self.use_roma = use_roma
        self.enable_desktop_automation = enable_desktop_automation
        self.enable_browser_automation = enable_browser_automation
        self.enable_litellm = enable_litellm
        
        self.logger.info("🚀 Initializing Modern Agentic System")
        
        self.workflow_manager = PrefectWorkflowManager() if use_prefect else None
        
        self.task_orchestrator = TaskDecompositionOrchestrator(enable_orchestra=use_roma) if use_roma else None
        
        self.desktop_automation = OmniDesktopAutomation() if enable_desktop_automation else None
        
        self.browser_automation = PlaywrightAutomation() if enable_browser_automation else None
        
        self.llm_router = None
        if enable_litellm:
            try:
                self.llm_router = LiteLLMUnifiedRouter()
                self.logger.info("✅ LiteLLM router initialized")
            except Exception as e:
                self.logger.warning(f"LiteLLM not available: {e}")
        
        backend_map = {
            "memory": CommunicationBackend.MEMORY,
            "redis": CommunicationBackend.REDIS,
            "rabbitmq": CommunicationBackend.RABBITMQ
        }
        
        a2a_config = A2AConfig(backend=backend_map.get(a2a_backend, CommunicationBackend.MEMORY))
        self.a2a_manager = ModernA2ACommunicationManager(a2a_config)
        
        self.agent_builder = AgentBuilder()
        self.test_runner = TestRunner()
        
        self.agents = {}
        self.execution_history = []
        
        self.logger.info("✅ Modern Agentic System initialized")
        self.logger.info(f"   • Workflow: {'Prefect' if use_prefect else 'Legacy'}")
        self.logger.info(f"   • Task Decomposition: {'ROMA' if use_roma else 'Legacy'}")
        self.logger.info(f"   • Desktop Automation: {'Enabled' if enable_desktop_automation else 'Disabled'}")
        self.logger.info(f"   • Browser Automation: {'Enabled' if enable_browser_automation else 'Disabled'}")
        self.logger.info(f"   • LLM Router: {'LiteLLM' if self.llm_router else 'Legacy'}")
        self.logger.info(f"   • A2A Backend: {a2a_backend.upper()}")
    
    async def initialize(self):
        """Async initialization for components requiring it"""
        await self.a2a_manager.initialize()
        
        if self.browser_automation:
            try:
                await self.browser_automation.start_browser()
                self.logger.info("✅ Browser automation initialized")
            except Exception as e:
                self.logger.warning(f"Browser automation initialization failed: {e}")
    
    async def decompose_task(self, 
                           task_description: str, 
                           iterations: int = 3) -> Dict[str, Any]:
        """
        Decompose task using modern ROMA recursive planning
        """
        
        self.logger.info(f"📋 Decomposing task: {task_description}")
        
        if self.use_roma and self.task_orchestrator:
            try:
                plan = await self.task_orchestrator.create_plan(task_description, iterations)
                self.logger.info(f"✅ ROMA decomposition complete: {plan.get('total_subtasks', 0)} subtasks")
                automation = plan.get("automation_focus", {})
                if automation:
                    score = automation.get("automation_score", 0.0)
                    confidence = automation.get("confidence", 0.0)
                    self.logger.info(f"   • Automation potential: {score:.1%}")
                    self.logger.info(f"   • Confidence: {confidence:.1%}")
                    if automation.get("automation_iterations"):
                        self.logger.info(f"   • Automation iterations: {automation['automation_iterations']}")
                if plan.get("agent_teams"):
                    self.logger.info(f"   • Agent teams: {len(plan['agent_teams'])}")
                return plan
            except Exception as e:
                self.logger.error(f"ROMA decomposition failed: {e}")
                self.logger.info("Falling back to legacy decomposition")
        from src.task_manager.task_decomposer import TaskDecomposer
        legacy_decomposer = TaskDecomposer()
        return legacy_decomposer.decompose(task_description, iterations)
    
    async def execute_workflow(self, 
                              tasks: List[ModernTask],
                              workflow_name: str = "qweenone_workflow") -> Dict[str, Any]:
        """
        Execute workflow using modern Prefect orchestration
        """
        
        self.logger.info(f"⚡ Executing workflow: {workflow_name} ({len(tasks)} tasks)")
        
        if self.use_prefect and self.workflow_manager:
            try:
                result = await self.workflow_manager.execute_workflow(
                    tasks=tasks,
                    workflow_name=workflow_name,
                    parallel_execution=True
                )
                
                self.logger.info(f"✅ Workflow completed: {result.get('successful_tasks', 0)}/{result.get('total_tasks', 0)} successful")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Prefect workflow execution failed: {e}")
                self.logger.info("Falling back to legacy execution")
        
        from src.task_manager.advanced_task_manager import AdvancedTaskManager
        legacy_manager = AdvancedTaskManager()
        return {"status": "legacy_execution", "tasks": len(tasks)}
    
    async def execute_desktop_automation(self, 
                                       task_description: str) -> Dict[str, Any]:
        """
        Execute desktop automation using OmniParser vision-based approach
        """
        
        self.logger.info(f"🖥️ Executing desktop automation: {task_description}")
        
        if not self.enable_desktop_automation or not self.desktop_automation:
            return {"success": False, "error": "Desktop automation not enabled"}
        
        try:
            result = await self.desktop_automation.execute_natural_language_task(task_description)
            
            if result.get("overall_success"):
                self.logger.info(f"✅ Desktop automation completed successfully")
            else:
                self.logger.warning(f"⚠️ Desktop automation completed with issues")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Desktop automation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_browser_automation(self, 
                                        task_description: str) -> Dict[str, Any]:
        """
        Execute browser automation using Playwright
        """
        
        self.logger.info(f"🌐 Executing browser automation: {task_description}")
        
        if not self.enable_browser_automation or not self.browser_automation:
            return {"success": False, "error": "Browser automation not enabled"}
        
        try:
            result = await self.browser_automation.execute_natural_language_task(task_description)
            
            if result.get("overall_success"):
                self.logger.info(f"✅ Browser automation completed successfully")
            else:
                self.logger.warning(f"⚠️ Browser automation completed with issues")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Browser automation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _detect_automation_target(self, plan: Dict[str, Any]) -> str:
        for iteration in plan.get("iterations", []):
            for subtask in iteration.get("subtasks", []):
                text = f"{subtask.get('title', '')} {subtask.get('description', '')}".lower()
                if any(keyword in text for keyword in ["browser", "web", "playwright", "scrape"]):
                    return "browser"
        return "desktop"
    
    async def llm_query(self, 
                       prompt: str, 
                       model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """
        Query LLM using unified LiteLLM router
        """
        
        if not self.enable_litellm or not self.llm_router:
            return {"success": False, "error": "LiteLLM not enabled"}
        
        try:
            result = await self.llm_router.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM query failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def register_agent(self, agent_id: str, handler_func=None):
        """Register agent with modern A2A communication"""
        await self.a2a_manager.register_agent(agent_id, handler_func)
        
        self.agents[agent_id] = {
            "registered_at": datetime.now(),
            "type": "modern",
            "capabilities": []
        }
        
        self.logger.info(f"✅ Agent registered: {agent_id}")
    
    async def full_pipeline(self, 
                          task_description: str,
                          iterations: int = 3) -> Dict[str, Any]:
        """
        Execute full modern pipeline: decompose -> plan -> execute
        """
        
        self.logger.info(f"🚀 Starting full modern pipeline for: {task_description}")
        
        pipeline_result = {
            "task_description": task_description,
            "started_at": datetime.now().isoformat(),
            "phases": {}
        }
        
        try:
            decomposition = await self.decompose_task(task_description, iterations)
            pipeline_result["phases"]["decomposition"] = {
                "status": "completed",
                "total_subtasks": decomposition.get("total_subtasks", 0)
            }
            
            modern_tasks = []
            for iteration in decomposition.get("iterations", []):
                for subtask in iteration.get("subtasks", []):
                    modern_task = ModernTask(
                        id=subtask.get("id", f"task_{len(modern_tasks)}"),
                        title=subtask.get("title", "Untitled"),
                        description=subtask.get("description", ""),
                        agent_type=subtask.get("agent_type", "default")
                    )
                    modern_tasks.append(modern_task)
            
            if modern_tasks:
                workflow_result = await self.execute_workflow(
                    tasks=modern_tasks,
                    workflow_name=f"pipeline_{task_description[:20]}"
                )
                pipeline_result["phases"]["workflow_execution"] = workflow_result
            
            automation_focus = decomposition.get("automation_focus", {})
            if automation_focus:
                pipeline_result["automation_focus"] = automation_focus
            if automation_focus.get("high_priority"):
                target = self._detect_automation_target(decomposition)
                if target == "browser":
                    automation_result = await self.execute_browser_automation(task_description)
                else:
                    automation_result = await self.execute_desktop_automation(task_description)
                pipeline_result["phases"]["automation"] = automation_result
            elif "automation" in task_description.lower() or "gui" in task_description.lower():
                if "browser" in task_description.lower() or "web" in task_description.lower():
                    automation_result = await self.execute_browser_automation(task_description)
                else:
                    automation_result = await self.execute_desktop_automation(task_description)
                pipeline_result["phases"]["automation"] = automation_result
            
            pipeline_result["status"] = "completed"
            pipeline_result["completed_at"] = datetime.now().isoformat()
            
            self.logger.info("✅ Full pipeline completed successfully")
            
            return pipeline_result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            pipeline_result["status"] = "failed"
            pipeline_result["error"] = str(e)
            return pipeline_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive modern system status"""
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "system": "Modern Qweenone Agentic Architecture",
            "version": "2.0.0",
            "components": {
                "workflow_orchestration": {
                    "enabled": self.use_prefect,
                    "engine": "Prefect" if self.use_prefect else "Legacy",
                    "stats": self.workflow_manager.get_system_stats() if self.workflow_manager else {}
                },
                "task_decomposition": {
                    "enabled": self.use_roma,
                    "engine": "ROMA" if self.use_roma else "Legacy"
                },
                "desktop_automation": {
                    "enabled": self.enable_desktop_automation,
                    "stats": self.desktop_automation.get_automation_stats() if self.desktop_automation else {}
                },
                "browser_automation": {
                    "enabled": self.enable_browser_automation,
                    "stats": self.browser_automation.get_automation_stats() if self.browser_automation else {}
                },
                "llm_routing": {
                    "enabled": self.enable_litellm,
                    "stats": self.llm_router.get_statistics() if self.llm_router else {}
                },
                "a2a_communication": {
                    "stats": self.a2a_manager.get_communication_stats()
                }
            },
            "agents": {
                "count": len(self.agents),
                "ids": list(self.agents.keys())
            },
            "execution_history_size": len(self.execution_history)
        }
        
        return status
    
    async def close(self):
        """Cleanup and close all connections"""
        
        self.logger.info("🔌 Closing Modern Agentic System")
        
        if self.browser_automation:
            await self.browser_automation.close_browser()
        
        if self.a2a_manager:
            await self.a2a_manager.close()
        
        self.logger.info("✅ System closed successfully")


async def main_async():
    """Async main function"""
    
    parser = argparse.ArgumentParser(description="Modern Qweenone Agentic System")
    parser.add_argument("--task", type=str, help="Task to execute")
    parser.add_argument("--iterations", type=int, default=3, help="Number of decomposition iterations")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--no-prefect", action="store_true", help="Disable Prefect")
    parser.add_argument("--no-roma", action="store_true", help="Disable ROMA")
    parser.add_argument("--no-desktop", action="store_true", help="Disable desktop automation")
    parser.add_argument("--no-browser", action="store_true", help="Disable browser automation")
    parser.add_argument("--no-litellm", action="store_true", help="Disable LiteLLM")
    parser.add_argument("--a2a-backend", type=str, default="memory", choices=["memory", "redis", "rabbitmq"], help="A2A communication backend")
    
    args = parser.parse_args()
    
    system = ModernAgenticSystem(
        use_prefect=not args.no_prefect,
        use_roma=not args.no_roma,
        enable_desktop_automation=not args.no_desktop,
        enable_browser_automation=not args.no_browser,
        enable_litellm=not args.no_litellm,
        a2a_backend=args.a2a_backend
    )
    
    await system.initialize()
    
    try:
        if args.status:
            status = system.get_system_status()
            print(json.dumps(status, indent=2))
        
        elif args.demo:
            print("🎯 Running Modern Agentic System Demo")
            
            demo_tasks = [
                "Create a simple Python calculator script",
                "Research the latest trends in AI automation",
                "Take a screenshot of the current desktop"
            ]
            
            for task in demo_tasks:
                print(f"\n▶️ Executing: {task}")
                result = await system.full_pipeline(task, iterations=2)
                print(f"✅ Result: {result.get('status', 'unknown')}")
        
        elif args.task:
            print(f"🚀 Executing task: {args.task}")
            result = await system.full_pipeline(args.task, args.iterations)
            print("\n📊 Pipeline Result:")
            print(json.dumps(result, indent=2, default=str))
        
        else:
            print("Use --task to specify a task, --status for system status, or --demo for demonstration")
    
    finally:
        await system.close()


def main():
    """Main entry point"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
