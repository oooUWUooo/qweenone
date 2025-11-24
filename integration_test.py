#!/usr/bin/env python3
"""
Comprehensive Integration Test for Modernized qweenone System

Tests the integration of all new modern components:
- Prefect workflow orchestration
- ROMA-enhanced task decomposition  
- OmniParser desktop automation
- LiteLLM API routing (planned)

Demonstrates backward compatibility and new capabilities.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test availability of modern components
COMPONENTS_AVAILABLE = {}

try:
    from src.workflow_engine.prefect_manager import PrefectWorkflowManager, ModernTask
    COMPONENTS_AVAILABLE["prefect"] = True
    print("✅ Prefect workflow engine available")
except ImportError as e:
    COMPONENTS_AVAILABLE["prefect"] = False
    print(f"❌ Prefect not available: {e}")

try:
    from src.task_decomposition.roma_decomposer import ROMAAugmentedTaskDecomposer
    COMPONENTS_AVAILABLE["roma"] = True
    print("✅ ROMA task decomposer available")
except ImportError as e:
    COMPONENTS_AVAILABLE["roma"] = False
    print(f"❌ ROMA decomposer not available: {e}")

try:
    from src.desktop_automation.omni_automation import OmniDesktopAutomation
    from src.desktop_automation.vision_gui_agent import VisionGUIAgent
    COMPONENTS_AVAILABLE["desktop_automation"] = True
    print("✅ Desktop automation available")
except ImportError as e:
    COMPONENTS_AVAILABLE["desktop_automation"] = False
    print(f"❌ Desktop automation not available: {e}")

# Test legacy compatibility
try:
    from src.task_manager.task_decomposer import TaskDecomposer
    from src.task_manager.advanced_task_manager import AdvancedTaskManager
    COMPONENTS_AVAILABLE["legacy"] = True
    print("✅ Legacy components available")
except ImportError as e:
    COMPONENTS_AVAILABLE["legacy"] = False
    print(f"❌ Legacy components not available: {e}")


class IntegrationTester:
    """Comprehensive tester for modernized qweenone system"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
        # Initialize available components
        self.prefect_manager = None
        self.roma_decomposer = None
        self.desktop_automation = None
        self.vision_agent = None
        self.legacy_manager = None
        
        self._initialize_available_components()
    
    def _initialize_available_components(self):
        """Initialize all available components"""
        
        if COMPONENTS_AVAILABLE["prefect"]:
            self.prefect_manager = PrefectWorkflowManager(concurrent_limit=3)
            print("🔧 Prefect manager initialized")
        
        if COMPONENTS_AVAILABLE["roma"]:
            self.roma_decomposer = ROMAAugmentedTaskDecomposer()
            print("🔧 ROMA decomposer initialized")
        
        if COMPONENTS_AVAILABLE["desktop_automation"]:
            self.desktop_automation = OmniDesktopAutomation()
            self.vision_agent = VisionGUIAgent()
            print("🔧 Desktop automation initialized")
        
        if COMPONENTS_AVAILABLE["legacy"]:
            self.legacy_manager = AdvancedTaskManager()
            print("🔧 Legacy manager initialized")
    
    async def test_task_decomposition_comparison(self):
        """Test modern vs legacy task decomposition"""
        print("\n🧠 === TASK DECOMPOSITION COMPARISON ===")
        
        test_task = "Create a desktop automation tool for data entry with GUI recognition"
        
        results = {}
        
        # Test ROMA decomposition
        if self.roma_decomposer:
            print("🚀 Testing ROMA-enhanced decomposition...")
            start_time = time.time()
            
            try:
                roma_result = await self.roma_decomposer.decompose_with_roma(
                    task_description=test_task,
                    iterations=3,
                    use_recursive=True
                )
                
                roma_time = time.time() - start_time
                
                results["roma"] = {
                    "success": True,
                    "execution_time": roma_time,
                    "total_subtasks": roma_result.get("total_subtasks", 0),
                    "iterations": roma_result.get("total_iterations", 0),
                    "enhanced_features": "roma_enhanced" in roma_result
                }
                
                if "roma_enhanced" in roma_result:
                    roma_data = roma_result["roma_enhanced"]["recursive_plan"]
                    results["roma"].update({
                        "automation_score": roma_data.get("automation_score", 0),
                        "confidence": roma_data.get("confidence", 0),
                        "strategy": roma_data.get("strategy", "unknown")
                    })
                
                print(f"  ✅ ROMA: {results['roma']['total_subtasks']} subtasks in {roma_time:.2f}s")
                
                if results["roma"]["enhanced_features"]:
                    print(f"  🎯 Automation potential: {results['roma'].get('automation_score', 0):.1%}")
                    print(f"  🎯 Plan confidence: {results['roma'].get('confidence', 0):.1%}")
                
            except Exception as e:
                results["roma"] = {"success": False, "error": str(e)}
                print(f"  ❌ ROMA failed: {e}")
        
        # Test legacy decomposition for comparison
        if COMPONENTS_AVAILABLE["legacy"]:
            print("📊 Testing legacy decomposition...")
            start_time = time.time()
            
            try:
                legacy_decomposer = TaskDecomposer()
                legacy_result = await asyncio.to_thread(legacy_decomposer.decompose, test_task, 3)
                
                legacy_time = time.time() - start_time
                
                results["legacy"] = {
                    "success": True,
                    "execution_time": legacy_time,
                    "total_subtasks": legacy_result.get("total_subtasks", 0),
                    "iterations": legacy_result.get("total_iterations", 0)
                }
                
                print(f"  ✅ Legacy: {results['legacy']['total_subtasks']} subtasks in {legacy_time:.2f}s")
                
            except Exception as e:
                results["legacy"] = {"success": False, "error": str(e)}
                print(f"  ❌ Legacy failed: {e}")
        
        # Comparison
        if "roma" in results and "legacy" in results:
            if results["roma"]["success"] and results["legacy"]["success"]:
                speedup = results["legacy"]["execution_time"] / results["roma"]["execution_time"]
                subtask_improvement = results["roma"]["total_subtasks"] / results["legacy"]["total_subtasks"]
                
                print(f"\n📈 Comparison Results:")
                print(f"  • Speed: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
                print(f"  • Subtask detail: {subtask_improvement:.2f}x more detailed")
                print(f"  • Enhanced features: {'✅' if results['roma'].get('enhanced_features') else '❌'}")
        
        self.test_results["task_decomposition"] = results
        return results
    
    async def test_workflow_orchestration(self):
        """Test Prefect workflow vs legacy task execution"""
        print("\n⚡ === WORKFLOW ORCHESTRATION TEST ===")
        
        # Define test tasks
        test_tasks = [
            {
                "id": "workflow_test_1",
                "title": "Analyze Requirements", 
                "description": "Analyze project requirements for automation tool",
                "agent_type": "analysis"
            },
            {
                "id": "workflow_test_2",
                "title": "Design Architecture",
                "description": "Design system architecture based on requirements", 
                "agent_type": "planning"
            },
            {
                "id": "workflow_test_3",
                "title": "Implement Core Features",
                "description": "Implement core automation features",
                "agent_type": "code_writer"
            }
        ]
        
        results = {}
        
        # Test Prefect workflow
        if self.prefect_manager:
            print("🚀 Testing Prefect workflow orchestration...")
            start_time = time.time()
            
            try:
                # Convert to ModernTask objects
                modern_tasks = []
                for task_data in test_tasks:
                    modern_task = ModernTask(
                        id=task_data["id"],
                        title=task_data["title"],
                        description=task_data["description"],
                        agent_type=task_data["agent_type"],
                        tags=["integration_test", "modern"]
                    )
                    modern_tasks.append(modern_task)
                
                # Execute workflow
                workflow_result = await self.prefect_manager.execute_workflow(
                    tasks=modern_tasks,
                    workflow_name="integration_test_workflow",
                    parallel_execution=True
                )
                
                prefect_time = time.time() - start_time
                
                results["prefect"] = {
                    "success": workflow_result.get("success", False),
                    "execution_time": prefect_time,
                    "total_tasks": workflow_result.get("total_tasks", 0),
                    "successful_tasks": workflow_result.get("successful_tasks", 0),
                    "orchestration_overhead": workflow_result.get("total_execution_time", 0) - prefect_time
                }
                
                print(f"  ✅ Prefect: {results['prefect']['successful_tasks']}/{results['prefect']['total_tasks']} tasks in {prefect_time:.2f}s")
                
            except Exception as e:
                results["prefect"] = {"success": False, "error": str(e)}
                print(f"  ❌ Prefect failed: {e}")
        
        # Test legacy workflow for comparison
        if self.legacy_manager:
            print("📊 Testing legacy task execution...")
            start_time = time.time()
            
            try:
                legacy_tasks = []
                successful_count = 0
                
                for task_data in test_tasks:
                    legacy_task = self.legacy_manager.create_task(
                        title=task_data["title"],
                        description=task_data["description"],
                        task_type=task_data["agent_type"],
                        priority=3
                    )
                    legacy_tasks.append(legacy_task)
                
                # Execute tasks
                for task in legacy_tasks:
                    success = self.legacy_manager.submit_task(task)
                    if success:
                        successful_count += 1
                
                legacy_time = time.time() - start_time
                
                results["legacy"] = {
                    "success": successful_count == len(legacy_tasks),
                    "execution_time": legacy_time,
                    "total_tasks": len(legacy_tasks),
                    "successful_tasks": successful_count
                }
                
                print(f"  ✅ Legacy: {results['legacy']['successful_tasks']}/{results['legacy']['total_tasks']} tasks in {legacy_time:.2f}s")
                
            except Exception as e:
                results["legacy"] = {"success": False, "error": str(e)}
                print(f"  ❌ Legacy failed: {e}")
        
        self.test_results["workflow_orchestration"] = results
        return results
    
    async def test_desktop_automation(self):
        """Test desktop automation capabilities"""
        print("\n🖥️ === DESKTOP AUTOMATION TEST ===")
        
        if not self.desktop_automation:
            print("❌ Desktop automation not available")
            return {"success": False, "error": "Desktop automation not available"}
        
        # Test basic automation
        print("👁️ Testing OmniParser desktop automation...")
        
        try:
            # Simple automation test
            result = await self.desktop_automation.execute_natural_language_task(
                "Take a screenshot and analyze the current desktop"
            )
            
            automation_stats = self.desktop_automation.get_automation_stats()
            
            desktop_result = {
                "success": result.get("overall_success", False),
                "vision_enabled": automation_stats.get("vision_enabled", False),
                "omniparser_available": automation_stats.get("omniparser_available", False),
                "elements_detected": automation_stats.get("detected_elements_count", 0),
                "steps_executed": len(result.get("step_results", []))
            }
            
            print(f"  ✅ Desktop automation: {desktop_result['steps_executed']} steps executed")
            print(f"  👁️ Vision enabled: {desktop_result['vision_enabled']}")
            print(f"  🎯 Elements detected: {desktop_result['elements_detected']}")
            
            # Test vision agent integration
            if self.vision_agent:
                print("🤖 Testing VisionGUIAgent...")
                
                agent_status = self.vision_agent.get_agent_status()
                vision_result = {
                    "agent_active": True,
                    "capabilities": len(agent_status.get("capabilities", [])),
                    "integration_mode": agent_status.get("integration_mode", "unknown")
                }
                
                print(f"  ✅ Vision agent: {vision_result['capabilities']} capabilities")
                print(f"  🔧 Integration: {vision_result['integration_mode']}")
                
                desktop_result["vision_agent"] = vision_result
            
            self.test_results["desktop_automation"] = desktop_result
            return desktop_result
            
        except Exception as e:
            print(f"❌ Desktop automation test failed: {e}")
            error_result = {"success": False, "error": str(e)}
            self.test_results["desktop_automation"] = error_result
            return error_result
    
    async def test_integration_compatibility(self):
        """Test backward compatibility with existing system"""
        print("\n🔄 === BACKWARD COMPATIBILITY TEST ===")
        
        compatibility_results = {
            "prefect_with_legacy_agents": False,
            "roma_with_legacy_format": False,
            "desktop_automation_with_base_agent": False
        }
        
        # Test 1: Prefect with legacy agent system
        if self.prefect_manager and COMPONENTS_AVAILABLE["legacy"]:
            try:
                # Create task using legacy format but execute with Prefect
                legacy_task = {
                    "id": "compat_test_1",
                    "title": "Compatibility Test Task",
                    "description": "Test task to verify Prefect works with legacy agents",
                    "agent_type": "default"
                }
                
                modern_task = ModernTask(
                    id=legacy_task["id"],
                    title=legacy_task["title"],
                    description=legacy_task["description"],
                    agent_type=legacy_task["agent_type"],
                    tags=["compatibility_test"]
                )
                
                result = await self.prefect_manager.execute_workflow([modern_task])
                compatibility_results["prefect_with_legacy_agents"] = result.get("success", False)
                
                print(f"  ✅ Prefect + Legacy Agents: {compatibility_results['prefect_with_legacy_agents']}")
                
            except Exception as e:
                print(f"  ❌ Prefect + Legacy Agents failed: {e}")
        
        # Test 2: ROMA with legacy format output
        if self.roma_decomposer:
            try:
                roma_result = await self.roma_decomposer.decompose_with_roma(
                    "Test compatibility task",
                    iterations=2,
                    use_recursive=True
                )
                
                # Check if output contains legacy-compatible format
                has_legacy_format = (
                    "iterations" in roma_result and
                    "total_subtasks" in roma_result and
                    "original_task" in roma_result
                )
                
                compatibility_results["roma_with_legacy_format"] = has_legacy_format
                print(f"  ✅ ROMA + Legacy Format: {has_legacy_format}")
                
            except Exception as e:
                print(f"  ❌ ROMA + Legacy Format failed: {e}")
        
        # Test 3: Desktop automation with BaseAgent
        if self.vision_agent and COMPONENTS_AVAILABLE["legacy"]:
            try:
                # Test if VisionGUIAgent can work as BaseAgent
                test_task = {
                    "title": "Desktop Compatibility Test",
                    "description": "Test desktop automation through BaseAgent interface"
                }
                
                # This would call execute_task() which should be BaseAgent compatible
                agent_result = await self.vision_agent.execute_task(test_task)
                
                compatibility_results["desktop_automation_with_base_agent"] = True
                print(f"  ✅ Desktop Agent + BaseAgent: True")
                
            except Exception as e:
                print(f"  ❌ Desktop Agent + BaseAgent failed: {e}")
        
        self.test_results["compatibility"] = compatibility_results
        return compatibility_results
    
    async def test_full_integration_scenario(self):
        """Test complete end-to-end integration scenario"""
        print("\n🎯 === FULL INTEGRATION SCENARIO ===")
        
        if not (self.prefect_manager and self.roma_decomposer):
            print("❌ Core components not available for full integration test")
            return {"success": False, "error": "Core components missing"}
        
        try:
            # Scenario: "Create and test a simple desktop calculator automation"
            complex_task = "Create automation to open calculator, perform calculation 15+25, and verify result"
            
            print(f"📋 Complex Task: {complex_task}")
            
            # Step 1: ROMA-enhanced decomposition
            print("🧠 Step 1: ROMA task decomposition...")
            decomposition = await self.roma_decomposer.decompose_with_roma(
                task_description=complex_task,
                iterations=3
            )
            
            print(f"  📊 Decomposed into {decomposition.get('total_subtasks', 0)} subtasks")
            
            # Step 2: Convert to modern tasks
            print("⚡ Step 2: Converting to Prefect workflow...")
            modern_tasks = []
            
            for iteration in decomposition.get("iterations", []):
                for subtask in iteration.get("subtasks", []):
                    modern_task = ModernTask(
                        id=subtask.get("id", f"integration_{len(modern_tasks)}"),
                        title=subtask["title"],
                        description=subtask["description"],
                        agent_type=subtask.get("agent_type", "default"),
                        tags=["integration_scenario", f"iteration_{iteration['iteration_number']}"]
                    )
                    modern_tasks.append(modern_task)
            
            # Step 3: Execute via Prefect
            print("🚀 Step 3: Executing via Prefect workflow...")
            workflow_result = await self.prefect_manager.execute_workflow(
                tasks=modern_tasks,
                workflow_name="full_integration_scenario"
            )
            
            # Step 4: Desktop automation (if applicable)
            desktop_result = None
            if self.desktop_automation:
                print("🖥️ Step 4: Testing desktop automation...")
                desktop_result = await self.desktop_automation.execute_natural_language_task(
                    "Take screenshot and analyze current desktop for automation"
                )
            
            # Compile results
            integration_result = {
                "success": workflow_result.get("success", False),
                "decomposition_subtasks": decomposition.get("total_subtasks", 0),
                "workflow_execution": workflow_result.get("successful_tasks", 0),
                "desktop_automation": desktop_result.get("overall_success", False) if desktop_result else None,
                "total_execution_time": workflow_result.get("total_execution_time", 0),
                "components_used": [
                    comp for comp, available in COMPONENTS_AVAILABLE.items() 
                    if available and comp != "legacy"
                ]
            }
            
            print(f"\n🎉 Integration Scenario Results:")
            print(f"  ✅ Overall success: {integration_result['success']}")
            print(f"  📊 Subtasks decomposed: {integration_result['decomposition_subtasks']}")
            print(f"  ⚡ Workflow tasks executed: {integration_result['workflow_execution']}")
            print(f"  🖥️ Desktop automation: {integration_result['desktop_automation']}")
            print(f"  🔧 Components used: {', '.join(integration_result['components_used'])}")
            
            self.test_results["full_integration"] = integration_result
            return integration_result
            
        except Exception as e:
            print(f"❌ Full integration scenario failed: {e}")
            error_result = {"success": False, "error": str(e)}
            self.test_results["full_integration"] = error_result
            return error_result
    
    def generate_integration_report(self) -> str:
        """Generate comprehensive integration test report"""
        
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        
        report = f"""
# 🔄 qweenone Modernization Integration Report

**Generated:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Test Time:** {total_time:.2f} seconds

## 📊 Component Availability

{self._format_component_availability()}

## 🧪 Test Results

{self._format_test_results()}

## 💡 Recommendations

{self._generate_recommendations()}

## 🚀 Next Steps

{self._generate_next_steps()}
"""
        return report
    
    def _format_component_availability(self) -> str:
        """Format component availability section"""
        lines = []
        for component, available in COMPONENTS_AVAILABLE.items():
            status = "✅ Available" if available else "❌ Not Available"
            lines.append(f"- **{component.title()}**: {status}")
        return "\n".join(lines)
    
    def _format_test_results(self) -> str:
        """Format test results section"""
        lines = []
        
        for test_name, results in self.test_results.items():
            lines.append(f"\n### {test_name.replace('_', ' ').title()}")
            
            if isinstance(results, dict):
                if "success" in results:
                    status = "✅ Passed" if results["success"] else "❌ Failed"
                    lines.append(f"**Status:** {status}")
                
                for key, value in results.items():
                    if key != "success":
                        lines.append(f"- **{key}:** {value}")
            else:
                lines.append(f"Results: {results}")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check which components passed tests
        if self.test_results.get("task_decomposition", {}).get("roma", {}).get("success"):
            recommendations.append("✅ ROMA decomposition working well - proceed with full integration")
        
        if self.test_results.get("workflow_orchestration", {}).get("prefect", {}).get("success"):
            recommendations.append("✅ Prefect orchestration successful - can replace legacy TaskManager")
        
        if self.test_results.get("desktop_automation", {}).get("success"):
            recommendations.append("✅ Desktop automation functional - ready for production use")
        
        if not recommendations:
            recommendations.append("⚠️ Some components need installation or configuration")
            recommendations.append("📦 Run: pip install -r requirements_modern.txt")
        
        return "\n".join(f"- {rec}" for rec in recommendations)
    
    def _generate_next_steps(self) -> str:
        """Generate next steps based on results"""
        steps = [
            "1. **Install Missing Dependencies:** Ensure all modern dependencies are installed",
            "2. **Configure Components:** Set up API keys and configuration files", 
            "3. **Run Production Tests:** Test with real workloads",
            "4. **Gradual Migration:** Replace legacy components one by one",
            "5. **Monitor Performance:** Track performance improvements vs legacy system"
        ]
        
        return "\n".join(steps)


async def main():
    """Main integration test function"""
    
    print("🔄 === QWEENONE MODERNIZATION INTEGRATION TEST ===")
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize tester
    tester = IntegrationTester()
    
    # Run comprehensive tests
    await tester.test_task_decomposition_comparison()
    await tester.test_workflow_orchestration()
    await tester.test_desktop_automation() 
    await tester.test_integration_compatibility()
    await tester.test_full_integration_scenario()
    
    # Generate final report
    report = tester.generate_integration_report()
    
    # Save report
    report_path = Path("integration_test_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n📄 Integration report saved: {report_path}")
    print("\n🎉 === INTEGRATION TEST COMPLETED ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Integration test interrupted by user")
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()