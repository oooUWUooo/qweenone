"""
Vision-Based GUI Agent

Specialized agent for computer vision-guided GUI automation.
Integrates seamlessly with existing qweenone agent architecture
while providing advanced desktop automation capabilities.
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

try:
    from ..agents.base_agent import BaseAgent, AgentResponse, AgentCapability
    from ..builders.agent_builder import AgentBuilder
    from ..utils.logger import setup_logger
except ImportError:
    BaseAgent = None
    AgentResponse = None
    AgentCapability = None
    AgentBuilder = None
    setup_logger = lambda x: None

from .omni_automation import OmniDesktopAutomation, AutomationStep


@dataclass
class VisionTask:
    """Task specifically designed for vision-based automation"""
    description: str
    target_application: str = "any"
    interaction_mode: str = "intelligent"  # intelligent, precise, fast
    verification_required: bool = True
    max_attempts: int = 3
    screenshot_evidence: bool = True


class VisionGUIAgent:
    """
    Specialized agent for vision-guided desktop automation
    
    Combines computer vision capabilities with intelligent action planning
    to automate any GUI application without requiring specific selectors.
    """
    
    def __init__(self, agent_id: str = None):
        # Initialize base agent if available
        if BaseAgent:
            self.base_agent = BaseAgent(
                name="VisionGUIAgent",
                description="Vision-based GUI automation specialist",
                agent_id=agent_id
            )
            self.base_agent.add_capability(AgentCapability.TASK_EXECUTION)
            # Add custom capabilities if enum is extensible
            try:
                self.base_agent.add_capability("COMPUTER_VISION")
                self.base_agent.add_capability("GUI_AUTOMATION")
                self.base_agent.add_capability("DESKTOP_INTERACTION")
            except:
                pass  # Custom capabilities not supported
        else:
            self.base_agent = None
        
        # Initialize automation engine
        self.automation = OmniDesktopAutomation(
            enable_vision=True,
            enable_ocr_fallback=True,
            safety_mode=True
        )
        
        self.logger = setup_logger("VisionGUIAgent")
        
        # Agent state
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        
        if self.logger:
            self.logger.info("VisionGUIAgent initialized with OmniParser integration")
    
    async def execute_vision_task(self, vision_task: VisionTask) -> Dict[str, Any]:
        """Execute vision-specific automation task"""
        
        task_id = f"vision_{int(asyncio.get_event_loop().time())}"
        
        if self.logger:
            self.logger.info(f"Executing vision task {task_id}: {vision_task.description}")
        
        # Track active task
        self.active_tasks[task_id] = {
            "task": vision_task,
            "started_at": asyncio.get_event_loop().time(),
            "status": "running"
        }
        
        try:
            # Execute based on interaction mode
            if vision_task.interaction_mode == "intelligent":
                result = await self._intelligent_execution(vision_task)
            elif vision_task.interaction_mode == "precise":
                result = await self._precise_execution(vision_task)
            elif vision_task.interaction_mode == "fast":
                result = await self._fast_execution(vision_task)
            else:
                result = await self._default_execution(vision_task)
            
            # Add task tracking info
            result.update({
                "task_id": task_id,
                "interaction_mode": vision_task.interaction_mode,
                "verification_enabled": vision_task.verification_required,
                "vision_agent": "OmniParser"
            })
            
            # Move to completed tasks
            self.completed_tasks.append({
                "task_id": task_id,
                "result": result,
                "completed_at": asyncio.get_event_loop().time()
            })
            
            del self.active_tasks[task_id]
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Vision task {task_id} failed: {e}")
            
            # Update task status
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["error"] = str(e)
            
            return {
                "task_id": task_id,
                "overall_success": False,
                "error": str(e),
                "interaction_mode": vision_task.interaction_mode
            }
    
    async def _intelligent_execution(self, vision_task: VisionTask) -> Dict[str, Any]:
        """
        Intelligent execution with LLM planning and adaptive behavior
        """
        return await self.automation.execute_natural_language_task(vision_task.description)
    
    async def _precise_execution(self, vision_task: VisionTask) -> Dict[str, Any]:
        """
        Precise execution with detailed element analysis
        """
        # Take screenshot and analyze thoroughly
        screenshot = await self.automation.take_screenshot()
        elements = await self.automation.analyze_screen(screenshot, vision_task.description)
        
        if len(elements) == 0:
            return {"overall_success": False, "error": "No GUI elements detected"}
        
        # Create precise automation steps
        steps = []
        
        # Add verification step
        steps.append(AutomationStep(
            action="verify",
            target="main interface elements",
            screenshot_before=True
        ))
        
        # Add interaction steps based on detected elements
        clickable_elements = [e for e in elements if e.is_clickable]
        
        if clickable_elements:
            # Click most relevant element
            primary_element = max(clickable_elements, key=lambda e: e.confidence)
            steps.append(AutomationStep(
                action="click",
                target=primary_element.description,
                expected_result="Element activated"
            ))
        
        # Execute with high precision
        return await self.automation.execute_automation_sequence(steps)
    
    async def _fast_execution(self, vision_task: VisionTask) -> Dict[str, Any]:
        """
        Fast execution with minimal verification
        """
        # Create streamlined steps
        steps = [
            AutomationStep(
                action="screenshot",
                target="current state",
                screenshot_after=False  # Skip extra screenshots for speed
            )
        ]
        
        # Analyze task for quick actions
        description_lower = vision_task.description.lower()
        
        if "click" in description_lower:
            # Extract click target from description
            click_target = self._extract_click_target(vision_task.description)
            if click_target:
                steps.append(AutomationStep(
                    action="click",
                    target=click_target,
                    timeout=5  # Shorter timeout for speed
                ))
        
        return await self.automation.execute_automation_sequence(steps)
    
    async def _default_execution(self, vision_task: VisionTask) -> Dict[str, Any]:
        """Default execution mode"""
        return await self._intelligent_execution(vision_task)
    
    def _extract_click_target(self, description: str) -> Optional[str]:
        """Extract click target from task description"""
        # Simple pattern matching for click targets
        import re
        
        # Look for patterns like "click on X", "click the X", "press X"
        patterns = [
            r"click (?:on |the )?(.+?)(?:\s|$)",
            r"press (?:the )?(.+?)(?:\s|$)",
            r"select (?:the )?(.+?)(?:\s|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description.lower())
            if match:
                return match.group(1).strip()
        
        return None
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and statistics"""
        automation_stats = self.automation.get_automation_stats()
        
        return {
            "agent_name": "VisionGUIAgent",
            "status": "active",
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "automation_stats": automation_stats,
            "capabilities": [
                "Desktop GUI automation",
                "Vision-based element detection", 
                "Natural language task processing",
                "Multi-step workflow execution"
            ],
            "integration_mode": "omniparser_enhanced"
        }
    
    async def batch_execute_tasks(self, vision_tasks: List[VisionTask]) -> List[Dict[str, Any]]:
        """Execute multiple vision tasks in sequence"""
        
        if self.logger:
            self.logger.info(f"Batch executing {len(vision_tasks)} vision tasks")
        
        results = []
        
        for i, task in enumerate(vision_tasks, 1):
            if self.logger:
                self.logger.info(f"Executing batch task {i}/{len(vision_tasks)}")
            
            result = await self.execute_vision_task(task)
            results.append(result)
            
            # Brief pause between tasks
            await asyncio.sleep(1.0)
        
        return results
    
    # Compatibility methods for BaseAgent integration
    if BaseAgent:
        async def execute_task(self, task) -> AgentResponse:
            """BaseAgent interface compatibility"""
            
            # Convert legacy task to VisionTask
            if isinstance(task, dict):
                description = task.get("description", task.get("title", ""))
            else:
                description = getattr(task, 'description', getattr(task, 'title', ''))
            
            vision_task = VisionTask(
                description=description,
                verification_required=True,
                screenshot_evidence=True
            )
            
            result = await self.execute_vision_task(vision_task)
            
            return AgentResponse(
                success=result.get("overall_success", False),
                data=result,
                error=result.get("error", ""),
                metadata={"agent_type": "vision_gui"}
            )


# Integration with existing agent builder
def register_vision_gui_agent(agent_builder: 'AgentBuilder'):
    """Register VisionGUIAgent with existing AgentBuilder"""
    
    if not agent_builder:
        return False
    
    try:
        # Register the new agent type
        agent_builder.agent_registry["vision_gui"] = VisionGUIAgent
        agent_builder.agent_registry["desktop_automation"] = VisionGUIAgent
        
        # Add to agent profiles if available
        if hasattr(agent_builder, 'agent_profiles'):
            agent_builder.agent_profiles["vision_specialist"] = {
                "default_capabilities": ["COMPUTER_VISION", "GUI_AUTOMATION"],
                "default_parameters": {
                    "interaction_mode": "intelligent",
                    "verification_required": True
                },
                "required_parameters": []
            }
        
        return True
        
    except Exception as e:
        print(f"Failed to register VisionGUIAgent: {e}")
        return False


if __name__ == "__main__":
    """Test vision GUI agent"""
    
    async def test_vision_agent():
        """Test the VisionGUIAgent"""
        
        print("👁️ === VISION GUI AGENT TEST ===")
        
        # Create agent
        agent = VisionGUIAgent()
        
        # Test tasks
        test_tasks = [
            VisionTask(
                description="Take screenshot and analyze current desktop",
                interaction_mode="intelligent"
            ),
            VisionTask(
                description="Open calculator application",
                target_application="calculator",
                interaction_mode="precise"
            )
        ]
        
        # Execute tasks
        for i, task in enumerate(test_tasks, 1):
            print(f"\n🎯 Test {i}: {task.description}")
            
            result = await agent.execute_vision_task(task)
            
            print(f"✅ Success: {result.get('overall_success', False)}")
            print(f"📊 Steps: {len(result.get('step_results', []))}")
        
        # Show agent status
        status = agent.get_agent_status()
        print(f"\n📈 Agent Status:")
        print(f"  • Active tasks: {status['active_tasks']}")
        print(f"  • Completed tasks: {status['completed_tasks']}")
        print(f"  • Success rate: {status['automation_stats']['success_rate']:.1f}%")
    
    # Run test
    asyncio.run(test_vision_agent())