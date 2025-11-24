"""
CrewAI Integration for Multi-Agent Orchestration

Integrates CrewAI's role-based agent orchestration framework with qweenone.
Provides production-ready multi-agent collaboration with specialized roles.

CrewAI Benefits:
- Role-based agent design with clear responsibilities
- Built-in collaboration patterns and communication
- Task delegation and coordination
- Memory and context sharing between agents
- Production-ready orchestration engine
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import json

# CrewAI imports (with fallback)
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    from crewai.agent import Agent as CrewAgent
    from crewai.task import Task as CrewTask
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Agent = None
    Task = None
    Crew = None
    Process = None
    BaseTool = None

# Import existing qweenone components
try:
    from ..agents.base_agent import BaseAgent, AgentResponse
    from ..utils.logger import setup_logger
    from ..desktop_automation.omni_automation import OmniDesktopAutomation
    from ..browser_automation.playwright_automation import PlaywrightAutomation
except ImportError:
    BaseAgent = None
    AgentResponse = None
    setup_logger = lambda x: None
    OmniDesktopAutomation = None
    PlaywrightAutomation = None


@dataclass
class CrewAIAgentRole:
    """Predefined roles for CrewAI agents with qweenone integration"""
    name: str
    role: str
    goal: str
    backstory: str
    tools: List[str] = field(default_factory=list)
    allow_delegation: bool = False
    verbose: bool = True
    memory: bool = True


class QweenoneTools:
    """
    qweenone-specific tools for CrewAI agents
    
    Bridges qweenone's automation capabilities with CrewAI's tool system
    """
    
    @staticmethod
    def create_desktop_automation_tool():
        """Create desktop automation tool for CrewAI"""
        
        if not CREWAI_AVAILABLE or not OmniDesktopAutomation:
            return None
        
        class DesktopAutomationTool(BaseTool):
            name = "desktop_automation"
            description = "Automate desktop applications using computer vision and natural language"
            
            def __init__(self):
                super().__init__()
                self.automation = OmniDesktopAutomation()
            
            def _run(self, task_description: str) -> str:
                """Execute desktop automation task"""
                try:
                    # Note: CrewAI tools are sync, but our automation is async
                    # In production, would need async tool support or adapter
                    result = asyncio.run(self.automation.execute_natural_language_task(task_description))
                    
                    if result.get("overall_success", False):
                        return f"Desktop automation successful: {len(result.get('step_results', []))} steps executed"
                    else:
                        return f"Desktop automation failed: {result.get('error', 'Unknown error')}"
                        
                except Exception as e:
                    return f"Desktop automation error: {str(e)}"
        
        return DesktopAutomationTool()
    
    @staticmethod
    def create_browser_automation_tool():
        """Create browser automation tool for CrewAI"""
        
        if not CREWAI_AVAILABLE or not PlaywrightAutomation:
            return None
        
        class BrowserAutomationTool(BaseTool):
            name = "browser_automation"
            description = "Automate web browsers using Playwright for web scraping and interaction"
            
            def _run(self, task_description: str) -> str:
                """Execute browser automation task"""
                try:
                    # Create Playwright automation
                    automation = PlaywrightAutomation()
                    
                    # Execute async task in sync context
                    async def execute_browser_task():
                        async with automation:
                            return await automation.execute_natural_language_task(task_description)
                    
                    result = asyncio.run(execute_browser_task())
                    
                    if result.get("overall_success", False):
                        return f"Browser automation successful: {result.get('current_url', 'No URL')} - {len(result.get('action_results', []))} actions"
                    else:
                        return f"Browser automation failed: {result.get('error', 'Unknown error')}"
                        
                except Exception as e:
                    return f"Browser automation error: {str(e)}"
        
        return BrowserAutomationTool()
    
    @staticmethod
    def create_task_decomposition_tool():
        """Create task decomposition tool for CrewAI"""
        
        if not CREWAI_AVAILABLE:
            return None
        
        class TaskDecompositionTool(BaseTool):
            name = "task_decomposition"
            description = "Decompose complex tasks into manageable subtasks using ROMA recursive planning"
            
            def _run(self, complex_task: str) -> str:
                """Decompose complex task"""
                try:
                    from ..task_decomposition.roma_decomposer import ROMAAugmentedTaskDecomposer
                    
                    decomposer = ROMAAugmentedTaskDecomposer()
                    
                    # Execute async decomposition
                    result = asyncio.run(decomposer.decompose_with_roma(complex_task, iterations=3))
                    
                    subtasks_count = result.get("total_subtasks", 0)
                    automation_score = 0
                    
                    if "roma_enhanced" in result:
                        automation_score = result["roma_enhanced"]["recursive_plan"].get("automation_score", 0)
                    
                    return f"Task decomposed into {subtasks_count} subtasks with {automation_score:.1%} automation potential"
                    
                except Exception as e:
                    return f"Task decomposition error: {str(e)}"
        
        return TaskDecompositionTool()
    
    @staticmethod
    def create_workflow_orchestration_tool():
        """Create workflow orchestration tool for CrewAI"""
        
        if not CREWAI_AVAILABLE:
            return None
        
        class WorkflowOrchestrationTool(BaseTool):
            name = "workflow_orchestration"
            description = "Orchestrate complex workflows using Prefect modern workflow engine"
            
            def _run(self, workflow_description: str) -> str:
                """Execute workflow orchestration"""
                try:
                    from ..workflow_engine.prefect_manager import PrefectWorkflowManager, ModernTask
                    
                    manager = PrefectWorkflowManager()
                    
                    # Create simple workflow task
                    task = ModernTask(
                        id="crewai_workflow",
                        title="CrewAI Orchestrated Workflow",
                        description=workflow_description,
                        tags=["crewai", "orchestrated"]
                    )
                    
                    # Execute workflow
                    result = asyncio.run(manager.execute_workflow([task]))
                    
                    if result.get("success", False):
                        return f"Workflow executed successfully: {result.get('successful_tasks', 0)} tasks completed"
                    else:
                        return f"Workflow execution failed: {result.get('failed_tasks', 0)} tasks failed"
                        
                except Exception as e:
                    return f"Workflow orchestration error: {str(e)}"
        
        return WorkflowOrchestrationTool()


class CrewAIAgentOrchestrator:
    """
    CrewAI-based agent orchestrator for qweenone
    
    Provides production-ready multi-agent orchestration using CrewAI's
    proven role-based collaboration patterns.
    """
    
    def __init__(self):
        self.logger = setup_logger("CrewAIAgentOrchestrator")
        self.active_crews: Dict[str, Crew] = {}
        self.agent_roles: Dict[str, CrewAIAgentRole] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        if not CREWAI_AVAILABLE:
            if self.logger:
                self.logger.warning("CrewAI not available - install with: pip install crewai")
            return
        
        # Initialize predefined agent roles
        self._initialize_predefined_roles()
        
        if self.logger:
            self.logger.info(f"CrewAI orchestrator initialized with {len(self.agent_roles)} predefined roles")
    
    def _initialize_predefined_roles(self):
        """Initialize predefined agent roles optimized for qweenone tasks"""
        
        roles = [
            CrewAIAgentRole(
                name="automation_specialist",
                role="Automation Specialist",
                goal="Design and implement automated solutions for complex tasks",
                backstory="You are an expert in automation technologies including GUI automation, web scraping, and workflow orchestration. You excel at creating efficient automated solutions.",
                tools=["desktop_automation", "browser_automation", "workflow_orchestration"],
                allow_delegation=True
            ),
            CrewAIAgentRole(
                name="task_planner",
                role="Task Planning Expert", 
                goal="Break down complex objectives into manageable, well-structured task plans",
                backstory="You are a strategic planning expert who excels at decomposing complex problems into clear, actionable steps with optimal resource allocation.",
                tools=["task_decomposition", "workflow_orchestration"],
                allow_delegation=False
            ),
            CrewAIAgentRole(
                name="systems_analyst",
                role="Systems Analysis Specialist",
                goal="Analyze system architectures and provide optimization recommendations",
                backstory="You are a systems architect with deep expertise in analyzing complex systems, identifying bottlenecks, and recommending improvements.",
                tools=["task_decomposition"],
                allow_delegation=False
            ),
            CrewAIAgentRole(
                name="integration_expert", 
                role="Integration Specialist",
                goal="Integrate different systems and technologies seamlessly",
                backstory="You are an integration expert who specializes in connecting disparate systems, APIs, and technologies to create cohesive solutions.",
                tools=["desktop_automation", "browser_automation"],
                allow_delegation=True
            ),
            CrewAIAgentRole(
                name="quality_assurance",
                role="Quality Assurance Engineer",
                goal="Ensure high quality and reliability of automated solutions",
                backstory="You are a QA expert focused on testing, validation, and quality assurance of automated systems and workflows.",
                tools=["workflow_orchestration"],
                allow_delegation=False
            )
        ]
        
        for role in roles:
            self.agent_roles[role.name] = role
    
    def create_crewai_crew(self, 
                          task_description: str,
                          required_roles: List[str] = None,
                          process_type: str = "sequential") -> Optional[str]:
        """
        Create CrewAI crew for specific task
        
        Returns crew_id for tracking
        """
        if not CREWAI_AVAILABLE:
            if self.logger:
                self.logger.error("CrewAI not available")
            return None
        
        # Auto-select roles if not specified
        if not required_roles:
            required_roles = self._auto_select_roles(task_description)
        
        try:
            # Create agents for selected roles
            crew_agents = []
            crew_tasks = []
            
            for role_name in required_roles:
                if role_name not in self.agent_roles:
                    if self.logger:
                        self.logger.warning(f"Role '{role_name}' not found, skipping")
                    continue
                
                role_config = self.agent_roles[role_name]
                
                # Create tools for agent
                agent_tools = []
                for tool_name in role_config.tools:
                    tool = self._get_qweenone_tool(tool_name)
                    if tool:
                        agent_tools.append(tool)
                
                # Create CrewAI agent
                agent = Agent(
                    role=role_config.role,
                    goal=role_config.goal,
                    backstory=role_config.backstory,
                    tools=agent_tools,
                    allow_delegation=role_config.allow_delegation,
                    verbose=role_config.verbose,
                    memory=role_config.memory
                )
                
                crew_agents.append(agent)
                
                # Create task for this agent
                task = Task(
                    description=f"As a {role_config.role}, contribute to: {task_description}",
                    agent=agent,
                    expected_output=f"Detailed {role_config.role.lower()} contribution to the task"
                )
                
                crew_tasks.append(task)
            
            if not crew_agents:
                if self.logger:
                    self.logger.error("No valid agents created for crew")
                return None
            
            # Create crew
            process_map = {
                "sequential": Process.sequential,
                "hierarchical": Process.hierarchical
            }
            
            crew = Crew(
                agents=crew_agents,
                tasks=crew_tasks,
                process=process_map.get(process_type, Process.sequential),
                verbose=True,
                memory=True
            )
            
            # Store crew
            crew_id = f"crew_{int(asyncio.get_event_loop().time())}"
            self.active_crews[crew_id] = crew
            
            if self.logger:
                self.logger.info(f"Created CrewAI crew '{crew_id}' with {len(crew_agents)} agents")
            
            return crew_id
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to create CrewAI crew: {e}")
            return None
    
    def _auto_select_roles(self, task_description: str) -> List[str]:
        """Automatically select appropriate roles based on task description"""
        
        task_lower = task_description.lower()
        selected_roles = []
        
        # Always include task planner for complex tasks
        if len(task_description.split()) > 10:
            selected_roles.append("task_planner")
        
        # Automation-related tasks
        if any(word in task_lower for word in ["automate", "automation", "gui", "browser", "scrape"]):
            selected_roles.append("automation_specialist")
        
        # System analysis tasks
        if any(word in task_lower for word in ["analyze", "system", "architecture", "optimize"]):
            selected_roles.append("systems_analyst")
        
        # Integration tasks
        if any(word in task_lower for word in ["integrate", "connect", "api", "interface"]):
            selected_roles.append("integration_expert")
        
        # Quality/testing tasks
        if any(word in task_lower for word in ["test", "quality", "validate", "verify"]):
            selected_roles.append("quality_assurance")
        
        # Ensure at least one role
        if not selected_roles:
            selected_roles = ["automation_specialist"]  # Default role
        
        return selected_roles
    
    def _get_qweenone_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get qweenone-specific tool by name"""
        
        if not CREWAI_AVAILABLE:
            return None
        
        tool_map = {
            "desktop_automation": QweenoneTools.create_desktop_automation_tool,
            "browser_automation": QweenoneTools.create_browser_automation_tool,
            "task_decomposition": QweenoneTools.create_task_decomposition_tool,
            "workflow_orchestration": QweenoneTools.create_workflow_orchestration_tool
        }
        
        tool_creator = tool_map.get(tool_name)
        if tool_creator:
            return tool_creator()
        
        return None
    
    async def execute_crew_task(self, 
                              crew_id: str,
                              execution_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute task using CrewAI crew"""
        
        if crew_id not in self.active_crews:
            return {"success": False, "error": f"Crew '{crew_id}' not found"}
        
        crew = self.active_crews[crew_id]
        
        if self.logger:
            self.logger.info(f"Executing CrewAI crew '{crew_id}'")
        
        try:
            # Execute crew (note: CrewAI kickoff is sync)
            start_time = datetime.now()
            
            # Run in thread pool to handle sync CrewAI with async qweenone
            loop = asyncio.get_event_loop()
            crew_result = await loop.run_in_executor(None, crew.kickoff)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Process results
            result = {
                "success": True,
                "crew_id": crew_id,
                "execution_time": execution_time,
                "crew_result": str(crew_result),  # CrewAI returns string result
                "agents_involved": len(crew.agents),
                "tasks_completed": len(crew.tasks),
                "process_type": crew.process.value if hasattr(crew.process, 'value') else str(crew.process),
                "framework": "crewai"
            }
            
            # Store execution history
            self.execution_history.append({
                "crew_id": crew_id,
                "result": result,
                "context": execution_context,
                "timestamp": start_time.isoformat()
            })
            
            if self.logger:
                self.logger.info(f"CrewAI execution completed in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"CrewAI crew execution failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "crew_id": crew_id,
                "framework": "crewai"
            }
    
    async def create_and_execute_crew(self, 
                                    task_description: str,
                                    roles: List[str] = None,
                                    process_type: str = "sequential") -> Dict[str, Any]:
        """
        Create crew and execute task in one operation
        
        Convenience method for quick CrewAI orchestration
        """
        
        # Create crew
        crew_id = self.create_crewai_crew(
            task_description=task_description,
            required_roles=roles,
            process_type=process_type
        )
        
        if not crew_id:
            return {
                "success": False,
                "error": "Failed to create CrewAI crew",
                "task_description": task_description
            }
        
        # Execute crew
        result = await self.execute_crew_task(crew_id)
        
        # Add task context
        result["task_description"] = task_description
        result["roles_used"] = roles or self._auto_select_roles(task_description)
        
        return result
    
    def get_crew_status(self, crew_id: str) -> Dict[str, Any]:
        """Get status of specific crew"""
        
        if crew_id not in self.active_crews:
            return {"status": "not_found", "crew_id": crew_id}
        
        crew = self.active_crews[crew_id]
        
        return {
            "status": "active",
            "crew_id": crew_id,
            "agents_count": len(crew.agents),
            "tasks_count": len(crew.tasks),
            "process_type": crew.process.value if hasattr(crew.process, 'value') else str(crew.process),
            "memory_enabled": getattr(crew, 'memory', False),
            "verbose_mode": getattr(crew, 'verbose', False)
        }
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestration statistics"""
        
        total_executions = len(self.execution_history)
        successful_executions = sum(
            1 for exec_record in self.execution_history 
            if exec_record["result"].get("success", False)
        )
        
        return {
            "crewai_available": CREWAI_AVAILABLE,
            "total_crews_created": len(self.active_crews),
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
            "available_roles": list(self.agent_roles.keys()),
            "qweenone_tools_integrated": 4,  # desktop, browser, decomposition, workflow
            "average_execution_time": self._calculate_average_execution_time()
        }
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time across all crews"""
        
        if not self.execution_history:
            return 0.0
        
        total_time = sum(
            record["result"].get("execution_time", 0) 
            for record in self.execution_history
        )
        
        return total_time / len(self.execution_history)


# Integration adapter for legacy compatibility
class CrewAIAgentAdapter:
    """
    Adapter to use CrewAI agents as qweenone BaseAgent
    
    Provides seamless integration between CrewAI's role-based agents
    and qweenone's existing agent architecture.
    """
    
    def __init__(self, crewai_orchestrator: CrewAIAgentOrchestrator):
        self.orchestrator = crewai_orchestrator
        self.logger = setup_logger("CrewAIAgentAdapter")
    
    async def execute_as_base_agent(self, 
                                  task: Union[Dict[str, Any], 'LegacyTask'],
                                  preferred_roles: List[str] = None) -> Union['AgentResponse', Dict[str, Any]]:
        """
        Execute task using CrewAI but return BaseAgent-compatible response
        """
        
        # Extract task description
        if isinstance(task, dict):
            task_description = task.get("description", task.get("title", ""))
            task_id = task.get("id", "unknown")
        else:
            task_description = getattr(task, 'description', getattr(task, 'title', ''))
            task_id = getattr(task, 'task_id', 'unknown')
        
        if self.logger:
            self.logger.info(f"Executing task via CrewAI adapter: {task_id}")
        
        try:
            # Execute using CrewAI orchestration
            result = await self.orchestrator.create_and_execute_crew(
                task_description=task_description,
                roles=preferred_roles
            )
            
            # Convert to BaseAgent response format
            if AgentResponse:
                return AgentResponse(
                    success=result.get("success", False),
                    data=result,
                    error=result.get("error", ""),
                    metadata={
                        "framework": "crewai",
                        "roles_used": result.get("roles_used", []),
                        "agents_involved": result.get("agents_involved", 0)
                    }
                )
            else:
                # Fallback format
                return {
                    "success": result.get("success", False),
                    "data": result,
                    "error": result.get("error", ""),
                    "metadata": {"framework": "crewai"}
                }
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"CrewAI adapter execution failed: {e}")
            
            error_response = {
                "success": False,
                "error": str(e),
                "framework": "crewai_adapter"
            }
            
            if AgentResponse:
                return AgentResponse(**error_response)
            else:
                return error_response


# Convenience functions for easy integration
async def execute_with_crewai(task_description: str, roles: List[str] = None) -> Dict[str, Any]:
    """Quick CrewAI execution for any task"""
    
    orchestrator = CrewAIAgentOrchestrator()
    return await orchestrator.create_and_execute_crew(task_description, roles)


def get_available_crewai_roles() -> List[str]:
    """Get list of available CrewAI roles"""
    
    orchestrator = CrewAIAgentOrchestrator()
    return list(orchestrator.agent_roles.keys())


if __name__ == "__main__":
    """Demo CrewAI integration"""
    
    async def demo_crewai_integration():
        """Demonstrate CrewAI integration with qweenone"""
        
        print("👥 === CREWAI MULTI-AGENT ORCHESTRATION DEMO ===")
        
        if not CREWAI_AVAILABLE:
            print("❌ CrewAI not available. Install with: pip install crewai")
            return
        
        # Create orchestrator
        orchestrator = CrewAIAgentOrchestrator()
        
        # Test tasks with different role combinations
        test_scenarios = [
            {
                "task": "Design and implement a web scraping solution for e-commerce data",
                "roles": ["task_planner", "automation_specialist", "quality_assurance"],
                "description": "Complex multi-agent collaboration"
            },
            {
                "task": "Analyze system performance and recommend optimizations",
                "roles": ["systems_analyst"],
                "description": "Single specialist agent"
            },
            {
                "task": "Create automated testing pipeline for desktop applications",
                "roles": ["automation_specialist", "integration_expert", "quality_assurance"],
                "description": "Automation-focused team"
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n🎯 Scenario {i}: {scenario['description']}")
            print(f"📋 Task: {scenario['task']}")
            print(f"👥 Roles: {', '.join(scenario['roles'])}")
            
            result = await orchestrator.create_and_execute_crew(
                task_description=scenario["task"],
                roles=scenario["roles"],
                process_type="sequential"
            )
            
            print(f"✅ Success: {result.get('success', False)}")
            print(f"⏱️ Execution time: {result.get('execution_time', 0):.2f}s")
            print(f"🤖 Agents involved: {result.get('agents_involved', 0)}")
            
            if result.get("crew_result"):
                result_preview = str(result["crew_result"])[:100]
                print(f"📄 Result preview: {result_preview}...")
        
        # Show orchestration stats
        stats = orchestrator.get_orchestration_stats()
        print(f"\n📊 CrewAI Orchestration Stats:")
        print(f"  • Total crews: {stats['total_crews_created']}")
        print(f"  • Total executions: {stats['total_executions']}")
        print(f"  • Success rate: {stats['success_rate']:.1f}%")
        print(f"  • Available roles: {len(stats['available_roles'])}")
        print(f"  • Integrated tools: {stats['qweenone_tools_integrated']}")
    
    # Run demo
    asyncio.run(demo_crewai_integration())

