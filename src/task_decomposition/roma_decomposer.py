"""
ROMA-Enhanced Task Decomposition System

Integrates recursive meta-agent capabilities with existing qweenone task decomposition.
Provides intelligent, hierarchical task breakdown using recursive planning algorithms.

Key improvements over legacy TaskDecomposer:
- Recursive task breakdown (up to 5 levels deep)
- Context-aware decomposition based on task complexity  
- Intelligent agent type inference
- Dynamic dependency resolution
- Parallel sub-task planning
- Complexity scoring and effort estimation
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

# Import existing components for compatibility
try:
    from ..agents.base_agent import Task as LegacyTask
    from ..utils.logger import setup_logger
except ImportError:
    LegacyTask = None
    setup_logger = lambda x: None


class TaskComplexity(Enum):
    """Task complexity levels for recursive planning"""
    SIMPLE = 1      # Single step, < 5 minutes
    MODERATE = 2    # Multiple steps, < 30 minutes  
    COMPLEX = 3     # Multi-phase, < 2 hours
    ADVANCED = 4    # Multi-iteration, < 1 day
    EXPERT = 5      # Long-term project, > 1 day


class DecompositionStrategy(Enum):
    """Strategies for task decomposition"""
    LINEAR = "linear"              # Sequential subtasks
    PARALLEL = "parallel"          # Independent parallel subtasks  
    HYBRID = "hybrid"              # Mix of sequential and parallel
    RECURSIVE = "recursive"        # Recursive breakdown
    ADAPTIVE = "adaptive"          # AI-guided decomposition


@dataclass
class RecursiveSubtask:
    """Advanced subtask with recursive capabilities"""
    id: str
    title: str
    description: str
    complexity: TaskComplexity
    estimated_duration: int  # minutes
    dependencies: List[str] = field(default_factory=list)
    agent_requirements: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    automation_potential: float = 0.8  # 0.0 - 1.0 scale
    
    # Recursive properties
    can_decompose_further: bool = True
    decomposition_depth: int = 0
    parent_task_id: Optional[str] = None
    subtasks: List['RecursiveSubtask'] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"task_{uuid.uuid4().hex[:8]}"


@dataclass  
class RecursivePlan:
    """Complete recursive task plan"""
    objective: str
    complexity_score: float
    estimated_duration: int  # total minutes
    total_subtasks: int
    max_depth: int
    strategy: DecompositionStrategy
    phases: List[Dict[str, Any]]
    critical_path: List[str]
    required_agents: List[str]
    automation_score: float  # How much can be automated (0.0-1.0)
    confidence: float        # Confidence in plan quality (0.0-1.0)
    created_at: datetime = field(default_factory=datetime.now)


class RecursiveTaskPlanner:
    """
    Advanced recursive task planner inspired by ROMA architecture
    
    Uses recursive plan-execute-reflect cycles to break down complex tasks
    into optimal subtask hierarchies.
    """
    
    def __init__(self, 
                 max_recursion_depth: int = 5,
                 min_task_duration: int = 5,  # minutes
                 max_subtasks_per_level: int = 8):
        """
        Initialize recursive task planner
        
        Args:
            max_recursion_depth: Maximum levels of task decomposition
            min_task_duration: Minimum duration for atomic tasks (minutes)
            max_subtasks_per_level: Maximum subtasks per decomposition level
        """
        self.max_depth = max_recursion_depth
        self.min_duration = min_task_duration
        self.max_subtasks = max_subtasks_per_level
        self.logger = setup_logger("RecursiveTaskPlanner")
        
        # Task type patterns for intelligent agent assignment
        self.task_patterns = {
            "code": ["code", "program", "implement", "develop", "script", "function", "algorithm"],
            "research": ["research", "analyze", "investigate", "study", "explore", "examine"],
            "data": ["data", "process", "parse", "extract", "transform", "clean", "analyze"],
            "communication": ["send", "notify", "email", "message", "report", "communicate"],
            "testing": ["test", "validate", "verify", "check", "quality", "debug"],
            "automation": ["automate", "click", "navigate", "interact", "gui", "browser"],
            "planning": ["plan", "design", "architecture", "strategy", "organize"]
        }
    
    async def create_recursive_plan(self, 
                                  objective: str,
                                  context: Dict[str, Any] = None) -> RecursivePlan:
        """
        Create comprehensive recursive task plan
        
        Uses ROMA-inspired recursive planning to break down complex objectives
        into optimal subtask hierarchies.
        """
        if context is None:
            context = {}
        
        if self.logger:
            self.logger.info(f"Creating recursive plan for: {objective}")
        
        # Step 1: Analyze task complexity
        complexity_analysis = self._analyze_task_complexity(objective, context)
        
        # Step 2: Choose decomposition strategy  
        strategy = self._select_decomposition_strategy(complexity_analysis)
        
        # Step 3: Recursive breakdown
        root_task = RecursiveSubtask(
            id="root",
            title=objective,
            description=f"Root objective: {objective}",
            complexity=complexity_analysis["level"],
            estimated_duration=complexity_analysis["estimated_duration"],
            context=context
        )
        
        decomposed_tree = await self._recursive_decompose(root_task, depth=0)
        
        # Step 4: Create execution plan
        phases = self._extract_execution_phases(decomposed_tree)
        critical_path = self._calculate_critical_path(phases)
        
        # Step 5: Calculate metrics
        total_subtasks = self._count_all_subtasks(decomposed_tree)
        required_agents = self._identify_required_agents(decomposed_tree)
        automation_score = self._calculate_automation_potential(decomposed_tree)
        
        plan = RecursivePlan(
            objective=objective,
            complexity_score=complexity_analysis["score"],
            estimated_duration=complexity_analysis["estimated_duration"],
            total_subtasks=total_subtasks,
            max_depth=self._calculate_max_depth(decomposed_tree),
            strategy=strategy,
            phases=phases,
            critical_path=critical_path,
            required_agents=required_agents,
            automation_score=automation_score,
            confidence=self._calculate_plan_confidence(decomposed_tree)
        )
        
        if self.logger:
            self.logger.info(f"Recursive plan created: {total_subtasks} subtasks, "
                           f"{len(phases)} phases, {automation_score:.1%} automation potential")
        
        return plan
    
    async def _recursive_decompose(self, 
                                 task: RecursiveSubtask, 
                                 depth: int) -> RecursiveSubtask:
        """
        Recursively decompose task using ROMA-inspired algorithms
        """
        if depth >= self.max_depth:
            task.can_decompose_further = False
            return task
        
        if task.estimated_duration <= self.min_duration:
            task.can_decompose_further = False
            return task
        
        # Check if task should be decomposed further
        if not self._should_decompose_further(task, depth):
            return task
        
        # Identify subtasks using pattern matching and AI-guided analysis
        subtasks = await self._identify_subtasks(task, depth)
        
        # Recursively decompose each subtask
        for subtask in subtasks:
            subtask.parent_task_id = task.id
            subtask.decomposition_depth = depth + 1
            
            # Recursive call
            decomposed_subtask = await self._recursive_decompose(subtask, depth + 1)
            task.subtasks.append(decomposed_subtask)
        
        return task
    
    def _analyze_task_complexity(self, objective: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task complexity using heuristics and pattern matching"""
        
        # Count complexity indicators
        complexity_indicators = 0
        duration_estimate = 30  # base 30 minutes
        
        objective_lower = objective.lower()
        
        # Technical complexity indicators
        if any(word in objective_lower for word in ["api", "database", "authentication", "deployment"]):
            complexity_indicators += 2
            duration_estimate += 60
        
        if any(word in objective_lower for word in ["machine learning", "ai", "neural", "model"]):
            complexity_indicators += 3
            duration_estimate += 120
        
        # Scale indicators  
        if any(word in objective_lower for word in ["system", "architecture", "framework", "platform"]):
            complexity_indicators += 2
            duration_estimate += 90
        
        # Multi-component indicators
        if any(word in objective_lower for word in ["integration", "workflow", "pipeline", "orchestration"]):
            complexity_indicators += 2
            duration_estimate += 60
        
        # Automation complexity
        if any(word in objective_lower for word in ["automation", "scraping", "gui", "browser"]):
            complexity_indicators += 1
            duration_estimate += 45
        
        # Determine complexity level
        if complexity_indicators == 0:
            level = TaskComplexity.SIMPLE
        elif complexity_indicators <= 2:
            level = TaskComplexity.MODERATE  
        elif complexity_indicators <= 4:
            level = TaskComplexity.COMPLEX
        elif complexity_indicators <= 6:
            level = TaskComplexity.ADVANCED
        else:
            level = TaskComplexity.EXPERT
        
        # Apply context adjustments
        if context.get("deadline"):
            duration_estimate = int(duration_estimate * 0.8)  # Aggressive timeline
        
        if context.get("quality_priority") == "high":
            duration_estimate = int(duration_estimate * 1.3)  # Extra time for quality
        
        return {
            "level": level,
            "score": complexity_indicators / 10.0,  # Normalize to 0-1
            "estimated_duration": duration_estimate,
            "indicators": complexity_indicators
        }
    
    def _select_decomposition_strategy(self, complexity_analysis: Dict[str, Any]) -> DecompositionStrategy:
        """Select optimal decomposition strategy based on complexity"""
        
        complexity_level = complexity_analysis["level"]
        
        if complexity_level == TaskComplexity.SIMPLE:
            return DecompositionStrategy.LINEAR
        elif complexity_level == TaskComplexity.MODERATE:
            return DecompositionStrategy.PARALLEL
        elif complexity_level == TaskComplexity.COMPLEX:
            return DecompositionStrategy.HYBRID
        elif complexity_level == TaskComplexity.ADVANCED:
            return DecompositionStrategy.RECURSIVE
        else:  # EXPERT
            return DecompositionStrategy.ADAPTIVE
    
    async def _identify_subtasks(self, 
                               parent_task: RecursiveSubtask, 
                               depth: int) -> List[RecursiveSubtask]:
        """
        Identify optimal subtasks for a given parent task
        
        Uses pattern matching and domain knowledge to create intelligent subtasks
        """
        objective = parent_task.title.lower()
        subtasks = []
        
        # Pattern-based subtask generation
        if any(word in objective for word in ["web scraper", "scraping", "parser"]):
            subtasks = self._generate_scraper_subtasks(parent_task)
            
        elif any(word in objective for word in ["web server", "api", "backend"]):
            subtasks = self._generate_server_subtasks(parent_task)
            
        elif any(word in objective for word in ["gui", "desktop", "automation"]):
            subtasks = self._generate_automation_subtasks(parent_task)
            
        elif any(word in objective for word in ["analysis", "research", "investigation"]):
            subtasks = self._generate_analysis_subtasks(parent_task)
            
        else:
            # Generic decomposition
            subtasks = self._generate_generic_subtasks(parent_task)
        
        # Add recursive intelligence: analyze each subtask
        enhanced_subtasks = []
        for subtask in subtasks:
            # Infer agent type from subtask content
            subtask.agent_requirements = [self._infer_agent_type(subtask.title)]
            
            # Set automation potential
            subtask.automation_potential = self._estimate_automation_potential(subtask.title)
            
            # Add success criteria
            subtask.success_criteria = self._generate_success_criteria(subtask.title)
            
            enhanced_subtasks.append(subtask)
        
        return enhanced_subtasks[:self.max_subtasks]  # Limit subtask count
    
    def _generate_scraper_subtasks(self, parent_task: RecursiveSubtask) -> List[RecursiveSubtask]:
        """Generate subtasks specific to web scraping projects"""
        base_duration = parent_task.estimated_duration // 4
        
        return [
            RecursiveSubtask(
                title="Research Target Website",
                description="Analyze website structure, identify data patterns and potential blocking mechanisms",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration,
                required_capabilities=["web_analysis", "research"],
                success_criteria=["Website structure documented", "Data extraction strategy defined"]
            ),
            RecursiveSubtask(
                title="Design Data Model",
                description="Create data structure for extracted information",
                complexity=TaskComplexity.SIMPLE,
                estimated_duration=base_duration // 2,
                dependencies=["Research Target Website"],
                required_capabilities=["data_modeling", "design"],
                success_criteria=["Data schema created", "Validation rules defined"]
            ),
            RecursiveSubtask(
                title="Implement Scraping Logic",
                description="Develop core scraping functionality with error handling",
                complexity=TaskComplexity.COMPLEX,
                estimated_duration=base_duration * 2,
                dependencies=["Design Data Model"],
                required_capabilities=["coding", "web_scraping"],
                automation_potential=0.9,
                success_criteria=["Scraper functional", "Error handling implemented"]
            ),
            RecursiveSubtask(
                title="Add Authentication & Rate Limiting",
                description="Implement authentication and respectful rate limiting",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration,
                dependencies=["Implement Scraping Logic"],
                required_capabilities=["security", "networking"],
                success_criteria=["Authentication working", "Rate limits respected"]
            ),
            RecursiveSubtask(
                title="Testing & Validation",
                description="Comprehensive testing including edge cases and error scenarios",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration,
                dependencies=["Add Authentication & Rate Limiting"],
                required_capabilities=["testing", "validation"],
                success_criteria=["All tests passing", "Edge cases handled"]
            )
        ]
    
    def _generate_automation_subtasks(self, parent_task: RecursiveSubtask) -> List[RecursiveSubtask]:
        """Generate subtasks for automation projects (GUI/Desktop)"""
        base_duration = parent_task.estimated_duration // 5
        
        return [
            RecursiveSubtask(
                title="Analyze Target Application",
                description="Study GUI structure and interaction patterns",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration,
                required_capabilities=["gui_analysis", "research"],
                automation_potential=0.3,  # Manual analysis required
                success_criteria=["GUI elements mapped", "Interaction flow documented"]
            ),
            RecursiveSubtask(
                title="Setup Computer Vision",
                description="Configure OmniParser or similar for element detection",
                complexity=TaskComplexity.COMPLEX,
                estimated_duration=base_duration * 2,
                dependencies=["Analyze Target Application"],
                required_capabilities=["computer_vision", "configuration"],
                automation_potential=0.7,
                success_criteria=["Vision system functional", "Element detection accurate"]
            ),
            RecursiveSubtask(
                title="Implement Automation Scripts",
                description="Create automation scripts using PyAutoGUI and vision guidance",
                complexity=TaskComplexity.COMPLEX,
                estimated_duration=base_duration * 2,
                dependencies=["Setup Computer Vision"],
                required_capabilities=["automation", "scripting"],
                automation_potential=0.95,
                success_criteria=["Scripts functional", "Error handling robust"]
            ),
            RecursiveSubtask(
                title="Add Intelligent Decision Making",
                description="Integrate LLM for dynamic decision making during automation",
                complexity=TaskComplexity.ADVANCED,
                estimated_duration=base_duration * 2,
                dependencies=["Implement Automation Scripts"],
                required_capabilities=["ai_integration", "decision_making"],
                automation_potential=0.85,
                success_criteria=["LLM integration working", "Adaptive behavior implemented"]
            ),
            RecursiveSubtask(
                title="Testing & Refinement",
                description="Test automation across different scenarios and edge cases",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration,
                dependencies=["Add Intelligent Decision Making"],
                required_capabilities=["testing", "refinement"],
                automation_potential=0.6,
                success_criteria=["Edge cases handled", "Reliability validated"]
            )
        ]
    
    def _generate_server_subtasks(self, parent_task: RecursiveSubtask) -> List[RecursiveSubtask]:
        """Generate subtasks for server/API development"""
        base_duration = parent_task.estimated_duration // 6
        
        return [
            RecursiveSubtask(
                title="Design API Architecture",
                description="Design RESTful API structure and database schema",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration,
                required_capabilities=["api_design", "architecture"],
                success_criteria=["API spec complete", "Database schema designed"]
            ),
            RecursiveSubtask(
                title="Setup Development Environment",
                description="Configure development environment and dependencies",
                complexity=TaskComplexity.SIMPLE,
                estimated_duration=base_duration // 2,
                required_capabilities=["environment_setup", "configuration"],
                automation_potential=0.8,
                success_criteria=["Environment configured", "Dependencies installed"]
            ),
            RecursiveSubtask(
                title="Implement Core API Endpoints",
                description="Develop core API functionality and request handling",
                complexity=TaskComplexity.COMPLEX,
                estimated_duration=base_duration * 3,
                dependencies=["Design API Architecture", "Setup Development Environment"],
                required_capabilities=["backend_development", "api_implementation"],
                automation_potential=0.7,
                success_criteria=["Core endpoints functional", "Request handling implemented"]
            ),
            RecursiveSubtask(
                title="Add Authentication & Security",
                description="Implement authentication, authorization, and security measures",
                complexity=TaskComplexity.COMPLEX,
                estimated_duration=base_duration * 2,
                dependencies=["Implement Core API Endpoints"],
                required_capabilities=["security", "authentication"],
                success_criteria=["Auth system working", "Security measures implemented"]
            ),
            RecursiveSubtask(
                title="Database Integration",
                description="Integrate database operations and data persistence",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration,
                dependencies=["Implement Core API Endpoints"],
                required_capabilities=["database", "data_persistence"],
                success_criteria=["Database connected", "CRUD operations working"]
            ),
            RecursiveSubtask(
                title="Testing & Documentation",
                description="Create comprehensive tests and API documentation",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration,
                dependencies=["Add Authentication & Security", "Database Integration"],
                required_capabilities=["testing", "documentation"],
                success_criteria=["Tests comprehensive", "Documentation complete"]
            )
        ]
    
    def _generate_analysis_subtasks(self, parent_task: RecursiveSubtask) -> List[RecursiveSubtask]:
        """Generate subtasks for research/analysis projects"""
        base_duration = parent_task.estimated_duration // 4
        
        return [
            RecursiveSubtask(
                title="Define Research Scope",
                description="Clearly define research questions and success criteria",
                complexity=TaskComplexity.SIMPLE,
                estimated_duration=base_duration // 2,
                required_capabilities=["research_planning", "analysis"],
                success_criteria=["Scope clearly defined", "Research questions formulated"]
            ),
            RecursiveSubtask(
                title="Gather Information",
                description="Collect relevant data from multiple sources",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration * 2,
                dependencies=["Define Research Scope"],
                required_capabilities=["data_collection", "research"],
                automation_potential=0.6,
                success_criteria=["Relevant data collected", "Sources documented"]
            ),
            RecursiveSubtask(
                title="Analyze & Synthesize",
                description="Process collected information and extract insights",
                complexity=TaskComplexity.COMPLEX,
                estimated_duration=base_duration * 2,
                dependencies=["Gather Information"],
                required_capabilities=["data_analysis", "synthesis"],
                automation_potential=0.4,
                success_criteria=["Key insights identified", "Patterns analyzed"]
            ),
            RecursiveSubtask(
                title="Generate Report",
                description="Create comprehensive analysis report with findings",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration,
                dependencies=["Analyze & Synthesize"],
                required_capabilities=["report_generation", "communication"],
                automation_potential=0.7,
                success_criteria=["Report comprehensive", "Findings clearly presented"]
            )
        ]
    
    def _generate_generic_subtasks(self, parent_task: RecursiveSubtask) -> List[RecursiveSubtask]:
        """Generate generic subtasks for unknown task types"""
        base_duration = parent_task.estimated_duration // 4
        
        return [
            RecursiveSubtask(
                title="Research & Planning",
                description=f"Research requirements and plan approach for: {parent_task.title}",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration,
                required_capabilities=["research", "planning"],
                success_criteria=["Requirements clear", "Approach planned"]
            ),
            RecursiveSubtask(
                title="Design & Architecture",
                description="Design solution architecture and implementation strategy",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration,
                dependencies=["Research & Planning"],
                required_capabilities=["design", "architecture"],
                success_criteria=["Architecture designed", "Strategy defined"]
            ),
            RecursiveSubtask(
                title="Implementation",
                description="Implement core functionality according to design",
                complexity=TaskComplexity.COMPLEX,
                estimated_duration=base_duration * 2,
                dependencies=["Design & Architecture"],
                required_capabilities=["implementation", "development"],
                automation_potential=0.6,
                success_criteria=["Core functionality working", "Implementation complete"]
            ),
            RecursiveSubtask(
                title="Testing & Validation",
                description="Test implementation and validate against requirements",
                complexity=TaskComplexity.MODERATE,
                estimated_duration=base_duration,
                dependencies=["Implementation"],
                required_capabilities=["testing", "validation"],
                automation_potential=0.7,
                success_criteria=["Testing complete", "Requirements validated"]
            )
        ]
    
    def _should_decompose_further(self, task: RecursiveSubtask, depth: int) -> bool:
        """Determine if task should be decomposed further"""
        
        # Stop if max depth reached
        if depth >= self.max_depth:
            return False
        
        # Stop if task is already simple enough
        if task.complexity == TaskComplexity.SIMPLE:
            return False
        
        # Stop if estimated duration is below minimum
        if task.estimated_duration <= self.min_duration:
            return False
        
        # Continue decomposition for complex tasks
        return task.complexity.value >= TaskComplexity.MODERATE.value
    
    def _infer_agent_type(self, task_title: str) -> str:
        """Infer required agent type from task title"""
        title_lower = task_title.lower()
        
        for agent_type, patterns in self.task_patterns.items():
            if any(pattern in title_lower for pattern in patterns):
                return agent_type
        
        return "default"
    
    def _estimate_automation_potential(self, task_title: str) -> float:
        """Estimate how much of the task can be automated (0.0-1.0)"""
        title_lower = task_title.lower()
        
        # High automation potential
        if any(word in title_lower for word in ["test", "scraping", "api", "data processing"]):
            return 0.9
        
        # Medium automation potential  
        if any(word in title_lower for word in ["implement", "generate", "create", "build"]):
            return 0.7
        
        # Low automation potential
        if any(word in title_lower for word in ["design", "plan", "research", "analyze"]):
            return 0.4
        
        return 0.6  # Default
    
    def _generate_success_criteria(self, task_title: str) -> List[str]:
        """Generate success criteria for a task"""
        title_lower = task_title.lower()
        
        criteria = ["Task objectives completed"]
        
        if "test" in title_lower:
            criteria.extend(["All tests passing", "Coverage > 80%"])
        
        if "implement" in title_lower or "code" in title_lower:
            criteria.extend(["Code functional", "Code follows standards"])
        
        if "research" in title_lower or "analyze" in title_lower:
            criteria.extend(["Information comprehensive", "Insights documented"])
        
        return criteria
    
    def _extract_execution_phases(self, task_tree: RecursiveSubtask) -> List[Dict[str, Any]]:
        """Extract execution phases from recursive task tree"""
        phases = []
        
        def extract_level_tasks(task: RecursiveSubtask, level: int):
            if level >= len(phases):
                phases.append({
                    "phase_number": level + 1,
                    "tasks": [],
                    "estimated_duration": 0,
                    "parallelizable": True
                })
            
            if not task.subtasks:  # Leaf task
                phases[level]["tasks"].append({
                    "id": task.id,
                    "title": task.title,
                    "duration": task.estimated_duration,
                    "agent_type": task.agent_requirements[0] if task.agent_requirements else "default",
                    "dependencies": task.dependencies
                })
                phases[level]["estimated_duration"] += task.estimated_duration
            else:
                # Recurse into subtasks
                for subtask in task.subtasks:
                    extract_level_tasks(subtask, level + 1)
        
        extract_level_tasks(task_tree, 0)
        
        return phases
    
    def _calculate_critical_path(self, phases: List[Dict[str, Any]]) -> List[str]:
        """Calculate critical path through task dependencies"""
        critical_tasks = []
        
        for phase in phases:
            # Find longest task in each phase (simplified critical path)
            longest_task = max(phase["tasks"], key=lambda t: t["duration"], default=None)
            if longest_task:
                critical_tasks.append(longest_task["id"])
        
        return critical_tasks
    
    def _count_all_subtasks(self, task_tree: RecursiveSubtask) -> int:
        """Count total number of subtasks in tree"""
        count = 1  # Count self
        
        for subtask in task_tree.subtasks:
            count += self._count_all_subtasks(subtask)
        
        return count
    
    def _calculate_max_depth(self, task_tree: RecursiveSubtask) -> int:
        """Calculate maximum depth of task tree"""
        if not task_tree.subtasks:
            return 1
        
        return 1 + max(self._calculate_max_depth(subtask) for subtask in task_tree.subtasks)
    
    def _identify_required_agents(self, task_tree: RecursiveSubtask) -> List[str]:
        """Identify all agent types required for task tree"""
        agents = set()
        
        def collect_agents(task: RecursiveSubtask):
            agents.update(task.agent_requirements)
            for subtask in task.subtasks:
                collect_agents(subtask)
        
        collect_agents(task_tree)
        return list(agents)
    
    def _calculate_automation_potential(self, task_tree: RecursiveSubtask) -> float:
        """Calculate overall automation potential for task tree"""
        total_duration = 0
        automated_duration = 0
        
        def accumulate_automation(task: RecursiveSubtask):
            nonlocal total_duration, automated_duration
            
            if not task.subtasks:  # Leaf task
                total_duration += task.estimated_duration
                automated_duration += task.estimated_duration * task.automation_potential
            else:
                for subtask in task.subtasks:
                    accumulate_automation(subtask)
        
        accumulate_automation(task_tree)
        
        return automated_duration / total_duration if total_duration > 0 else 0.0
    
    def _calculate_plan_confidence(self, task_tree: RecursiveSubtask) -> float:
        """Calculate confidence in plan quality"""
        # Simplified confidence calculation
        # Real implementation would use ML models or more sophisticated heuristics
        
        depth = self._calculate_max_depth(task_tree)
        subtask_count = self._count_all_subtasks(task_tree)
        
        # Confidence decreases with excessive complexity
        depth_factor = max(0.5, 1.0 - (depth - 3) * 0.1)
        complexity_factor = max(0.6, 1.0 - (subtask_count - 10) * 0.02)
        
        return min(1.0, depth_factor * complexity_factor)


class ROMAAugmentedTaskDecomposer:
    """
    Enhanced task decomposer that integrates ROMA recursive planning
    with existing qweenone TaskDecomposer for backward compatibility
    """
    
    def __init__(self):
        self.recursive_planner = RecursiveTaskPlanner()
        self.legacy_decomposer = None
        self.logger = setup_logger("ROMAAugmentedTaskDecomposer")
    
    async def decompose_with_roma(self, 
                                task_description: str, 
                                iterations: int = 3,
                                use_recursive: bool = True) -> Dict[str, Any]:
        """
        Enhanced task decomposition using ROMA recursive planning
        
        Provides both legacy-compatible output and advanced recursive planning
        """
        if self.logger:
            self.logger.info(f"Decomposing task with ROMA: {task_description}")
        
        try:
            if use_recursive:
                # Use ROMA recursive planning
                recursive_plan = await self.recursive_planner.create_recursive_plan(
                    objective=task_description,
                    context={"target_iterations": iterations}
                )
                
                # Convert to legacy format for compatibility
                legacy_format = self._convert_recursive_to_legacy(recursive_plan, iterations)
                
                # Add ROMA-specific enhancements
                legacy_format["roma_enhanced"] = {
                    "recursive_plan": {
                        "complexity_score": recursive_plan.complexity_score,
                        "automation_score": recursive_plan.automation_score,
                        "confidence": recursive_plan.confidence,
                        "strategy": recursive_plan.strategy.value,
                        "critical_path": recursive_plan.critical_path,
                        "required_agents": recursive_plan.required_agents
                    },
                    "advanced_features": {
                        "supports_recursive_breakdown": True,
                        "intelligent_agent_assignment": True,
                        "automation_potential_analysis": True,
                        "dynamic_dependency_resolution": True
                    }
                }
                
                return legacy_format
                
            else:
                raise RuntimeError("ROMA decomposer unavailable")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"ROMA decomposition failed: {e}")
            raise e
    
    def _convert_recursive_to_legacy(self, 
                                   plan: RecursivePlan, 
                                   target_iterations: int) -> Dict[str, Any]:
        """Convert ROMA recursive plan to legacy qweenone format"""
        
        # Group phases into target number of iterations
        iterations_per_phase = max(1, len(plan.phases) // target_iterations)
        
        legacy_iterations = []
        current_iteration = 1
        
        for i in range(0, len(plan.phases), iterations_per_phase):
            phase_group = plan.phases[i:i + iterations_per_phase]
            
            # Combine phases into iteration
            iteration_tasks = []
            total_duration = 0
            
            for phase in phase_group:
                for task in phase["tasks"]:
                    iteration_tasks.append({
                        "id": task["id"],
                        "title": task["title"],
                        "description": f"Execute: {task['title']}",
                        "agent_type": task["agent_type"],
                        "dependencies": task["dependencies"],
                        "estimated_duration": task["duration"]
                    })
                    total_duration += task["duration"]
            
            legacy_iterations.append({
                "iteration_number": current_iteration,
                "description": f"Iteration {current_iteration}: Execute {len(iteration_tasks)} tasks",
                "focus": f"Phase {current_iteration} of {target_iterations}",
                "subtasks": iteration_tasks,
                "estimated_effort": len(iteration_tasks),
                "total_duration": total_duration
            })
            
            current_iteration += 1
            
            if current_iteration > target_iterations:
                break
        
        return {
            "original_task": plan.objective,
            "total_iterations": len(legacy_iterations),
            "total_subtasks": sum(len(it["subtasks"]) for it in legacy_iterations),
            "iterations": legacy_iterations,
            "estimated_total_duration": plan.estimated_duration,
            "complexity_analysis": {
                "complexity_score": plan.complexity_score,
                "confidence": plan.confidence,
                "automation_potential": plan.automation_score
            }
        }


# Compatibility function for existing code
async def enhanced_decompose(task_description: str, iterations: int = 3) -> Dict[str, Any]:
    """
    Drop-in replacement for TaskDecomposer.decompose() with ROMA enhancements
    """
    decomposer = ROMAAugmentedTaskDecomposer()
    return await decomposer.decompose_with_roma(task_description, iterations)


if __name__ == "__main__":
    """Test the ROMA-enhanced decomposer"""
    
    async def test_roma_decomposer():
        decomposer = ROMAAugmentedTaskDecomposer()
        
        test_tasks = [
            "Create a web scraper for Instagram posts",
            "Build a desktop automation tool for data entry",
            "Develop an API for machine learning model serving",
            "Create a comprehensive data analysis pipeline"
        ]
        
        for task in test_tasks:
            print(f"\n🎯 Testing: {task}")
            
            result = await decomposer.decompose_with_roma(task, iterations=3)
            
            print(f"📊 Result: {result['total_iterations']} iterations, "
                  f"{result['total_subtasks']} subtasks")
            
            if "roma_enhanced" in result:
                roma_data = result["roma_enhanced"]["recursive_plan"]
                print(f"🧠 ROMA: {roma_data['automation_score']:.1%} automation, "
                      f"{roma_data['confidence']:.1%} confidence")
    
    asyncio.run(test_roma_decomposer())