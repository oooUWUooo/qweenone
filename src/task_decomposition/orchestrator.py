from typing import Any, Dict, List
import asyncio

from .roma_decomposer import ROMAAugmentedTaskDecomposer

try:
    from ..enhanced_agents.orchestra_integration import AgentOrchestra
except ImportError:
    AgentOrchestra = None


try:
    from ..enhanced_agents.sgr_adapter import SGRConfig
except ImportError:  # pragma: no cover - optional dependency
    SGRConfig = None

try:
    from ..web_research.scrapybara_adapter import ScrapybaraConfig
except ImportError:  # pragma: no cover - optional dependency
    ScrapybaraConfig = None


class TaskDecompositionOrchestrator:
    def __init__(self, enable_orchestra: bool = True, use_sgr: bool = False, sgr_config: SGRConfig | None = None, use_scrapybara: bool = False, scrapybara_config: ScrapybaraConfig | None = None) -> None:
        self.decomposer = ROMAAugmentedTaskDecomposer()
        self.enable_orchestra = enable_orchestra and AgentOrchestra is not None
        self.agent_orchestra = AgentOrchestra() if self.enable_orchestra else None
        self.use_sgr = bool(use_sgr and SGRConfig is not None)
        self.sgr_config = sgr_config or (SGRConfig.from_env() if self.use_sgr and SGRConfig else None)
        if self.use_sgr and self.sgr_config and not self.sgr_config.enabled:
            self.use_sgr = False

        self.use_scrapybara = bool(use_scrapybara and ScrapybaraConfig is not None)
        self.scrapybara_config = scrapybara_config or (
            ScrapybaraConfig.from_env() if self.use_scrapybara and ScrapybaraConfig else None
        )
        if self.use_scrapybara and self.scrapybara_config and not self.scrapybara_config.enabled:
            self.use_scrapybara = False

    async def create_plan(self, task_description: str, iterations: int = 3) -> Dict[str, Any]:
        decomposition = await self.decomposer.decompose_with_roma(task_description, iterations, use_recursive=True)
        agent_teams = []
        if self.agent_orchestra:
            for iteration in decomposition.get("iterations", []):
                capabilities = self._infer_capabilities(iteration.get("subtasks", []))
                if not capabilities:
                    capabilities = ["general"]
                team = self.agent_orchestra.create_agent_team(
                    team_name=f"Iteration{iteration['iteration_number']}",
                    task_objective=iteration.get("description", task_description),
                    required_capabilities=capabilities
                )
                agent_teams.append({
                    "iteration_number": iteration["iteration_number"],
                    "agents": self._serialize_team(team)
                })
        plan = {**decomposition}
        plan["agent_teams"] = agent_teams
        plan["automation_focus"] = self._calculate_automation_focus(decomposition)
        plan["sgr_assignments"] = self._build_sgr_assignments(decomposition, task_description) if self.use_sgr else []
        plan["scrapybara_jobs"] = (
            self._build_scrapybara_jobs(decomposition, task_description)
            if self.use_scrapybara
            else []
        )
        return plan

    def create_plan_sync(self, task_description: str, iterations: int = 3) -> Dict[str, Any]:
        return asyncio.run(self.create_plan(task_description, iterations))

    def _infer_capabilities(self, subtasks: List[Dict[str, Any]]) -> List[str]:
        capabilities = set()
        for subtask in subtasks:
            title = subtask.get("title", "").lower()
            description = subtask.get("description", "").lower()
            text = f"{title} {description}"
            if any(keyword in text for keyword in ["test", "validate", "qa", "quality"]):
                capabilities.add("testing")
            if any(keyword in text for keyword in ["code", "implement", "build", "develop"]):
                capabilities.add("development")
            if any(keyword in text for keyword in ["research", "analyze", "investigate"]):
                capabilities.add("research")
            if any(keyword in text for keyword in ["automation", "gui", "desktop", "pyautogui"]):
                capabilities.add("desktop_automation")
            if any(keyword in text for keyword in ["browser", "web", "playwright", "scrape"]):
                capabilities.add("browser_automation")
            if any(keyword in text for keyword in ["scrape", "crawler", "scraping", "dataset", "harvest"]):
                capabilities.add("scraping")
            if any(keyword in text for keyword in ["llm", "prompt", "routing", "api"]):
                capabilities.add("llm_routing")
            if any(keyword in text for keyword in ["plan", "strategy", "architecture", "design"]):
                capabilities.add("planning")
        return sorted(capabilities)

    def _serialize_team(self, team: Dict[str, Any]) -> List[Dict[str, Any]]:
        serialized = []
        for agent in team.values():
            serialized.append({
                "id": agent.id,
                "name": agent.name,
                "role": agent.role.value,
                "capabilities": list(agent.capabilities)
            })
        return serialized

    def _calculate_automation_focus(self, decomposition: Dict[str, Any]) -> Dict[str, Any]:
        roma_data = decomposition.get("roma_enhanced", {})
        recursive_plan = roma_data.get("recursive_plan", {})
        automation_score = recursive_plan.get("automation_score", 0.0)
        confidence = recursive_plan.get("confidence", 0.0)
        iterations = decomposition.get("iterations", [])
        automation_iterations = []
        for iteration in iterations:
            hit = any("automation" in sub.get("title", "").lower() or "gui" in sub.get("title", "").lower() for sub in iteration.get("subtasks", []))
            if hit:
                automation_iterations.append(iteration["iteration_number"])
        return {
            "automation_score": automation_score,
            "confidence": confidence,
            "automation_iterations": automation_iterations,
            "high_priority": automation_score >= 0.7
        }

    def _build_sgr_assignments(self, decomposition: Dict[str, Any], fallback_prompt: str) -> List[Dict[str, Any]]:
        if not self.sgr_config:
            return []
        assignments: List[Dict[str, Any]] = []
        for iteration in decomposition.get("iterations", []):
            subtasks = iteration.get("subtasks", [])
            tags = self._infer_capabilities(subtasks)
            if "research" not in tags:
                continue
            prompt = self._compose_research_prompt(iteration, fallback_prompt)
            assignments.append({
                "iteration_number": iteration.get("iteration_number"),
                "prompt": prompt,
                "subtasks": [sub.get("title") for sub in subtasks],
                "config": {
                    "endpoint": self.sgr_config.endpoint,
                    "stream": self.sgr_config.stream,
                    "timeout": self.sgr_config.timeout,
                },
            })
        return assignments

    def _compose_research_prompt(self, iteration: Dict[str, Any], fallback_prompt: str) -> str:
        prefix = iteration.get("description") or fallback_prompt
        lines = [prefix.strip()]
        for subtask in iteration.get("subtasks", [])[:5]:
            title = subtask.get("title", "").strip()
            if title:
                lines.append(f"- {title}")
        return "\n".join(lines)

    def _build_scrapybara_jobs(self, decomposition: Dict[str, Any], fallback_prompt: str) -> List[Dict[str, Any]]:
        if not self.scrapybara_config:
            return []
        jobs: List[Dict[str, Any]] = []
        for iteration in decomposition.get("iterations", []):
            subtasks = iteration.get("subtasks", [])
            tags = self._infer_capabilities(subtasks)
            if "scraping" not in tags and "browser_automation" not in tags:
                continue
            instructions = self._compose_research_prompt(iteration, fallback_prompt)
            jobs.append({
                "iteration_number": iteration.get("iteration_number"),
                "instructions": instructions,
                "instance": self.scrapybara_config.default_instance,
                "timeout": self.scrapybara_config.task_timeout,
            })
        return jobs
