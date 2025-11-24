from typing import Any, Dict
import asyncio

from ..task_decomposition.orchestrator import TaskDecompositionOrchestrator


class TaskDecomposer:
    def __init__(self, default_iterations: int = 3) -> None:
        self.default_iterations = default_iterations
        self.orchestrator = TaskDecompositionOrchestrator(enable_orchestra=True)

    async def decompose_async(self, task_description: str, iterations: int = None) -> Dict[str, Any]:
        target_iterations = iterations if iterations is not None else self.default_iterations
        return await self.orchestrator.create_plan(task_description, target_iterations)

    def decompose(self, task_description: str, iterations: int = None) -> Dict[str, Any]:
        target_iterations = iterations if iterations is not None else self.default_iterations
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("Use decompose_async() within an active event loop")
        return asyncio.run(self.decompose_async(task_description, target_iterations))
