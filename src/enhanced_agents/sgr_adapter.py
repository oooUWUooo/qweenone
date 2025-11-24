#!/usr/bin/env python3

"""
SGR Agent Adapter

Bridge between qweenone agents and the `sgr-agent-core` Schema-Guided Reasoning stack.
The adapter keeps integration optional and isolated behind configuration flags so the
main system can operate without the SGR runtime.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

from src.agents.base_agent import AgentCapability, AgentResponse, BaseAgent, Task
from src.utils.logger import setup_logger

try:  # Optional import – sgr-agent-core might be installed as a package
    import sgr_agent_core  # type: ignore  # noqa: F401

    SGR_PACKAGE_AVAILABLE = True
except Exception:  # pragma: no cover - missing dependency is fine
    SGR_PACKAGE_AVAILABLE = False


@dataclass
class SGRConfig:
    """Configuration for SGR integration."""

    enabled: bool = False
    base_url: str = "http://localhost:8100"
    api_key: Optional[str] = None
    timeout: float = 60.0
    endpoint: str = "/api/v1/research"
    stream: bool = True

    @classmethod
    def from_env(cls) -> "SGRConfig":
        getenv = os.getenv
        enabled = getenv("SGR_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
        return cls(
            enabled=enabled,
            base_url=getenv("SGR_API_BASE", "http://localhost:8100"),
            api_key=getenv("SGR_API_KEY"),
            timeout=float(getenv("SGR_TIMEOUT", "60")),
            endpoint=getenv("SGR_API_ENDPOINT", "/api/v1/research"),
            stream=getenv("SGR_STREAM", "1").lower() in {"1", "true", "yes", "on"},
        )


class SGRClient:
    """Thin HTTP client for interacting with sgr-agent-core REST endpoints."""

    def __init__(self, config: SGRConfig):
        self.config = config
        self.session = requests.Session()
        self.logger = setup_logger("SGRClient")

    def run_research(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = urljoin(self.config.base_url.rstrip("/"), self.config.endpoint.lstrip("/"))
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        response = self.session.post(url, json=payload, headers=headers, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json()


class SGRAgent(BaseAgent):
    """Agent wrapper that proxies execution to the SGR research backend."""

    def __init__(self, name: str, description: str = "", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, description)
        self.agent_type = "sgr_researcher"
        self.add_capability(AgentCapability.ANALYSIS)
        self.add_capability(AgentCapability.TASK_EXECUTION)
        self.logger = setup_logger("SGRAgent")

        cfg = config or {}
        if isinstance(cfg, dict) and not isinstance(cfg, SGRConfig):
            cfg = cfg.get("parameters", cfg)
            merged = SGRConfig.from_env()
            for key, value in cfg.items():
                if hasattr(merged, key):
                    setattr(merged, key, value)
            self.config = merged
        elif isinstance(cfg, SGRConfig):
            self.config = cfg
        else:
            self.config = SGRConfig.from_env()

        self.client: Optional[SGRClient] = None
        if self.config.enabled:
            self.client = SGRClient(self.config)
        else:
            self.logger.info("SGR integration disabled via configuration")

    async def execute_task(self, task: Task) -> AgentResponse:
        if not self.client:
            return AgentResponse(success=False, error="SGR backend disabled")

        payload = self._build_payload(task)
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.client.run_research, payload)
            return AgentResponse(success=True, data=result)
        except Exception as exc:  # pragma: no cover - network errors runtime specific
            self.logger.error(f"SGR request failed: {exc}")
            return AgentResponse(success=False, error=str(exc))

    def _build_payload(self, task: Task) -> Dict[str, Any]:
        task_dict = task.to_dict() if isinstance(task, Task) else task
        prompt = task_dict.get("description") or task_dict.get("title", "")
        if not prompt:
            prompt = task_dict.get("context", "")
        metadata = {
            "task_id": task_dict.get("task_id"),
            "priority": task_dict.get("priority"),
            "params": task_dict.get("params", {}),
            "expected_output": task_dict.get("expected_output"),
        }
        return {
            "query": prompt,
            "metadata": metadata,
            "options": {
                "stream": self.config.stream,
                "source": "qweenone",
                "agent_type": self.agent_type,
            },
        }


def register_sgr_agent(agent_builder: "AgentBuilder", config: Optional[SGRConfig] = None) -> bool:
    """Register SGR agent type in AgentBuilder if integration enabled."""

    cfg = config or SGRConfig.from_env()
    if not cfg.enabled:
        return False

    agent_builder.register_agent_type(
        "sgr_researcher",
        SGRAgent,
        default_capabilities=[AgentCapability.ANALYSIS.value, AgentCapability.TASK_EXECUTION.value],
        description="Schema-Guided Reasoning research agent",
    )
    return True


__all__ = ["SGRConfig", "SGRClient", "SGRAgent", "register_sgr_agent"]
