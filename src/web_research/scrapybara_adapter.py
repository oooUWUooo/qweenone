#!/usr/bin/env python3

"""Scrapybara integration for qweenone.

This adapter keeps dependency optional and interacts with Scrapybara either through the
official Python SDK (if installed) or through the documented REST endpoints. The goal is
simple: allow agents to request remote computer-use sessions for deep scraping tasks and
retrieve summarized outputs back into the workflow.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

from src.agents.base_agent import AgentCapability, AgentResponse, BaseAgent, Task
from src.utils.logger import setup_logger

try:  # pragma: no cover - optional third‑party dependency
    from scrapybara import Scrapybara  # type: ignore

    SCRAPYBARA_SDK_AVAILABLE = True
except Exception:  # pragma: no cover
    Scrapybara = None  # type: ignore
    SCRAPYBARA_SDK_AVAILABLE = False


DEFAULT_API_BASE = "https://api.scrapybara.com"
DEFAULT_INSTANCE_TYPE = "ubuntu-browser"
DEFAULT_TIMEOUT = 600


@dataclass
class ScrapybaraConfig:
    api_key: Optional[str]
    api_base: str = DEFAULT_API_BASE
    default_instance: str = DEFAULT_INSTANCE_TYPE
    poll_interval: float = 2.0
    request_timeout: float = 30.0
    task_timeout: float = DEFAULT_TIMEOUT
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "ScrapybaraConfig":
        getenv = os.getenv
        api_key = getenv("SCRAPYBARA_API_KEY")
        enabled = bool(api_key and getenv("SCRAPYBARA_ENABLED", "1") not in {"0", "false"})
        return cls(
            api_key=api_key,
            api_base=getenv("SCRAPYBARA_API_BASE", DEFAULT_API_BASE),
            default_instance=getenv("SCRAPYBARA_INSTANCE", DEFAULT_INSTANCE_TYPE),
            poll_interval=float(getenv("SCRAPYBARA_POLL_INTERVAL", "2")),
            request_timeout=float(getenv("SCRAPYBARA_REQUEST_TIMEOUT", "30")),
            task_timeout=float(getenv("SCRAPYBARA_TASK_TIMEOUT", str(DEFAULT_TIMEOUT))),
            enabled=enabled,
        )


class ScrapybaraClient:
    """Thin wrapper around Scrapybara REST / SDK."""

    def __init__(self, config: ScrapybaraConfig):
        self.config = config
        self.logger = setup_logger("ScrapybaraClient")
        self.session = requests.Session()
        if config.api_key:
            self.session.headers.update({"Authorization": f"Bearer {config.api_key}"})
        if SCRAPYBARA_SDK_AVAILABLE and config.api_key:
            self.sdk: Optional[Scrapybara] = Scrapybara(api_key=config.api_key)
        else:  # pragma: no cover - optional path
            self.sdk = None

    def launch_and_run(self, instructions: str, instance_type: Optional[str] = None) -> Dict[str, Any]:
        if self.sdk:
            return self._launch_via_sdk(instructions, instance_type)
        return self._launch_via_rest(instructions, instance_type)

    # --- SDK path ---------------------------------------------------------
    def _launch_via_sdk(self, instructions: str, instance_type: Optional[str]) -> Dict[str, Any]:
        assert self.sdk is not None  # for mypy
        instance = self.sdk.instances.create(
            instance_type=instance_type or self.config.default_instance,
            instructions=instructions,
            timeout=self.config.task_timeout,
        )
        result = instance.wait(timeout=self.config.task_timeout)
        return {
            "instance_id": instance.id,
            "logs": result.logs,
            "artifacts": getattr(result, "artifacts", []),
            "status": result.status,
            "duration": result.duration,
        }

    # --- REST path --------------------------------------------------------
    def _launch_via_rest(self, instructions: str, instance_type: Optional[str]) -> Dict[str, Any]:
        base = self.config.api_base.rstrip("/")
        create_url = urljoin(base + "/", "v1/instances")
        payload = {
            "instance_type": instance_type or self.config.default_instance,
            "instructions": instructions,
            "timeout": self.config.task_timeout,
        }
        response = self.session.post(create_url, json=payload, timeout=self.config.request_timeout)
        response.raise_for_status()
        data = response.json()
        instance_id = data.get("id")
        status_url = urljoin(base + "/", f"v1/instances/{instance_id}")
        start_time = time.time()
        while True:
            status_resp = self.session.get(status_url, timeout=self.config.request_timeout)
            status_resp.raise_for_status()
            content = status_resp.json()
            if content.get("status") in {"completed", "failed", "timeout"}:
                return content
            if time.time() - start_time > self.config.task_timeout:
                raise TimeoutError(f"Scrapybara task timeout for instance {instance_id}")
            time.sleep(self.config.poll_interval)


class ScrapybaraAgent(BaseAgent):
    """Agent capable of launching Scrapybara computer-use jobs."""

    def __init__(self, name: str, description: str = "", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, description)
        self.agent_type = "scrapybara_operator"
        self.add_capability(AgentCapability.ANALYSIS)
        self.add_capability(AgentCapability.TASK_EXECUTION)
        cfg = ScrapybaraConfig.from_env()
        if config:
            params = config.get("parameters", config)
            for key, value in params.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        self.config = cfg
        self.logger = setup_logger("ScrapybaraAgent")
        self.client = ScrapybaraClient(cfg) if cfg.enabled else None

    async def execute_task(self, task: Task) -> AgentResponse:
        if not self.client:
            return AgentResponse(success=False, error="Scrapybara disabled or API key missing")
        instructions = self._build_instructions(task)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self.client.launch_and_run, instructions, None)
            return AgentResponse(success=True, data=result)
        except Exception as exc:  # pragma: no cover - runtime network errors
            self.logger.error(f"Scrapybara execution failed: {exc}")
            return AgentResponse(success=False, error=str(exc))

    def _build_instructions(self, task: Task) -> str:
        task_dict = task.to_dict() if isinstance(task, Task) else task
        lines = [f"Primary goal: {task_dict.get('description') or task_dict.get('title')}".strip()]
        params = task_dict.get("params", {})
        if params:
            lines.append("Parameters:")
            for key, value in params.items():
                lines.append(f"- {key}: {value}")
        constraints = task_dict.get("constraints", [])
        if constraints:
            lines.append("Constraints:")
            for item in constraints:
                lines.append(f"- {item}")
        expected = task_dict.get("expected_output")
        if expected:
            lines.append(f"Expected Output: {expected}")
        return "\n".join(filter(None, lines))


def register_scrapybara_agent(agent_builder: "AgentBuilder", config: Optional[ScrapybaraConfig] = None) -> bool:
    cfg = config or ScrapybaraConfig.from_env()
    if not cfg.enabled:
        return False
    agent_builder.register_agent_type(
        "scrapybara_operator",
        ScrapybaraAgent,
        default_capabilities=[
            AgentCapability.ANALYSIS.value,
            AgentCapability.TASK_EXECUTION.value,
        ],
        description="Scrapybara remote-computer operator",
    )
    return True


__all__ = ["ScrapybaraAgent", "ScrapybaraClient", "ScrapybaraConfig", "register_scrapybara_agent"]
