"""
Browser Automation System

Exports:
- PlaywrightAutomation: Modern browser automation
- BrowserAutomationAgent: Agent system integration
- Configuration and helpers
"""

from .playwright_automation import (
    PlaywrightAutomation,
    BrowserAutomationAgent,
    BrowserConfig,
    WebAction,
    quick_browser_automation
)

__all__ = [
    "PlaywrightAutomation",
    "BrowserAutomationAgent",
    "BrowserConfig",
    "WebAction",
    "quick_browser_automation"
]
