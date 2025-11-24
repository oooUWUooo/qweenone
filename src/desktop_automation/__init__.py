"""
Desktop Automation System

Exports:
- OmniDesktopAutomation: Vision-based desktop automation
- DesktopAutomationAgent: Agent system integration
- Helper functions
"""

from .omni_automation import (
    OmniDesktopAutomation,
    DesktopAutomationAgent,
    GUIElement,
    AutomationStep,
    execute_desktop_automation
)

__all__ = [
    "OmniDesktopAutomation",
    "DesktopAutomationAgent",
    "GUIElement",
    "AutomationStep",
    "execute_desktop_automation"
]
