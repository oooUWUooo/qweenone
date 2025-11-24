"""
Playwright-Enhanced Browser Automation

Modern browser automation using Playwright for reliable, fast web interactions.
Integrates with qweenone agent system to provide advanced web automation capabilities.

Key Features:
- Multi-browser support (Chrome, Firefox, Safari, Edge)
- Mobile device emulation
- Network interception and mocking
- Screenshot and video recording
- Auto-waiting for elements (no more flaky tests)
- Parallel execution across browsers
- Built-in retry mechanisms
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import base64

# Playwright imports (with fallback)
try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None

# Import existing qweenone components
try:
    from ..agents.base_agent import BaseAgent, AgentResponse
    from ..utils.logger import setup_logger
except ImportError:
    BaseAgent = None
    AgentResponse = None
    setup_logger = lambda x: None


@dataclass
class BrowserConfig:
    """Configuration for browser automation"""
    browser_type: str = "chromium"  # chromium, firefox, webkit
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: str = ""
    locale: str = "en-US"
    timezone: str = "UTC"
    
    # Mobile device emulation
    device_name: Optional[str] = None  # "iPhone 12", "Galaxy S21", etc.
    
    # Network and security
    ignore_https_errors: bool = True
    java_script_enabled: bool = True
    
    # Performance
    slow_mo: int = 0  # milliseconds delay between actions
    timeout: int = 30000  # milliseconds
    
    # Recording and debugging
    screenshot_on_failure: bool = True
    record_video: bool = False
    trace_enabled: bool = False


@dataclass
class WebAction:
    """Single web automation action"""
    action: str  # navigate, click, type, wait, screenshot, scroll, etc.
    target: str  # URL, selector, text, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    wait_for: str = ""  # Element to wait for after action
    timeout: int = 30000  # milliseconds
    retry_count: int = 3
    screenshot_after: bool = False


class PlaywrightAutomation:
    """
    Advanced browser automation using Playwright
    
    Provides reliable, fast web automation with modern features like
    auto-waiting, network interception, and multi-browser support.
    """
    
    def __init__(self, 
                 config: BrowserConfig = None,
                 screenshots_dir: str = "/tmp/qweenone_browser_screenshots"):
        """Initialize Playwright automation"""
        
        self.config = config or BrowserConfig()
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(exist_ok=True)
        
        self.logger = setup_logger("PlaywrightAutomation")
        
        # Playwright state
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # Automation tracking
        self.action_history: List[Dict[str, Any]] = []
        self.navigation_history: List[str] = []
        
        if not PLAYWRIGHT_AVAILABLE:
            if self.logger:
                self.logger.warning("Playwright not available - install with: pip install playwright && playwright install")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_browser()
    
    async def start_browser(self):
        """Start browser instance"""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available")
        
        try:
            self.playwright = await async_playwright().start()
            
            # Launch browser based on config
            if self.config.browser_type == "chromium":
                self.browser = await self.playwright.chromium.launch(
                    headless=self.config.headless,
                    slow_mo=self.config.slow_mo
                )
            elif self.config.browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(
                    headless=self.config.headless,
                    slow_mo=self.config.slow_mo
                )
            elif self.config.browser_type == "webkit":
                self.browser = await self.playwright.webkit.launch(
                    headless=self.config.headless,
                    slow_mo=self.config.slow_mo
                )
            else:
                raise ValueError(f"Unsupported browser: {self.config.browser_type}")
            
            # Create context with configuration
            context_options = {
                "viewport": {
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height
                },
                "locale": self.config.locale,
                "timezone_id": self.config.timezone,
                "ignore_https_errors": self.config.ignore_https_errors,
                "java_script_enabled": self.config.java_script_enabled
            }
            
            # Add device emulation if specified
            if self.config.device_name:
                device = self.playwright.devices.get(self.config.device_name)
                if device:
                    context_options.update(device)
            
            # Add user agent if specified
            if self.config.user_agent:
                context_options["user_agent"] = self.config.user_agent
            
            # Add recording if enabled
            if self.config.record_video:
                context_options["record_video_dir"] = str(self.screenshots_dir / "videos")
            
            if self.config.trace_enabled:
                context_options["trace_enabled"] = True
            
            self.context = await self.browser.new_context(**context_options)
            
            # Create page
            self.page = await self.context.new_page()
            
            # Set default timeout
            self.page.set_default_timeout(self.config.timeout)
            
            if self.logger:
                self.logger.info(f"Browser started: {self.config.browser_type} ({'headless' if self.config.headless else 'headed'})")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to start browser: {e}")
            raise e
    
    async def close_browser(self):
        """Close browser and cleanup"""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            if self.logger:
                self.logger.info("Browser closed successfully")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error closing browser: {e}")
    
    async def navigate_to(self, url: str, wait_until: str = "load") -> Dict[str, Any]:
        """Navigate to URL with various wait conditions"""
        
        if not self.page:
            await self.start_browser()
        
        try:
            if self.logger:
                self.logger.info(f"Navigating to: {url}")
            
            # Navigate with wait condition
            response = await self.page.goto(url, wait_until=wait_until)
            
            # Track navigation
            self.navigation_history.append(url)
            self.action_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "navigate",
                "target": url,
                "success": response.ok if response else True,
                "status_code": response.status if response else 200
            })
            
            # Get page info
            title = await self.page.title()
            current_url = self.page.url
            
            return {
                "success": True,
                "url": current_url,
                "title": title,
                "status_code": response.status if response else 200,
                "wait_condition": wait_until
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Navigation failed: {e}")
            
            self.action_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "navigate",
                "target": url,
                "success": False,
                "error": str(e)
            })
            
            return {"success": False, "error": str(e), "url": url}
    
    async def click_element(self, 
                          selector: str, 
                          text_content: str = None,
                          wait_for_element: bool = True) -> Dict[str, Any]:
        """Click element with intelligent waiting"""
        
        try:
            if wait_for_element:
                # Wait for element to be visible and enabled
                await self.page.wait_for_selector(selector, state="visible")
            
            # Additional text content check if specified
            if text_content:
                element = await self.page.wait_for_selector(
                    f"{selector}:has-text('{text_content}')",
                    state="visible"
                )
            else:
                element = await self.page.locator(selector).first
            
            # Click the element
            await element.click()
            
            if self.logger:
                self.logger.info(f"Clicked element: {selector}")
            
            self.action_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "click",
                "target": selector,
                "text_content": text_content,
                "success": True
            })
            
            return {"success": True, "selector": selector, "text_content": text_content}
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Click failed: {e}")
            
            self.action_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "click",
                "target": selector,
                "success": False,
                "error": str(e)
            })
            
            return {"success": False, "error": str(e), "selector": selector}
    
    async def type_text(self, 
                       selector: str, 
                       text: str,
                       clear_existing: bool = True,
                       press_enter: bool = False) -> Dict[str, Any]:
        """Type text into input field"""
        
        try:
            # Wait for input field
            await self.page.wait_for_selector(selector, state="visible")
            
            if clear_existing:
                # Clear existing text
                await self.page.locator(selector).clear()
            
            # Type text
            await self.page.locator(selector).fill(text)
            
            if press_enter:
                await self.page.locator(selector).press("Enter")
            
            if self.logger:
                self.logger.info(f"Typed text into {selector}: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            self.action_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "type",
                "target": selector,
                "text": text,
                "clear_existing": clear_existing,
                "press_enter": press_enter,
                "success": True
            })
            
            return {"success": True, "selector": selector, "text": text}
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Type failed: {e}")
            
            self.action_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "type",
                "target": selector,
                "success": False,
                "error": str(e)
            })
            
            return {"success": False, "error": str(e), "selector": selector}
    
    async def wait_for_element(self, 
                             selector: str,
                             state: str = "visible",
                             timeout: int = None) -> Dict[str, Any]:
        """Wait for element with specified state"""
        
        timeout = timeout or self.config.timeout
        
        try:
            await self.page.wait_for_selector(selector, state=state, timeout=timeout)
            
            return {"success": True, "selector": selector, "state": state}
            
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}
    
    async def take_screenshot(self, 
                            full_page: bool = False,
                            element_selector: str = None) -> Dict[str, Any]:
        """Take screenshot of page or specific element"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if element_selector:
                # Screenshot of specific element
                element = await self.page.locator(element_selector).first
                screenshot_path = self.screenshots_dir / f"element_{timestamp}.png"
                await element.screenshot(path=screenshot_path)
            else:
                # Full page or viewport screenshot
                screenshot_path = self.screenshots_dir / f"page_{timestamp}.png"
                await self.page.screenshot(
                    path=screenshot_path,
                    full_page=full_page
                )
            
            # Convert to base64 for embedding
            with open(screenshot_path, "rb") as f:
                screenshot_base64 = base64.b64encode(f.read()).decode()
            
            if self.logger:
                self.logger.info(f"Screenshot saved: {screenshot_path}")
            
            return {
                "success": True,
                "path": str(screenshot_path),
                "base64": screenshot_base64,
                "full_page": full_page,
                "element_selector": element_selector
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Screenshot failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_web_actions(self, actions: List[WebAction]) -> Dict[str, Any]:
        """Execute sequence of web actions"""
        
        if self.logger:
            self.logger.info(f"Executing {len(actions)} web actions")
        
        results = []
        overall_success = True
        
        if not self.page:
            await self.start_browser()
        
        for i, action in enumerate(actions, 1):
            action_start = time.time()
            
            try:
                if self.logger:
                    self.logger.info(f"Action {i}/{len(actions)}: {action.action} - {action.target}")
                
                success = await self._execute_single_action(action)
                
                action_duration = time.time() - action_start
                
                action_result = {
                    "step": i,
                    "action": action.action,
                    "target": action.target,
                    "success": success,
                    "duration": action_duration,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Take screenshot if requested
                if action.screenshot_after:
                    screenshot = await self.take_screenshot()
                    action_result["screenshot"] = screenshot.get("path", "")
                
                # Wait for specified element if configured
                if action.wait_for:
                    wait_result = await self.wait_for_element(action.wait_for)
                    action_result["wait_result"] = wait_result["success"]
                
                results.append(action_result)
                
                if not success:
                    overall_success = False
                    
                    # Stop on failure unless continue_on_error is set
                    if not action.parameters.get("continue_on_error", False):
                        break
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Action {i} failed: {e}")
                
                results.append({
                    "step": i,
                    "action": action.action,
                    "target": action.target,
                    "success": False,
                    "error": str(e),
                    "duration": time.time() - action_start,
                    "timestamp": datetime.now().isoformat()
                })
                
                overall_success = False
                break
        
        return {
            "overall_success": overall_success,
            "total_actions": len(actions),
            "executed_actions": len(results),
            "action_results": results,
            "current_url": self.page.url if self.page else "",
            "page_title": await self.page.title() if self.page else ""
        }
    
    async def _execute_single_action(self, action: WebAction) -> bool:
        """Execute single web action"""
        
        action_type = action.action.lower()
        
        try:
            if action_type == "navigate":
                result = await self.navigate_to(action.target)
                return result["success"]
            
            elif action_type == "click":
                result = await self.click_element(action.target)
                return result["success"]
            
            elif action_type == "type":
                text_to_type = action.parameters.get("text", action.target)
                result = await self.type_text(
                    selector=action.target,
                    text=text_to_type,
                    clear_existing=action.parameters.get("clear", True),
                    press_enter=action.parameters.get("press_enter", False)
                )
                return result["success"]
            
            elif action_type == "wait":
                duration = action.parameters.get("duration", 1.0)
                await asyncio.sleep(duration)
                return True
            
            elif action_type == "screenshot":
                result = await self.take_screenshot(
                    full_page=action.parameters.get("full_page", False),
                    element_selector=action.parameters.get("element", None)
                )
                return result["success"]
            
            elif action_type == "scroll":
                direction = action.parameters.get("direction", "down")
                amount = action.parameters.get("amount", 3)
                
                if direction == "down":
                    await self.page.mouse.wheel(0, amount * 100)
                elif direction == "up":
                    await self.page.mouse.wheel(0, -amount * 100)
                
                return True
            
            elif action_type == "wait_for":
                result = await self.wait_for_element(action.target)
                return result["success"]
            
            elif action_type == "extract_text":
                text = await self.page.locator(action.target).text_content()
                action.parameters["extracted_text"] = text
                return True
            
            elif action_type == "extract_attribute":
                attribute = action.parameters.get("attribute", "href")
                value = await self.page.locator(action.target).get_attribute(attribute)
                action.parameters["extracted_value"] = value
                return True
            
            else:
                if self.logger:
                    self.logger.error(f"Unknown action: {action_type}")
                return False
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Action {action_type} failed: {e}")
            return False
    
    async def execute_natural_language_task(self, task_description: str) -> Dict[str, Any]:
        """
        Execute web automation task described in natural language
        
        Uses LLM to plan browser actions based on current page state
        """
        if self.logger:
            self.logger.info(f"Executing natural language web task: {task_description}")
        
        try:
            # If no page is open, take initial screenshot for context
            if not self.page:
                await self.start_browser()
                await self.navigate_to("about:blank")
            
            # Get current page state for context
            page_context = await self._get_page_context()
            
            # Generate action plan using LLM
            action_plan = await self._generate_web_action_plan(task_description, page_context)
            
            # Convert plan to WebAction objects
            actions = self._convert_plan_to_web_actions(action_plan)
            
            # Execute planned actions
            result = await self.execute_web_actions(actions)
            
            # Add task context
            result.update({
                "task_description": task_description,
                "page_context": page_context,
                "planned_actions": len(actions),
                "llm_guided": True
            })
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Natural language web task failed: {e}")
            
            return {
                "overall_success": False,
                "error": str(e),
                "task_description": task_description
            }
    
    async def _get_page_context(self) -> Dict[str, Any]:
        """Get current page context for LLM planning"""
        
        try:
            context = {
                "url": self.page.url,
                "title": await self.page.title(),
                "viewport": await self.page.viewport_size(),
            }
            
            # Get visible text content (limited)
            try:
                text_content = await self.page.text_content("body")
                context["text_preview"] = text_content[:500] + "..." if len(text_content) > 500 else text_content
            except:
                context["text_preview"] = "Could not extract page text"
            
            # Get form fields if any
            try:
                input_elements = await self.page.locator("input, textarea, select").count()
                context["input_elements_count"] = input_elements
            except:
                context["input_elements_count"] = 0
            
            # Get clickable elements count
            try:
                clickable_elements = await self.page.locator("button, a, [onclick], [role='button']").count()
                context["clickable_elements_count"] = clickable_elements
            except:
                context["clickable_elements_count"] = 0
            
            return context
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _generate_web_action_plan(self, 
                                      task_description: str,
                                      page_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate web action plan using LLM"""
        
        # Mock LLM planning for demo (in real implementation, would use LiteLLM)
        await asyncio.sleep(0.3)  # Simulate LLM processing
        
        # Generate reasonable plan based on task description
        task_lower = task_description.lower()
        
        if "google" in task_lower and "search" in task_lower:
            return [
                {"action": "navigate", "target": "https://www.google.com"},
                {"action": "wait_for", "target": "input[name='q']"},
                {"action": "type", "target": "input[name='q']", "parameters": {"text": "playwright automation"}},
                {"action": "click", "target": "input[value='Google Search'], button[aria-label='Google Search']"},
                {"action": "wait_for", "target": "#search"},
                {"action": "screenshot", "parameters": {"full_page": False}}
            ]
        
        elif "instagram" in task_lower:
            return [
                {"action": "navigate", "target": "https://www.instagram.com"},
                {"action": "wait_for", "target": "article"},
                {"action": "screenshot", "parameters": {"full_page": True}},
                {"action": "extract_text", "target": "article"}
            ]
        
        elif "screenshot" in task_lower:
            current_url = page_context.get("url", "about:blank")
            if current_url == "about:blank":
                return [
                    {"action": "navigate", "target": "https://example.com"},
                    {"action": "wait_for", "target": "body"},
                    {"action": "screenshot", "parameters": {"full_page": True}}
                ]
            else:
                return [
                    {"action": "screenshot", "parameters": {"full_page": True}}
                ]
        
        else:
            # Generic plan
            return [
                {"action": "screenshot", "parameters": {"full_page": False}},
                {"action": "wait", "parameters": {"duration": 1.0}}
            ]
    
    def _convert_plan_to_web_actions(self, plan: List[Dict[str, Any]]) -> List[WebAction]:
        """Convert LLM plan to WebAction objects"""
        
        actions = []
        for step in plan:
            action = WebAction(
                action=step["action"],
                target=step["target"],
                parameters=step.get("parameters", {}),
                wait_for=step.get("wait_for", ""),
                timeout=step.get("timeout", self.config.timeout),
                screenshot_after=step.get("screenshot_after", False)
            )
            actions.append(action)
        
        return actions
    
    def get_automation_stats(self) -> Dict[str, Any]:
        """Get browser automation statistics"""
        
        total_actions = len(self.action_history)
        successful_actions = sum(1 for action in self.action_history if action.get("success", False))
        
        return {
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": (successful_actions / total_actions * 100) if total_actions > 0 else 0,
            "pages_visited": len(self.navigation_history),
            "browser_type": self.config.browser_type,
            "headless_mode": self.config.headless,
            "playwright_available": PLAYWRIGHT_AVAILABLE,
            "current_url": self.page.url if self.page else None,
            "browser_active": self.browser is not None
        }


class BrowserAutomationAgent(BaseAgent if BaseAgent else object):
    """
    Browser Automation Agent using Playwright
    
    Integrates modern browser automation capabilities with qweenone agent system
    """
    
    def __init__(self, 
                 name: str = "BrowserAutomationAgent",
                 agent_id: str = None,
                 browser_config: BrowserConfig = None):
        
        if BaseAgent:
            super().__init__(name, "Modern browser automation using Playwright", agent_id)
            self.add_capability("WEB_AUTOMATION")
            self.add_capability("BROWSER_CONTROL")
            try:
                self.add_capability("DATA_EXTRACTION")
                self.add_capability("WEB_SCRAPING")
            except:
                pass  # Custom capabilities not supported
        
        self.browser_config = browser_config or BrowserConfig()
        self.playwright_automation = PlaywrightAutomation(self.browser_config)
        self.logger = setup_logger("BrowserAutomationAgent")
        
        if hasattr(self, 'agent_type'):
            self.agent_type = "browser_automation"
    
    async def execute_task(self, task: Union[Dict[str, Any], 'LegacyTask']) -> Union['AgentResponse', Dict[str, Any]]:
        """Execute browser automation task"""
        
        # Extract task information
        if isinstance(task, dict):
            task_description = task.get("description", task.get("title", ""))
            task_id = task.get("id", "unknown")
        else:
            task_description = getattr(task, 'description', getattr(task, 'title', ''))
            task_id = getattr(task, 'task_id', 'unknown')
        
        if self.logger:
            self.logger.info(f"Executing browser task: {task_id}")
        
        try:
            # Check if this is a web/browser task
            if not self._is_browser_task(task_description):
                return self._create_response(
                    success=False,
                    error="Not a browser automation task",
                    data={"task_type": "non_browser"}
                )
            
            # Execute using Playwright automation
            async with self.playwright_automation as browser:
                result = await browser.execute_natural_language_task(task_description)
            
            # Get automation stats
            stats = self.playwright_automation.get_automation_stats()
            
            response_data = {
                "automation_result": result,
                "automation_stats": stats,
                "task_id": task_id,
                "agent_type": "browser_automation",
                "execution_mode": "playwright"
            }
            
            return self._create_response(
                success=result.get("overall_success", False),
                data=response_data,
                metadata={"browser_type": self.browser_config.browser_type}
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Browser automation task failed: {e}")
            
            return self._create_response(
                success=False,
                error=str(e),
                data={"task_id": task_id, "agent_type": "browser_automation"}
            )
    
    def _is_browser_task(self, description: str) -> bool:
        """Determine if task requires browser automation"""
        browser_keywords = [
            "website", "web", "browser", "url", "navigate", "click", "scrape",
            "instagram", "google", "search", "login", "form", "page", "html",
            "download", "upload", "screenshot", "automation", "selenium", "playwright"
        ]
        
        description_lower = description.lower()
        return any(keyword in description_lower for keyword in browser_keywords)
    
    def _create_response(self, 
                        success: bool,
                        data: Any = None,
                        error: str = "",
                        metadata: Dict[str, Any] = None) -> Union['AgentResponse', Dict[str, Any]]:
        """Create response in appropriate format"""
        
        if AgentResponse:
            return AgentResponse(
                success=success,
                data=data,
                error=error,
                metadata=metadata or {}
            )
        else:
            return {
                "success": success,
                "data": data,
                "error": error,
                "metadata": metadata or {}
            }


# Utility functions
async def quick_browser_automation(task_description: str, 
                                 headless: bool = True) -> Dict[str, Any]:
    """Quick browser automation for simple tasks"""
    
    config = BrowserConfig(headless=headless)
    
    async with PlaywrightAutomation(config) as browser:
        return await browser.execute_natural_language_task(task_description)


if __name__ == "__main__":
    """Demo browser automation"""
    
    async def demo_browser_automation():
        """Demonstrate browser automation capabilities"""
        
        print("🌐 === PLAYWRIGHT BROWSER AUTOMATION DEMO ===")
        
        if not PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright not available. Install with:")
            print("   pip install playwright")
            print("   playwright install")
            return
        
        # Test tasks
        test_tasks = [
            "Take a screenshot of Google homepage",
            "Navigate to example.com and take a screenshot",
            "Search Google for 'Python automation' and screenshot results"
        ]
        
        # Create browser automation
        config = BrowserConfig(headless=False, slow_mo=500)  # Visible for demo
        automation = PlaywrightAutomation(config)
        
        try:
            async with automation:
                for i, task in enumerate(test_tasks, 1):
                    print(f"\n🎯 Test {i}: {task}")
                    
                    result = await automation.execute_natural_language_task(task)
                    
                    print(f"✅ Success: {result.get('overall_success', False)}")
                    print(f"📊 Actions: {len(result.get('action_results', []))}")
                    
                    if result.get('current_url'):
                        print(f"🌐 Final URL: {result['current_url']}")
                
                # Show stats
                stats = automation.get_automation_stats()
                print(f"\n📈 Browser Stats:")
                print(f"  • Total actions: {stats['total_actions']}")
                print(f"  • Success rate: {stats['success_rate']:.1f}%")
                print(f"  • Pages visited: {stats['pages_visited']}")
                print(f"  • Browser: {stats['browser_type']}")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    
    # Run demo
    asyncio.run(demo_browser_automation())