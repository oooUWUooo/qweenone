#!/usr/bin/env python3

from typing import Dict, Any, List
from src.agents.base_agent import Task
import json

class TaskDecomposer:
    
    def __init__(self):
        self.iteration_templates = {
            "server_development": [
                {"iteration": 1, "description": "Write basic server", "focus": "core functionality"},
                {"iteration": 2, "description": "Dockerize the server", "focus": "containerization"},
                {"iteration": 3, "description": "Create multi-component architecture", "focus": "scalability and communication"}
            ],
            "parser_development": [
                {"iteration": 1, "description": "Create basic parsing module", "focus": "core parsing logic"},
                {"iteration": 2, "description": "Add authentication module", "focus": "security and access"},
                {"iteration": 3, "description": "Integrate with main system", "focus": "integration and testing"}
            ]
        }
    
    def decompose(self, task_description: str, iterations: int = 3) -> Dict[str, Any]:
        # Identify the type of task to determine appropriate decomposition pattern
        task_type = self._identify_task_type(task_description)
        
        # Create the main decomposition plan
        plan = {
            "original_task": task_description,
            "task_type": task_type,
            "iterations": [],
            "total_subtasks": 0
        }
        
        # Generate iterations based on the task type
        for i in range(1, iterations + 1):
            iteration = self._create_iteration(task_description, task_type, i, iterations)
            plan["iterations"].append(iteration)
            plan["total_subtasks"] += len(iteration["subtasks"])
        
        return plan
    
    def _identify_task_type(self, task_description: str) -> str:
        task_lower = task_description.lower()
        
        if any(keyword in task_lower for keyword in ["server", "api", "backend", "web server", "http"]):
            return "server_development"
        elif any(keyword in task_lower for keyword in ["parser", "scraping", "instagram", "parsing", "extract"]):
            return "parser_development"
        elif any(keyword in task_lower for keyword in ["docker", "container", "deployment", "infrastructure"]):
            return "infrastructure_development"
        else:
            return "general_development"
    
    def _create_iteration(self, task_description: str, task_type: str, iteration_num: int, total_iterations: int) -> Dict[str, Any]:
        # Get template for this iteration if available
        template = None
        if task_type in self.iteration_templates:
            templates = self.iteration_templates[task_type]
            if iteration_num <= len(templates):
                template = templates[iteration_num - 1]
        
        # Create iteration-specific description
        if template:
            iteration_description = template["description"]
        else:
            iteration_description = f"Iteration {iteration_num}: Develop initial version of {task_description}"
        
        # Decompose the iteration into subtasks
        subtasks = self._decompose_into_subtasks(task_description, iteration_num, total_iterations)
        
        return {
            "iteration_number": iteration_num,
            "description": iteration_description,
            "focus": template["focus"] if template else "development",
            "subtasks": subtasks,
            "estimated_effort": len(subtasks)  # Simplified effort estimation
        }
    
    def _decompose_into_subtasks(self, task_description: str, iteration_num: int, total_iterations: int) -> List[Dict[str, Any]]:
        subtasks = []
        
        # Apply different decomposition strategies based on task type
        if "server" in task_description.lower():
            subtasks = self._decompose_server_task(task_description, iteration_num)
        elif "parser" in task_description.lower() or "instagram" in task_description.lower():
            subtasks = self._decompose_parser_task(task_description, iteration_num)
        else:
            # General decomposition for other tasks
            subtasks = self._decompose_general_task(task_description, iteration_num)
        
        # Further decompose each subtask if needed
        fully_decomposed = []
        for subtask in subtasks:
            fully_decomposed.extend(self._further_decompose_subtask(subtask, iteration_num))
        
        return fully_decomposed
    
    def _decompose_server_task(self, task_description: str, iteration_num: int) -> List[Dict[str, Any]]:
        base_subtasks = []
        
        if iteration_num == 1:  # Basic server
            base_subtasks = [
                {"title": "Set up project structure", "description": "Create directory structure and initial files"},
                {"title": "Implement basic server logic", "description": "Write the core server functionality"},
                {"title": "Add basic routing", "description": "Implement endpoint routing"},
                {"title": "Test basic functionality", "description": "Verify server runs and responds to requests"}
            ]
        elif iteration_num == 2:  # Dockerization
            base_subtasks = [
                {"title": "Create Dockerfile", "description": "Write Dockerfile for containerization"},
                {"title": "Configure build process", "description": "Set up build context and dependencies"},
                {"title": "Test container build", "description": "Verify Docker image builds correctly"},
                {"title": "Test container execution", "description": "Verify container runs server correctly"}
            ]
        elif iteration_num == 3:  # Multi-component
            base_subtasks = [
                {"title": "Design component architecture", "description": "Plan multi-component system structure"},
                {"title": "Implement inter-component communication", "description": "Add protocols for server-to-server communication"},
                {"title": "Create deployment manager", "description": "Build system to manage multiple server instances"},
                {"title": "Test multi-component functionality", "description": "Verify all components work together"}
            ]
        
        return base_subtasks
    
    def _decompose_parser_task(self, task_description: str, iteration_num: int) -> List[Dict[str, Any]]:
        base_subtasks = []
        
        if iteration_num == 1:  # Basic parser
            base_subtasks = [
                {"title": "Research API endpoints", "description": "Find available endpoints and data structures"},
                {"title": "Design parser architecture", "description": "Plan the parsing system structure"},
                {"title": "Implement basic parsing logic", "description": "Write core parsing functions"},
                {"title": "Test basic parsing", "description": "Verify parser can extract basic data"}
            ]
        elif iteration_num == 2:  # Add authentication
            base_subtasks = [
                {"title": "Research authentication methods", "description": "Find how to authenticate with the service"},
                {"title": "Implement auth module", "description": "Create authentication system"},
                {"title": "Integrate auth with parser", "description": "Connect authentication to parsing functions"},
                {"title": "Test authenticated parsing", "description": "Verify parser works with authentication"}
            ]
        elif iteration_num == 3:  # Integration
            base_subtasks = [
                {"title": "Design integration points", "description": "Plan how parser integrates with main system"},
                {"title": "Implement data storage", "description": "Add database or storage for parsed data"},
                {"title": "Create monitoring system", "description": "Add error handling and monitoring"},
                {"title": "Test full integration", "description": "Verify end-to-end functionality"}
            ]
        
        return base_subtasks
    
    def _decompose_general_task(self, task_description: str, iteration_num: int) -> List[Dict[str, Any]]:
        # For general tasks, we'll apply a generic decomposition approach
        # This could be expanded based on more sophisticated NLP analysis
        
        # Example: break down into research, design, implementation, testing
        phases = ["research", "design", "implementation", "testing"]
        phase_descriptions = {
            "research": "Research and analysis phase",
            "design": "System design phase", 
            "implementation": "Implementation phase",
            "testing": "Testing and validation phase"
        }
        
        subtasks = []
        for i, phase in enumerate(phases):
            subtasks.append({
                "title": f"{phase.capitalize()} phase",
                "description": phase_descriptions[phase],
                "phase": phase,
                "subtasks": []  # This could be further decomposed based on specific requirements
            })
        
        return subtasks
    
    def _further_decompose_subtask(self, subtask: Dict[str, Any], iteration_num: int) -> List[Dict[str, Any]]:
        # For now, return the subtask as-is, but in a real system,
        # this would recursively decompose until reaching implementation-level tasks
        return [subtask]
    
    def create_task_board(self, decomposition_plan: Dict[str, Any]) -> List[Task]:
        tasks = []
        
        for iteration in decomposition_plan["iterations"]:
            for subtask in iteration["subtasks"]:
                task = Task(
                    title=f"Iteration {iteration['iteration_number']}: {subtask['title']}",
                    description=subtask["description"],
                    priority=3,  # Default priority
                    params={
                        "iteration": iteration["iteration_number"],
                        "parent_task": decomposition_plan["original_task"]
                    }
                )
                tasks.append(task)
        
        return tasks