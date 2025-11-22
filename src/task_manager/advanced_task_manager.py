#!/usr/bin/env python3
"""
Advanced Task Manager System
This module provides a comprehensive framework for managing complex tasks,
task dependencies, scheduling, and execution across multiple agents.
"""

from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
import inspect
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
import copy
import heapq
from collections import defaultdict, deque

from src.utils.logger import setup_logger


class TaskStatus(Enum):
    """Status of a task in the system"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskType(Enum):
    """Types of tasks"""
    COMPUTATION = "computation"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    MONITORING = "monitoring"
    DEPLOYMENT = "deployment"
    TESTING = "testing"


@dataclass
class TaskDependency:
    """Dependency between tasks"""
    dependent_task_id: str
    dependency_task_id: str
    dependency_type: str = "finish_to_start"  # finish_to_start, start_to_start, etc.


@dataclass
class TaskResource:
    """Resource requirements for a task"""
    cpu_cores: int = 1
    memory_mb: int = 128
    disk_space_mb: int = 50
    network_bandwidth: str = "10Mbps"
    gpu_required: bool = False
    gpu_memory_mb: int = 0
    execution_time_estimate: float = 0.0  # in seconds


@dataclass
class TaskMetrics:
    """Performance metrics for a task"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_duration: float = 0.0
    memory_usage_peak: float = 0.0
    cpu_usage_average: float = 0.0
    retry_count: int = 0
    error_count: int = 0
    progress_percentage: float = 0.0


class Task:
    """Represents a single task in the system"""
    
    def __init__(self, 
                 task_id: str,
                 title: str,
                 description: str = "",
                 task_type: TaskType = TaskType.COMPUTATION,
                 priority: TaskPriority = TaskPriority.MEDIUM,
                 dependencies: List[TaskDependency] = None,
                 resources: TaskResource = None,
                 parameters: Dict[str, Any] = None,
                 timeout: int = 3600,  # seconds
                 max_retries: int = 3,
                 agent_requirements: List[str] = None,
                 tags: List[str] = None):
        self.id = task_id
        self.title = title
        self.description = description
        self.type = task_type
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.dependencies = dependencies or []
        self.resources = resources or TaskResource()
        self.parameters = parameters or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.agent_requirements = agent_requirements or []
        self.tags = tags or []
        self.metrics = TaskMetrics()
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.result = None
        self.error_message = None
        self.progress_callback = None
        
        # For tracking execution
        self.current_retry = 0
        self.executor_agent_id = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "type": self.type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "dependencies": [
                {
                    "dependent_task_id": dep.dependent_task_id,
                    "dependency_task_id": dep.dependency_task_id,
                    "dependency_type": dep.dependency_type
                }
                for dep in self.dependencies
            ],
            "resources": {
                "cpu_cores": self.resources.cpu_cores,
                "memory_mb": self.resources.memory_mb,
                "disk_space_mb": self.resources.disk_space_mb,
                "network_bandwidth": self.resources.network_bandwidth,
                "gpu_required": self.resources.gpu_required,
                "gpu_memory_mb": self.resources.gpu_memory_mb,
                "execution_time_estimate": self.resources.execution_time_estimate
            },
            "parameters": self.parameters,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "agent_requirements": self.agent_requirements,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "result": self.result,
            "error_message": self.error_message
        }
    
    def update_status(self, status: TaskStatus):
        """Update the status of the task"""
        self.status = status
        self.updated_at = datetime.now()
    
    def update_progress(self, percentage: float, message: str = ""):
        """Update the progress of the task"""
        self.metrics.progress_percentage = percentage
        if self.progress_callback:
            self.progress_callback(self.id, percentage, message)


class TaskQueue:
    """Priority queue for managing tasks"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.queue = []  # Will be a heap queue
        self.lock = threading.Lock()
        self.task_lookup = {}  # task_id -> task
        self.logger = setup_logger(f"TaskQueue_{name}")
    
    def add_task(self, task: Task):
        """Add a task to the queue"""
        with self.lock:
            # Add task to heap with priority (negative for max-heap behavior)
            heapq.heappush(self.queue, (-task.priority.value, task.created_at.timestamp(), task))
            self.task_lookup[task.id] = task
            self.logger.info(f"Added task {task.id} to queue {self.name}")
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task from the queue based on priority"""
        with self.lock:
            if self.queue:
                priority, timestamp, task = heapq.heappop(self.queue)
                del self.task_lookup[task.id]
                self.logger.info(f"Retrieved task {task.id} from queue {self.name}")
                return task
        return None
    
    def peek_next_task(self) -> Optional[Task]:
        """Peek at the next task without removing it"""
        with self.lock:
            if self.queue:
                return self.queue[0][2]  # The task is the third element
        return None
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a specific task from the queue"""
        with self.lock:
            if task_id in self.task_lookup:
                task = self.task_lookup[task_id]
                # Remove task by marking it as None in the heap
                # In a real implementation, we'd use a more sophisticated approach
                new_queue = []
                for item in self.queue:
                    if item[2].id != task_id:
                        heapq.heappush(new_queue, item)
                self.queue = new_queue
                del self.task_lookup[task_id]
                self.logger.info(f"Removed task {task_id} from queue {self.name}")
                return True
        return False
    
    def size(self) -> int:
        """Get the size of the queue"""
        with self.lock:
            return len(self.queue)
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks in the queue"""
        with self.lock:
            return [item[2] for item in self.queue]
    
    def filter_tasks(self, status: TaskStatus = None, priority: TaskPriority = None) -> List[Task]:
        """Filter tasks by status or priority"""
        with self.lock:
            tasks = self.get_all_tasks()
            if status:
                tasks = [t for t in tasks if t.status == status]
            if priority:
                tasks = [t for t in tasks if t.priority == priority]
            return tasks


class TaskScheduler:
    """Schedules tasks based on dependencies and resource availability"""
    
    def __init__(self):
        self.logger = setup_logger("TaskScheduler")
        self.scheduled_tasks = {}  # task_id -> scheduled_time
        self.ready_tasks = deque()  # Tasks ready to execute
        self.dependency_graph = defaultdict(list)  # task_id -> list of dependent tasks
        self.dependency_count = defaultdict(int)  # task_id -> number of dependencies
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def add_task_with_dependencies(self, task: Task):
        """Add a task and its dependencies to the scheduler"""
        # Add the task
        self.scheduled_tasks[task.id] = None  # Will be scheduled later
        
        # Process dependencies
        for dep in task.dependencies:
            if dep.dependency_task_id not in self.dependency_graph:
                self.dependency_graph[dep.dependency_task_id] = []
            self.dependency_graph[dep.dependency_task_id].append(task.id)
            self.dependency_count[task.id] += 1
        
        # If no dependencies, add to ready queue
        if self.dependency_count[task.id] == 0:
            self.ready_tasks.append(task.id)
    
    def task_completed(self, task_id: str):
        """Notify the scheduler that a task has completed"""
        # Update dependent tasks
        for dependent_task_id in self.dependency_graph[task_id]:
            self.dependency_count[dependent_task_id] -= 1
            if self.dependency_count[dependent_task_id] == 0:
                self.ready_tasks.append(dependent_task_id)
        
        # Remove from scheduled tasks
        if task_id in self.scheduled_tasks:
            del self.scheduled_tasks[task_id]
    
    def get_ready_tasks(self) -> List[str]:
        """Get a list of task IDs that are ready to execute"""
        ready_list = []
        while self.ready_tasks:
            ready_list.append(self.ready_tasks.popleft())
        return ready_list
    
    def schedule_task_execution(self, task_id: str, execution_time: datetime = None):
        """Schedule a task for execution"""
        if task_id in self.scheduled_tasks:
            self.scheduled_tasks[task_id] = execution_time or datetime.now()
            self.logger.info(f"Scheduled task {task_id} for execution")
            return True
        return False


class ResourceAllocator:
    """Manages allocation of system resources to tasks"""
    
    def __init__(self, 
                 total_cpu_cores: int = 8,
                 total_memory_mb: int = 8192,
                 total_disk_space_mb: int = 102400):
        self.total_cpu_cores = total_cpu_cores
        self.total_memory_mb = total_memory_mb
        self.total_disk_space_mb = total_disk_space_mb
        
        self.allocated_cpu_cores = 0
        self.allocated_memory_mb = 0
        self.allocated_disk_space_mb = 0
        
        self.task_allocations = {}  # task_id -> resources allocated
        self.logger = setup_logger("ResourceAllocator")
    
    def allocate_resources(self, task_id: str, resources: TaskResource) -> bool:
        """Allocate resources to a task"""
        required_cpu = resources.cpu_cores
        required_memory = resources.memory_mb
        required_disk = resources.disk_space_mb
        
        # Check if resources are available
        if (self.allocated_cpu_cores + required_cpu <= self.total_cpu_cores and
            self.allocated_memory_mb + required_memory <= self.total_memory_mb and
            self.allocated_disk_space_mb + required_disk <= self.total_disk_space_mb):
            
            # Allocate resources
            self.allocated_cpu_cores += required_cpu
            self.allocated_memory_mb += required_memory
            self.allocated_disk_space_mb += required_disk
            
            self.task_allocations[task_id] = resources
            self.logger.info(f"Allocated resources to task {task_id}: "
                           f"CPU={required_cpu}, Memory={required_memory}MB, Disk={required_disk}MB")
            return True
        else:
            self.logger.warning(f"Insufficient resources for task {task_id}")
            return False
    
    def release_resources(self, task_id: str) -> bool:
        """Release resources allocated to a task"""
        if task_id in self.task_allocations:
            resources = self.task_allocations[task_id]
            
            self.allocated_cpu_cores -= resources.cpu_cores
            self.allocated_memory_mb -= resources.memory_mb
            self.allocated_disk_space_mb -= resources.disk_space_mb
            
            del self.task_allocations[task_id]
            self.logger.info(f"Released resources from task {task_id}")
            return True
        return False
    
    def get_available_resources(self) -> Dict[str, int]:
        """Get available resources"""
        return {
            "cpu_cores": self.total_cpu_cores - self.allocated_cpu_cores,
            "memory_mb": self.total_memory_mb - self.allocated_memory_mb,
            "disk_space_mb": self.total_disk_space_mb - self.allocated_disk_space_mb
        }
    
    def get_utilization_percentage(self) -> Dict[str, float]:
        """Get resource utilization percentage"""
        return {
            "cpu_utilization": (self.allocated_cpu_cores / self.total_cpu_cores) * 100 if self.total_cpu_cores > 0 else 0,
            "memory_utilization": (self.allocated_memory_mb / self.total_memory_mb) * 100 if self.total_memory_mb > 0 else 0,
            "disk_utilization": (self.allocated_disk_space_mb / self.total_disk_space_mb) * 100 if self.total_disk_space_mb > 0 else 0
        }


class TaskExecutor:
    """Executes tasks with proper resource management and error handling"""
    
    def __init__(self, resource_allocator: ResourceAllocator):
        self.resource_allocator = resource_allocator
        self.running_tasks = {}  # task_id -> task
        self.completed_tasks = {}  # task_id -> task
        self.failed_tasks = {}  # task_id -> task
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.logger = setup_logger("TaskExecutor")
        self.task_callbacks = {}  # task_id -> list of callbacks
    
    def register_callback(self, task_id: str, callback: Callable[[Task], None]):
        """Register a callback to be called when a task completes"""
        if task_id not in self.task_callbacks:
            self.task_callbacks[task_id] = []
        self.task_callbacks[task_id].append(callback)
    
    def execute_task(self, task: Task, agent_id: str = None) -> bool:
        """Execute a single task"""
        # Check if resources are available
        if not self.resource_allocator.allocate_resources(task.id, task.resources):
            self.logger.error(f"Failed to allocate resources for task {task.id}")
            return False
        
        # Update task status
        task.update_status(TaskStatus.RUNNING)
        task.executor_agent_id = agent_id
        task.metrics.start_time = datetime.now()
        
        # Add to running tasks
        self.running_tasks[task.id] = task
        
        # Execute the task in a separate thread
        future = self.executor.submit(self._run_task, task)
        
        # Add done callback to handle completion
        future.add_done_callback(lambda f: self._handle_task_completion(task.id))
        
        self.logger.info(f"Started execution of task {task.id}")
        return True
    
    def _run_task(self, task: Task):
        """Internal method to run a task"""
        try:
            # Simulate task execution
            self.logger.info(f"Executing task {task.id}")
            
            # Update progress
            task.update_progress(10.0, "Initializing task")
            time.sleep(0.1)  # Simulate initialization
            
            task.update_progress(50.0, "Processing data")
            time.sleep(0.2)  # Simulate processing
            
            task.update_progress(90.0, "Finalizing task")
            time.sleep(0.1)  # Simulate finalization
            
            # Simulate result
            task.result = f"Task {task.id} completed successfully"
            task.update_progress(100.0, "Task completed")
            task.update_status(TaskStatus.COMPLETED)
            
            # Update metrics
            task.metrics.end_time = datetime.now()
            task.metrics.execution_duration = (task.metrics.end_time - task.metrics.start_time).total_seconds()
            
            self.logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            task.error_message = str(e)
            task.update_status(TaskStatus.FAILED)
            task.metrics.error_count += 1
            self.logger.error(f"Task {task.id} failed: {e}")
    
    def _handle_task_completion(self, task_id: str):
        """Handle task completion"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            
            # Release resources
            self.resource_allocator.release_resources(task_id)
            
            # Move task to appropriate collection
            if task.status == TaskStatus.COMPLETED:
                self.completed_tasks[task_id] = task
            else:
                self.failed_tasks[task_id] = task
            
            # Remove from running tasks
            del self.running_tasks[task_id]
            
            # Call registered callbacks
            if task_id in self.task_callbacks:
                for callback in self.task_callbacks[task_id]:
                    try:
                        callback(task)
                    except Exception as e:
                        self.logger.error(f"Error in callback for task {task_id}: {e}")
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.update_status(TaskStatus.CANCELLED)
            self.resource_allocator.release_resources(task_id)
            
            # Move to completed tasks (as cancelled)
            self.completed_tasks[task_id] = self.running_tasks[task_id]
            del self.running_tasks[task_id]
            
            self.logger.info(f"Cancelled task {task_id}")
            return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task"""
        if task_id in self.running_tasks:
            return self.running_tasks[task_id].status
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id].status
        elif task_id in self.failed_tasks:
            return self.failed_tasks[task_id].status
        return None
    
    def get_running_tasks(self) -> List[Task]:
        """Get all running tasks"""
        return list(self.running_tasks.values())
    
    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks"""
        return list(self.completed_tasks.values())
    
    def get_failed_tasks(self) -> List[Task]:
        """Get all failed tasks"""
        return list(self.failed_tasks.values())


class AdvancedTaskManager:
    """Advanced task management system"""
    
    def __init__(self):
        self.logger = setup_logger("AdvancedTaskManager")
        self.task_queues = {}
        self.scheduler = TaskScheduler()
        self.resource_allocator = ResourceAllocator()
        self.executor = TaskExecutor(self.resource_allocator)
        self.tasks = {}  # All tasks in the system
        self.task_history = []  # Completed and failed tasks
        self.max_history_size = 1000
        
        # Create default queue
        self.create_queue("default")
    
    def create_queue(self, queue_name: str) -> TaskQueue:
        """Create a new task queue"""
        if queue_name not in self.task_queues:
            self.task_queues[queue_name] = TaskQueue(queue_name)
        return self.task_queues[queue_name]
    
    def create_task(self, 
                   title: str, 
                   description: str = "",
                   task_type: TaskType = TaskType.COMPUTATION,
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   queue_name: str = "default",
                   dependencies: List[TaskDependency] = None,
                   resources: TaskResource = None,
                   parameters: Dict[str, Any] = None,
                   timeout: int = 3600,
                   max_retries: int = 3,
                   agent_requirements: List[str] = None,
                   tags: List[str] = None) -> Task:
        """Create a new task"""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            dependencies=dependencies or [],
            resources=resources or TaskResource(),
            parameters=parameters or {},
            timeout=timeout,
            max_retries=max_retries,
            agent_requirements=agent_requirements or [],
            tags=tags or []
        )
        
        # Add to system
        self.tasks[task_id] = task
        
        # Add to scheduler
        self.scheduler.add_task_with_dependencies(task)
        
        # Add to queue
        if queue_name in self.task_queues:
            self.task_queues[queue_name].add_task(task)
        else:
            self.task_queues["default"].add_task(task)
        
        self.logger.info(f"Created task {task_id}: {title}")
        return task
    
    def submit_task(self, 
                   task: Task, 
                   queue_name: str = "default",
                   agent_id: str = None) -> bool:
        """Submit a task for execution"""
        # Add to scheduler
        self.scheduler.add_task_with_dependencies(task)
        
        # Add to queue
        if queue_name in self.task_queues:
            self.task_queues[queue_name].add_task(task)
        else:
            self.task_queues["default"].add_task(task)
        
        # Try to schedule for execution
        ready_tasks = self.scheduler.get_ready_tasks()
        for ready_task_id in ready_tasks:
            ready_task = self.tasks[ready_task_id]
            if ready_task.status == TaskStatus.PENDING:
                self._attempt_task_execution(ready_task, agent_id)
        
        self.logger.info(f"Submitted task {task.id} for execution")
        return True
    
    def _attempt_task_execution(self, task: Task, agent_id: str = None):
        """Attempt to execute a task if resources are available"""
        if task.status == TaskStatus.PENDING:
            # Check if task is ready to run (no pending dependencies)
            if self._are_dependencies_met(task):
                # Try to execute
                if self.executor.execute_task(task, agent_id):
                    task.update_status(TaskStatus.SCHEDULED)
                    self.scheduler.schedule_task_execution(task.id)
    
    def _are_dependencies_met(self, task: Task) -> bool:
        """Check if all dependencies for a task are met"""
        for dep in task.dependencies:
            dep_task = self.tasks.get(dep.dependency_task_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task"""
        return self.executor.get_task_status(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        return self.executor.cancel_task(task_id)
    
    def retry_task(self, task_id: str) -> bool:
        """Retry a failed task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == TaskStatus.FAILED and task.current_retry < task.max_retries:
                task.current_retry += 1
                task.update_status(TaskStatus.PENDING)
                task.error_message = None
                task.result = None
                
                # Re-add to scheduler
                self.scheduler.add_task_with_dependencies(task)
                
                # Add to queue
                self.task_queues["default"].add_task(task)
                
                self.logger.info(f"Retrying task {task_id}, attempt {task.current_retry}")
                return True
        return False
    
    def batch_create_tasks(self, task_configs: List[Dict[str, Any]]) -> List[Task]:
        """Create multiple tasks in batch"""
        tasks = []
        for config in task_configs:
            task = self.create_task(**config)
            tasks.append(task)
        return tasks
    
    def batch_submit_tasks(self, tasks: List[Task], queue_name: str = "default") -> int:
        """Submit multiple tasks in batch"""
        submitted_count = 0
        for task in tasks:
            if self.submit_task(task, queue_name):
                submitted_count += 1
        return submitted_count
    
    def schedule_recurring_task(self, 
                              task_config: Dict[str, Any], 
                              interval_seconds: int,
                              start_time: datetime = None) -> str:
        """Schedule a recurring task"""
        # This would typically use a scheduling library in a real implementation
        # For now, we'll just create and return a task ID
        task = self.create_task(**task_config)
        
        # In a real system, this would schedule the task to repeat
        self.logger.info(f"Scheduled recurring task {task.id} with interval {interval_seconds}s")
        return task.id
    
    def get_queue_status(self, queue_name: str = "default") -> Dict[str, Any]:
        """Get status of a task queue"""
        if queue_name in self.task_queues:
            queue = self.task_queues[queue_name]
            return {
                "name": queue_name,
                "size": queue.size(),
                "tasks": [
                    {
                        "id": task.id,
                        "title": task.title,
                        "status": task.status.value,
                        "priority": task.priority.value
                    }
                    for task in queue.get_all_tasks()
                ]
            }
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "total_tasks": len(self.tasks),
            "running_tasks": len(self.executor.get_running_tasks()),
            "completed_tasks": len(self.executor.get_completed_tasks()),
            "failed_tasks": len(self.executor.get_failed_tasks()),
            "queues": list(self.task_queues.keys()),
            "resource_utilization": self.resource_allocator.get_utilization_percentage(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_task_metrics(self, task_id: str) -> Optional[TaskMetrics]:
        """Get metrics for a specific task"""
        task = self.get_task(task_id)
        if task:
            return task.metrics
        return None
    
    def get_all_task_metrics(self) -> Dict[str, TaskMetrics]:
        """Get metrics for all tasks"""
        metrics = {}
        for task_id, task in self.tasks.items():
            metrics[task_id] = task.metrics
        return metrics
    
    def export_task_history(self, output_path: str) -> bool:
        """Export task history to file"""
        try:
            history_data = []
            for task in self.task_history:
                history_data.append(task.to_dict())
            
            with open(output_path, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported task history to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export task history: {e}")
            return False
    
    def import_task_history(self, input_path: str) -> bool:
        """Import task history from file"""
        try:
            with open(input_path, 'r') as f:
                history_data = json.load(f)
            
            for task_data in history_data:
                # Convert back to Task object
                task = Task(
                    task_id=task_data["id"],
                    title=task_data["title"],
                    description=task_data["description"],
                    task_type=TaskType(task_data["type"]),
                    priority=TaskPriority(task_data["priority"])
                )
                # Set other attributes
                task.status = TaskStatus(task_data["status"])
                task.result = task_data["result"]
                task.error_message = task_data["error_message"]
                task.created_at = datetime.fromisoformat(task_data["created_at"])
                task.updated_at = datetime.fromisoformat(task_data["updated_at"])
                
                self.task_history.append(task)
            
            self.logger.info(f"Imported task history from {input_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to import task history: {e}")
            return False
    
    def cleanup_completed_tasks(self, keep_count: int = 100) -> int:
        """Clean up old completed tasks to free memory"""
        completed_tasks = self.executor.get_completed_tasks()
        
        # Sort by completion time (newest first) and keep only the most recent
        completed_tasks.sort(key=lambda t: t.metrics.end_time or t.updated_at, reverse=True)
        
        tasks_to_remove = completed_tasks[keep_count:]
        removed_count = 0
        
        for task in tasks_to_remove:
            if task.id in self.tasks:
                del self.tasks[task.id]
                removed_count += 1
        
        self.logger.info(f"Cleaned up {removed_count} completed tasks")
        return removed_count
    
    def get_task_dependencies(self, task_id: str) -> List[str]:
        """Get dependencies for a task"""
        task = self.get_task(task_id)
        if task:
            return [dep.dependency_task_id for dep in task.dependencies]
        return []
    
    def get_dependent_tasks(self, task_id: str) -> List[str]:
        """Get tasks that depend on this task"""
        dependent_tasks = []
        for other_task_id, other_task in self.tasks.items():
            if other_task_id != task_id:
                for dep in other_task.dependencies:
                    if dep.dependency_task_id == task_id:
                        dependent_tasks.append(other_task_id)
                        break
        return dependent_tasks
    
    def pause_task_execution(self):
        """Pause all task execution"""
        # In a real implementation, this would pause the executor
        self.logger.info("Paused task execution")
    
    def resume_task_execution(self):
        """Resume task execution"""
        # In a real implementation, this would resume the executor
        self.logger.info("Resumed task execution")
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about tasks"""
        all_tasks = list(self.tasks.values())
        
        stats = {
            "total_tasks": len(all_tasks),
            "by_status": {},
            "by_priority": {},
            "by_type": {},
            "avg_execution_time": 0.0,
            "avg_memory_usage": 0.0,
            "total_execution_time": 0.0
        }
        
        # Count by status
        for task in all_tasks:
            status = task.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            
            priority = task.priority.value
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1
            
            task_type = task.type.value
            stats["by_type"][task_type] = stats["by_type"].get(task_type, 0) + 1
        
        # Calculate averages
        completed_tasks = [t for t in all_tasks if t.status == TaskStatus.COMPLETED]
        if completed_tasks:
            total_time = sum(
                (t.metrics.end_time - t.metrics.start_time).total_seconds() 
                if t.metrics.start_time and t.metrics.end_time 
                else t.metrics.execution_duration
                for t in completed_tasks
            )
            stats["total_execution_time"] = total_time
            stats["avg_execution_time"] = total_time / len(completed_tasks)
        
        return stats


class TaskMonitoringService:
    """Service for monitoring task execution and performance"""
    
    def __init__(self, task_manager: AdvancedTaskManager):
        self.task_manager = task_manager
        self.logger = setup_logger("TaskMonitoringService")
        self.alert_callbacks = []
        self.performance_thresholds = {
            "execution_time": 300,  # seconds
            "memory_usage": 1024,   # MB
            "error_rate": 0.1       # percentage
        }
    
    def register_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register a callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def check_performance_thresholds(self):
        """Check if any tasks are exceeding performance thresholds"""
        all_tasks = list(self.task_manager.tasks.values())
        
        for task in all_tasks:
            # Check execution time
            if (task.metrics.start_time and 
                task.status == TaskStatus.RUNNING and 
                (datetime.now() - task.metrics.start_time).total_seconds() > 
                self.performance_thresholds["execution_time"]):
                
                alert_data = {
                    "task_id": task.id,
                    "type": "execution_time_exceeded",
                    "current_time": (datetime.now() - task.metrics.start_time).total_seconds(),
                    "threshold": self.performance_thresholds["execution_time"]
                }
                self._trigger_alert(alert_data)
    
    def _trigger_alert(self, alert_data: Dict[str, Any]):
        """Trigger an alert"""
        self.logger.warning(f"Alert triggered: {alert_data}")
        for callback in self.alert_callbacks:
            try:
                callback("PERFORMANCE_ALERT", alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def generate_task_report(self, 
                           start_date: datetime = None, 
                           end_date: datetime = None) -> Dict[str, Any]:
        """Generate a report of task execution"""
        start_date = start_date or datetime.now() - timedelta(days=1)
        end_date = end_date or datetime.now()
        
        all_tasks = list(self.task_manager.tasks.values())
        
        # Filter tasks by date range
        filtered_tasks = [
            task for task in all_tasks
            if start_date <= (task.created_at or datetime.now()) <= end_date
        ]
        
        completed_tasks = [t for t in filtered_tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in filtered_tasks if t.status == TaskStatus.FAILED]
        
        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_tasks": len(filtered_tasks),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(completed_tasks) / len(filtered_tasks) if filtered_tasks else 0,
            "avg_execution_time": 0,
            "top_agents": {},
            "task_types_distribution": {}
        }
        
        if completed_tasks:
            total_time = sum(
                (t.metrics.end_time - t.metrics.start_time).total_seconds() 
                if t.metrics.start_time and t.metrics.end_time 
                else t.metrics.execution_duration
                for t in completed_tasks
            )
            report["avg_execution_time"] = total_time / len(completed_tasks)
        
        # Count by agent
        for task in all_tasks:
            if task.executor_agent_id:
                agent_id = task.executor_agent_id
                report["top_agents"][agent_id] = report["top_agents"].get(agent_id, 0) + 1
        
        # Count by task type
        for task in filtered_tasks:
            task_type = task.type.value
            report["task_types_distribution"][task_type] = report["task_types_distribution"].get(task_type, 0) + 1
        
        return report
    
    def export_monitoring_data(self, output_path: str) -> bool:
        """Export monitoring data to file"""
        try:
            report = self.generate_task_report()
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Exported monitoring data to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export monitoring data: {e}")
            return False


# Utility functions for advanced task operations

def create_complex_task_workflow(task_manager: AdvancedTaskManager) -> List[str]:
    """Create a complex workflow of interdependent tasks"""
    # Create tasks
    task1 = task_manager.create_task(
        title="Data Collection",
        description="Collect data from various sources",
        task_type=TaskType.DATA_PROCESSING,
        priority=TaskPriority.HIGH
    )
    
    task2 = task_manager.create_task(
        title="Data Processing",
        description="Process collected data",
        task_type=TaskType.DATA_PROCESSING,
        priority=TaskPriority.HIGH,
        dependencies=[TaskDependency(task2.id, task1.id)]
    )
    
    task3 = task_manager.create_task(
        title="Analysis",
        description="Analyze processed data",
        task_type=TaskType.ANALYSIS,
        priority=TaskPriority.MEDIUM,
        dependencies=[TaskDependency(task3.id, task2.id)]
    )
    
    task4 = task_manager.create_task(
        title="Validation",
        description="Validate analysis results",
        task_type=TaskType.VALIDATION,
        priority=TaskPriority.MEDIUM,
        dependencies=[TaskDependency(task4.id, task3.id)]
    )
    
    task5 = task_manager.create_task(
        title="Report Generation",
        description="Generate final report",
        task_type=TaskType.COMPUTATION,
        priority=TaskPriority.LOW,
        dependencies=[TaskDependency(task5.id, task4.id)]
    )
    
    return [task1.id, task2.id, task3.id, task4.id, task5.id]


def execute_workflow_with_agents(task_manager: AdvancedTaskManager, 
                                workflow_task_ids: List[str], 
                                agent_assignments: Dict[str, str] = None) -> bool:
    """Execute a workflow with specific agent assignments"""
    if agent_assignments is None:
        agent_assignments = {}
    
    success = True
    for task_id in workflow_task_ids:
        agent_id = agent_assignments.get(task_id)
        task = task_manager.get_task(task_id)
        if task:
            if not task_manager.submit_task(task, agent_id=agent_id):
                success = False
    
    return success


def validate_task_configuration(task_config: Dict[str, Any]) -> List[str]:
    """Validate a task configuration"""
    errors = []
    
    required_fields = ["title"]
    for field in required_fields:
        if field not in task_config:
            errors.append(f"Missing required field: {field}")
    
    # Validate priority if provided
    if "priority" in task_config:
        try:
            TaskPriority(task_config["priority"])
        except ValueError:
            errors.append(f"Invalid priority value: {task_config['priority']}")
    
    # Validate task type if provided
    if "task_type" in task_config:
        try:
            TaskType(task_config["task_type"])
        except ValueError:
            errors.append(f"Invalid task type value: {task_config['task_type']}")
    
    # Validate resources if provided
    if "resources" in task_config:
        resources = task_config["resources"]
        if not isinstance(resources, dict):
            errors.append("Resources must be a dictionary")
        else:
            for field in ["cpu_cores", "memory_mb", "disk_space_mb"]:
                if field in resources and not isinstance(resources[field], int):
                    errors.append(f"Resource field {field} must be an integer")
    
    return errors


def optimize_task_scheduling(task_manager: AdvancedTaskManager) -> Dict[str, Any]:
    """Optimize task scheduling based on current system state"""
    system_status = task_manager.get_system_status()
    resource_utilization = system_status["resource_utilization"]
    
    recommendations = {
        "adjust_priority": [],
        "reschedule_tasks": [],
        "add_resources": []
    }
    
    # If CPU utilization is high, recommend adjusting priorities
    if resource_utilization["cpu_utilization"] > 80:
        recommendations["adjust_priority"].append({
            "reason": "High CPU utilization",
            "suggestion": "Consider reducing priority of non-critical tasks"
        })
    
    # If memory utilization is high
    if resource_utilization["memory_utilization"] > 80:
        recommendations["add_resources"].append({
            "reason": "High memory utilization",
            "suggestion": "Consider adding more memory resources"
        })
    
    # Check for tasks that have been waiting too long
    for queue_name, queue in task_manager.task_queues.items():
        for task in queue.get_all_tasks():
            wait_time = (datetime.now() - task.created_at).total_seconds()
            if wait_time > 3600:  # More than 1 hour
                recommendations["reschedule_tasks"].append({
                    "task_id": task.id,
                    "wait_time": wait_time,
                    "suggestion": "Consider increasing priority or adding resources"
                })
    
    return recommendations


def backup_task_manager_state(task_manager: AdvancedTaskManager, backup_path: str) -> bool:
    """Backup the current state of the task manager"""
    try:
        state_data = {
            "tasks": {tid: task.to_dict() for tid, task in task_manager.tasks.items()},
            "task_queues": {name: [t.id for t in queue.get_all_tasks()] 
                           for name, queue in task_manager.task_queues.items()},
            "scheduler_state": {
                "scheduled_tasks": list(task_manager.scheduler.scheduled_tasks.keys()),
                "dependency_graph": dict(task_manager.scheduler.dependency_graph),
                "dependency_count": dict(task_manager.scheduler.dependency_count)
            },
            "backup_timestamp": datetime.now().isoformat()
        }
        
        backup_file = Path(backup_path) / f"task_manager_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        task_manager.logger.info(f"Task manager state backed up to {backup_file}")
        return True
    except Exception as e:
        task_manager.logger.error(f"Failed to backup task manager state: {e}")
        return False


def restore_task_manager_state(task_manager: AdvancedTaskManager, backup_path: str) -> bool:
    """Restore the task manager state from backup"""
    try:
        with open(backup_path, 'r') as f:
            state_data = json.load(f)
        
        # Clear current state
        task_manager.tasks = {}
        task_manager.task_queues = {}
        task_manager.scheduler = TaskScheduler()
        
        # Restore tasks
        for task_id, task_data in state_data["tasks"].items():
            # Create task from data
            task = Task(
                task_id=task_data["id"],
                title=task_data["title"],
                description=task_data["description"],
                task_type=TaskType(task_data["type"]),
                priority=TaskPriority(task_data["priority"])
            )
            # Set other attributes
            task.status = TaskStatus(task_data["status"])
            task.created_at = datetime.fromisoformat(task_data["created_at"])
            task.updated_at = datetime.fromisoformat(task_data["updated_at"])
            
            task_manager.tasks[task_id] = task
        
        # Restore queues
        for queue_name, task_ids in state_data["task_queues"].items():
            queue = task_manager.create_queue(queue_name)
            for task_id in task_ids:
                if task_id in task_manager.tasks:
                    queue.add_task(task_manager.tasks[task_id])
        
        # Restore scheduler state
        for task_id in state_data["scheduler_state"]["scheduled_tasks"]:
            if task_id in task_manager.tasks:
                task_manager.scheduler.scheduled_tasks[task_id] = None
        
        task_manager.scheduler.dependency_graph = defaultdict(list, 
            state_data["scheduler_state"]["dependency_graph"])
        task_manager.scheduler.dependency_count = defaultdict(int,
            state_data["scheduler_state"]["dependency_count"])
        
        task_manager.logger.info(f"Task manager state restored from {backup_path}")
        return True
    except Exception as e:
        task_manager.logger.error(f"Failed to restore task manager state: {e}")
        return False


# Example usage and testing functions

def example_usage():
    """Example of how to use the AdvancedTaskManager"""
    task_manager = AdvancedTaskManager()
    
    # Create a simple task
    task = task_manager.create_task(
        title="Sample Task",
        description="This is a sample task for demonstration",
        task_type=TaskType.COMPUTATION,
        priority=TaskPriority.HIGH
    )
    
    print(f"Created task: {task.id} - {task.title}")
    
    # Submit the task for execution
    task_manager.submit_task(task)
    
    # Get system status
    status = task_manager.get_system_status()
    print(f"System status: {status}")
    
    # Create a workflow
    workflow_ids = create_complex_task_workflow(task_manager)
    print(f"Created workflow with tasks: {workflow_ids}")
    
    # Get task statistics
    stats = task_manager.get_task_statistics()
    print(f"Task statistics: {stats}")


if __name__ == "__main__":
    example_usage()