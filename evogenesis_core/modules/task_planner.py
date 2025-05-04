"""
Task Planning & Execution Module - Handles goal decomposition and task execution.

This module is responsible for translating high-level goals into actionable plans,
assigning tasks to agents, monitoring progress, and handling replanning when needed.
"""

from typing import Dict, Any, List, Optional, Callable, Union, Set
import uuid
from enum import Enum
import time
from datetime import datetime, timedelta


class TaskStatus(str, Enum):
    """Possible statuses for a task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TaskPriority(int, Enum):
    """Priority levels for tasks."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class PlanningStrategy(str, Enum):
    """Available planning strategies."""
    HIERARCHICAL = "hierarchical"  # Break down into multi-level tasks
    SEQUENTIAL = "sequential"      # Simple linear sequence of tasks
    PARALLEL = "parallel"          # Maximize parallel execution
    ADAPTIVE = "adaptive"          # Dynamically adjust based on feedback
    GOAL_ORIENTED = "goal_oriented"  # Based on GOAP algorithms
    CRITICAL_PATH = "critical_path"  # Focus on tasks in the critical path


class Task:
    """Represents a single executable task in the system."""
    
    def __init__(self, task_id: Optional[str] = None, name: str = "Generic Task",
                 description: str = "", parent_id: Optional[str] = None):
        self.task_id = task_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.parent_id = parent_id
        self.status = TaskStatus.PENDING
        self.assigned_agent_id = None
        self.assigned_team_id = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.dependencies = []  # List of task_ids that must complete before this task
        self.subtasks = []      # List of task_ids that are children of this task
        self.result = None
        self.metadata = {}
        
        # Advanced task properties
        self.priority = TaskPriority.MEDIUM
        self.estimated_duration = None  # In seconds
        self.deadline = None  # Absolute timestamp
        self.retry_count = 0
        self.max_retries = 3
        self.required_capabilities = []  # Capabilities needed to execute this task
        self.required_resources = {}  # Dict of resource_type -> amount
        self.execution_strategy = {}  # Special execution instructions
        
        # Progress tracking
        self.progress = 0.0  # 0.0 to 1.0
        self.checkpoints = []  # List of milestone/checkpoint timestamps
        self.execution_logs = []  # List of execution logs/events
        
        # Self-improvement
        self.feedback = []  # Feedback on task execution
        self.improvement_suggestions = []  # Suggestions for better execution
    
    def __str__(self):
        return f"Task({self.name}, id={self.task_id}, status={self.status})"
    
    def update_progress(self, progress: float, checkpoint_name: Optional[str] = None):
        """Update the progress of this task."""
        self.progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
        
        if checkpoint_name:
            self.checkpoints.append({
                "name": checkpoint_name,
                "timestamp": time.time(),
                "progress": self.progress
            })
    
    def add_log(self, log_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Add an execution log entry."""
        self.execution_logs.append({
            "type": log_type,
            "message": message,
            "timestamp": time.time(),
            "data": data or {}
        })
    
    def add_feedback(self, source: str, rating: float, comments: str = ""):
        """Add feedback about the task execution."""
        self.feedback.append({
            "source": source,
            "rating": rating,  # 0.0 to 5.0
            "comments": comments,
            "timestamp": time.time()
        })
    
    def add_improvement_suggestion(self, suggestion: str, source: str = "system"):
        """Add a suggestion for improving this task's execution."""
        self.improvement_suggestions.append({
            "suggestion": suggestion,
            "source": source,
            "timestamp": time.time(),
            "implemented": False
        })
    
    def is_blocked(self) -> bool:
        """Check if this task is blocked by dependencies."""
        return self.status == TaskStatus.BLOCKED
    
    def is_overdue(self) -> bool:
        """Check if this task has passed its deadline."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline
    
    def get_estimated_completion_time(self) -> Optional[float]:
        """
        Estimate when this task will be completed based on progress.
        
        Returns:
            Estimated completion timestamp or None if cannot be estimated
        """
        if self.status == TaskStatus.COMPLETED:
            return self.completed_at
        
        if self.started_at is None or self.estimated_duration is None:
            return None
        
        # Simple linear extrapolation based on progress
        if self.progress > 0:
            elapsed = time.time() - self.started_at
            total_estimated = elapsed / self.progress
            return self.started_at + total_estimated
        
        # Fall back to simple estimate
        return self.started_at + self.estimated_duration


class Goal:
    """A high-level goal that can be decomposed into tasks."""
    
    def __init__(self, goal_id: Optional[str] = None, name: str = "Generic Goal",
                 description: str = "", source: str = "user"):
        self.goal_id = goal_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.source = source  # "user" or "system"
        self.status = "active"
        self.created_at = time.time()
        self.completed_at = None
        self.root_tasks = []  # List of top-level task_ids for this goal
        
        # Advanced goal properties
        self.priority = TaskPriority.MEDIUM
        self.deadline = None  # Absolute timestamp for completion
        self.planning_strategy = PlanningStrategy.HIERARCHICAL
        self.tags = []  # Tags for categorization
        self.context = {}  # Additional contextual information
        
        # Progress tracking
        self.progress = 0.0  # 0.0 to 1.0
        self.last_updated = time.time()
        
        # Team assignment
        self.assigned_team_id = None
        
        # Goal evolution
        self.parent_goal_id = None  # If this was derived from another goal
        self.child_goals = []  # Goals derived from this one
        self.versions = [{  # Version history
            "version": 1,
            "timestamp": time.time(),
            "description": description,
            "reason": "initial"
        }]
        self.current_version = 1
    
    def __str__(self):
        return f"Goal({self.name}, id={self.goal_id}, source={self.source})"
    
    def update_description(self, new_description: str, reason: str = "update"):
        """Update the goal description and track the change."""
        self.description = new_description
        self.current_version += 1
        self.versions.append({
            "version": self.current_version,
            "timestamp": time.time(),
            "description": new_description,
            "reason": reason
        })
        self.last_updated = time.time()
    
    def calculate_progress(self, tasks: Dict[str, Task]) -> float:
        """
        Calculate goal progress based on task completion.
        
        Args:
            tasks: Dictionary of task_id -> Task
            
        Returns:
            Progress as a float from 0.0 to 1.0
        """
        if not self.root_tasks:
            return 0.0
        
        total_progress = 0.0
        task_count = 0
        
        def calculate_task_tree_progress(task_id: str) -> float:
            if task_id not in tasks:
                return 0.0
            
            task = tasks[task_id]
            
            # If task has subtasks, calculate their progress
            if task.subtasks:
                subtask_progress = sum(calculate_task_tree_progress(st_id) for st_id in task.subtasks)
                return subtask_progress / len(task.subtasks)
            
            # For leaf tasks, use their individual progress
            if task.status == TaskStatus.COMPLETED:
                return 1.0
            elif task.status in [TaskStatus.PENDING, TaskStatus.BLOCKED, TaskStatus.CANCELLED]:
                return 0.0
            else:  # IN_PROGRESS or ASSIGNED
                return task.progress
        
        # Calculate progress for each root task
        for root_task_id in self.root_tasks:
            total_progress += calculate_task_tree_progress(root_task_id)
        
        self.progress = total_progress / len(self.root_tasks)
        self.last_updated = time.time()
        return self.progress
    
    def derive_subgoal(self, name: str, description: str, reason: str = "specialization") -> 'Goal':
        """
        Derive a sub-goal from this goal.
        
        Args:
            name: Name for the new goal
            description: Description for the new goal
            reason: Reason for creating this sub-goal
            
        Returns:
            The newly created sub-goal
        """
        subgoal = Goal(
            name=name,
            description=description,
            source="system"
        )
        subgoal.parent_goal_id = self.goal_id
        subgoal.priority = self.priority
        subgoal.planning_strategy = self.planning_strategy
        subgoal.context = self.context.copy()
        subgoal.context["parent_goal"] = {
            "id": self.goal_id,
            "name": self.name
        }
        
        self.child_goals.append(subgoal.goal_id)
        return subgoal


class TaskPlanner:
    """
    Manages task planning and execution in the EvoGenesis framework.
    
    Responsible for:
    - Translating high-level goals into detailed, executable plans
    - Assigning tasks to appropriate agents
    - Monitoring task progress and dependencies
    - Replanning when tasks fail or goals change
    """
    
    def __init__(self, kernel):
        """
        Initialize the Task Planner.
        
        Args:
            kernel: The EvoGenesis kernel instance
        """
        self.kernel = kernel
        self.goals = {}  # goal_id -> Goal
        self.tasks = {}  # task_id -> Task
        self.task_callbacks = {}  # task_id -> List[Callback functions]
    
    def start(self):
        """Start the Task Planner module."""
        pass
    
    def stop(self):
        """Stop the Task Planner module."""
        pass
    
    def get_status(self):
        """Get the current status of the Task Planner."""
        return {
            "status": "active",
            "goal_count": len(self.goals),
            "task_count": len(self.tasks),
            "pending_tasks": sum(1 for t in self.tasks.values() 
                               if t.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS])
        }
    
    def create_goal(self, name: str, description: str, source: str = "user") -> Goal:
        """
        Create a new goal in the system.
        
        Args:
            name: The name of the goal
            description: Detailed description of the goal
            source: Source of the goal ("user" or "system")
            
        Returns:
            The newly created Goal instance
        """
        goal = Goal(name=name, description=description, source=source)
        self.goals[goal.goal_id] = goal
        
        # Store in memory for learning and context
        self.kernel.memory_manager.store_goal(goal)
        
        return goal
    
    def decompose_goal(self, goal_id: str) -> List[str]:
        """
        Decompose a goal into executable tasks.
        
        Args:
            goal_id: The ID of the goal to decompose
            
        Returns:
            List of task IDs created from this goal
        """
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
        
        goal = self.goals[goal_id]
        
        # Use LLM to decompose the goal into tasks
        decomposition = self.kernel.llm_orchestrator.execute_prompt(
            task_type="goal_decomposition",
            prompt_template="decompose_goal.jinja2",
            params={
                "goal_name": goal.name,
                "goal_description": goal.description,
                "available_agent_types": self.kernel.agent_manager.get_agent_types(), # Reverted to agent_manager
                "available_tools": self.kernel.tooling_system.get_available_tools()
            }
        )
        
        # Create tasks from the decomposition
        for task_def in decomposition.get("tasks", []):
            task = self.create_task(
                name=task_def["name"],
                description=task_def["description"],
                parent_id=None  # This is a root task
            )
            goal.root_tasks.append(task.task_id)
            
            # Create subtasks if defined
            if "subtasks" in task_def:
                self._create_subtasks(task.task_id, task_def["subtasks"])
        
        return goal.root_tasks
    
    def _create_subtasks(self, parent_id: str, subtask_defs: List[Dict[str, Any]]):
        """
        Recursively create subtasks for a parent task.
        
        Args:
            parent_id: The ID of the parent task
            subtask_defs: List of subtask definitions
        """
        for subtask_def in subtask_defs:
            subtask = self.create_task(
                name=subtask_def["name"],
                description=subtask_def["description"],
                parent_id=parent_id
            )
            self.tasks[parent_id].subtasks.append(subtask.task_id)
            
            # Add dependencies if specified
            if "dependencies" in subtask_def:
                for dep_name in subtask_def["dependencies"]:
                    for task_id in self.tasks:
                        if self.tasks[task_id].name == dep_name and self.tasks[task_id].parent_id == parent_id:
                            subtask.dependencies.append(task_id)
            
            # Recurse if this subtask has its own subtasks
            if "subtasks" in subtask_def:
                self._create_subtasks(subtask.task_id, subtask_def["subtasks"])
    
    def create_task(self, name: str, description: str, parent_id: Optional[str] = None) -> Task:
        """
        Create a new task in the system.
        
        Args:
            name: The name of the task
            description: Detailed description of the task
            parent_id: Optional ID of the parent task
            
        Returns:
            The newly created Task instance
        """
        task = Task(name=name, description=description, parent_id=parent_id)
        self.tasks[task.task_id] = task
        
        # Log activity
        self.kernel.log_activity(
            activity_type="task_created",
            title="Task Created",
            message=f"Task '{name}' created",
            data={"task_id": task.task_id, "task_name": name, "parent_id": parent_id}
        )
        
        return task
    
    def assign_task(self, task_id: str, agent_id: Optional[str] = None) -> bool:
        """
        Assign a task to an agent for execution.
        
        Args:
            task_id: The ID of the task to assign
            agent_id: Optional specific agent ID to assign to, or None to auto-select
            
        Returns:
            True if assignment was successful, False otherwise
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Check if all dependencies are completed
        for dep_id in task.dependencies:
            if dep_id in self.tasks and self.tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        
        # Auto-select agent if none provided
        if not agent_id:
            # Find the best agent for this task
            agent_matches = self.kernel.agent_manager.find_agents_for_task(
                task_description=task.description,
                task_metadata=task.metadata
            )
            
            if not agent_matches:
                return False
            
            agent_id = agent_matches[0]["agent_id"]
        
        # Check if agent exists
        if agent_id not in self.kernel.agent_manager.agents:
            return False
        
        # Assign the task
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent_id = agent_id
        
        # Add to agent's task list
        self.kernel.agent_manager.agents[agent_id].assigned_tasks.append(task_id)
        
        return True
    
    def start_task(self, task_id: str) -> bool:
        """
        Mark a task as started.
        
        Args:
            task_id: The ID of the task to start
            
        Returns:
            True if successful, False otherwise
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status != TaskStatus.ASSIGNED:
            return False
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()
        
        # Log activity to kernel for tracking
        self.kernel.log_activity(
            activity_type="task_started",
            title="Task Started",
            message=f"Task '{task.name}' ({task_id}) started",
            data={"task_id": task_id, "task_name": task.name}
        )
        
        return True
        
    def complete_task(self, task_id: str, result: Any = None) -> bool:
        """
        Mark a task as completed with an optional result.
        
        Args:
            task_id: The ID of the task to complete
            result: The result data from the task execution
            
        Returns:
            True if successful, False otherwise
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status != TaskStatus.IN_PROGRESS:
            return False
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.result = result
        
        # Log activity to kernel
        self.kernel.log_activity(
            activity_type="task_completed",
            title="Task Completed",
            message=f"Task '{task.name}' ({task_id}) completed",
            data={"task_id": task_id, "task_name": task.name, "result": str(result) if result else None}
        )
        
        # Execute callbacks for this task
        for callback in self.task_callbacks.get(task_id, []):
            callback(task)
        
        # Update mission if this task is part of a scheduled mission
        if hasattr(self.kernel, 'mission_scheduler') and task.metadata and task.metadata.get('is_scheduled'):
            try:
                mission_id = task.metadata.get('mission_id')
                if mission_id:
                    self.kernel.mission_scheduler.update_mission_result(
                        task_id=task_id,
                        success=True,
                        result=result if isinstance(result, dict) else {'data': result}
                    )
            except Exception as e:
                self.logger.error(f"Error updating mission for completed task {task_id}: {str(e)}")
        
        # Check if parent task or goal is now complete
        self._check_parent_completion(task)
        
        # Start dependent tasks that are now ready
        self._start_dependent_tasks(task_id)
        
        return True
    
    def _check_parent_completion(self, task: Task):
        """
        Check if a parent task or goal is now complete due to this task's completion.
        
        Args:
            task: The completed task
        """
        # Check if parent task is complete
        if task.parent_id and task.parent_id in self.tasks:
            parent = self.tasks[task.parent_id]
            all_subtasks_complete = all(
                self.tasks[subtask_id].status == TaskStatus.COMPLETED
                for subtask_id in parent.subtasks
            )
            
            if all_subtasks_complete and parent.status != TaskStatus.COMPLETED:
                self.complete_task(parent.task_id)
        
        # Check if goal is complete
        for goal_id, goal in self.goals.items():
            if goal.status == "active" and task.task_id in goal.root_tasks:
                all_root_tasks_complete = all(
                    self.tasks[root_task_id].status == TaskStatus.COMPLETED
                    for root_task_id in goal.root_tasks
                )
                
                if all_root_tasks_complete:
                    goal.status = "completed"
                    goal.completed_at = time.time()
                    
                    # Notify HITL interface
                    self.kernel.hitl_interface.notify_goal_completed(goal_id)
    
    def _start_dependent_tasks(self, completed_task_id: str):
        """
        Start tasks that were waiting on this task to complete.
        
        Args:
            completed_task_id: The ID of the task that just completed
        """
        for task_id, task in self.tasks.items():
            if (completed_task_id in task.dependencies and 
                task.status == TaskStatus.PENDING):
                
                # Check if all dependencies are now satisfied
                all_deps_complete = all(
                    self.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                
                if all_deps_complete:
                    self.assign_task(task_id)
                    
    def fail_task(self, task_id: str, reason: str) -> bool:
        """
        Mark a task as failed with a reason.
        
        Args:
            task_id: The ID of the task to mark as failed
            reason: The reason for the failure
            
        Returns:
            True if successful, False otherwise
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        task.status = TaskStatus.FAILED
        task.add_log("failure", reason)
        
        # Log to kernel
        self.kernel.log_activity(
            activity_type="task_failed",
            title="Task Failed",
            message=f"Task '{task.name}' ({task_id}) failed: {reason}",
            data={"task_id": task_id, "task_name": task.name, "reason": reason}
        )
        
        return True
    
    def list_tasks(self) -> Dict[str, Task]:
        """
        Get a list of all tasks in the system.
        
        Returns:
            Dictionary of task_id -> Task
        """
        return self.tasks.copy()
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a specific task by ID.
        
        Args:
            task_id: The ID of the task to get
            
        Returns:
            The Task object if found, None otherwise
        """
        return self.tasks.get(task_id)
