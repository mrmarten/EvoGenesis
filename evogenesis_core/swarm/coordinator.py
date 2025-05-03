"""
Coordinator Agent - High-level coordinator for distributed task execution.

This module implements a coordinator agent that manages the distribution of
large tasks across multiple EvoGenesis instances in a swarm. It's responsible for:
- Breaking down mega-goals into smaller TaskSpecs
- Publishing tasks to the message bus
- Monitoring progress and failures
- Implementing conflict resolution
- Assembling final results
"""

import logging
import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import json
import asyncio
from enum import Enum

# Import necessary EvoGenesis components
from evogenesis_core.modules.task_planner import TaskPlanner, Goal, Task
from evogenesis_core.swarm.bus import (
    TaskProducer, TaskConsumer, EventPublisher, EventSubscriber,
    MessageBus, TaskSpec, TaskStatus, create_message_bus
)


class CoordinationStrategy(str, Enum):
    """Strategies for task coordination in the swarm."""
    HIERARCHICAL = "hierarchical"  # Uses a tree structure with sub-coordinators
    CENTRALIZED = "centralized"    # Single coordinator manages all tasks
    DISTRIBUTED = "distributed"    # Multiple peer coordinators manage tasks


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts between task results."""
    VOTING = "voting"              # Use majority voting
    CONFIDENCE = "confidence"      # Select highest confidence result
    MERGE = "merge"                # Attempt to merge results
    HUMAN_REVIEW = "human_review"  # Ask for human intervention
    RERUN = "rerun"                # Re-run the task with modified parameters


class SwarmCoordinator:
    """
    High-level coordinator for managing tasks across a swarm of EvoGenesis instances.
    
    This class extends the basic EvoGenesis TaskPlanner to coordinate task execution
    across multiple instances, monitor progress, handle failures, and resolve conflicts.
    """
    
    def __init__(self, 
                 kernel,
                 message_bus: MessageBus,
                 coordinator_id: Optional[str] = None,
                 coordination_strategy: CoordinationStrategy = CoordinationStrategy.CENTRALIZED,
                 conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.CONFIDENCE,
                 monitoring_interval: int = 10,  # seconds
                 **kwargs):
        """
        Initialize the swarm coordinator.
        
        Args:
            kernel: EvoGenesis kernel instance
            message_bus: Message bus for communication
            coordinator_id: Unique identifier for this coordinator
            coordination_strategy: Strategy for coordinating tasks
            conflict_resolution: Strategy for resolving conflicts
            monitoring_interval: How often to check task status (seconds)
            **kwargs: Additional parameters
        """
        self.kernel = kernel
        self.message_bus = message_bus
        self.coordinator_id = coordinator_id or f"coordinator-{uuid.uuid4()}"
        self.coordination_strategy = coordination_strategy
        self.conflict_resolution = conflict_resolution
        self.monitoring_interval = monitoring_interval
        
        # Task tracking
        self.active_goals = {}  # goal_id -> Goal metadata
        self.active_tasks = {}  # task_id -> TaskSpec
        self.task_results = {}  # task_id -> list of results
        self.task_dependencies = {}  # task_id -> list of dependent task_ids
        
        # Create communication components
        self.task_producer = TaskProducer(message_bus)
        self.task_consumer = TaskConsumer(message_bus, self.coordinator_id)
        self.event_publisher = EventPublisher(message_bus)
        self.event_subscriber = EventSubscriber(message_bus)
        
        # Create the conflict resolver
        self.conflict_resolver = SwarmConflictResolver(
            kernel=kernel,
            strategy=conflict_resolution
        )
        
        # Set up monitoring
        self.running = False
        self.monitor_thread = None
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SwarmCoordinator initialized with ID {self.coordinator_id}")
    
    def start(self) -> None:
        """Start the coordinator."""
        if self.running:
            return
        
        self.running = True
        
        # Set up event listeners
        self._setup_event_listeners()
        
        # Start the task consumer
        self.task_consumer.start()
        
        # Start the monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_tasks,
            daemon=True,
            name=f"coordinator-monitor-{self.coordinator_id}"
        )
        self.monitor_thread.start()
        
        self.logger.info(f"SwarmCoordinator {self.coordinator_id} started")
    
    def stop(self) -> None:
        """Stop the coordinator."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop the task consumer
        self.task_consumer.stop()
        
        # Unsubscribe from all events
        self.event_subscriber.unsubscribe_all()
        
        # Wait for the monitoring thread to exit
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        self.logger.info(f"SwarmCoordinator {self.coordinator_id} stopped")
    
    def submit_goal(self, 
                   name: str, 
                   description: str, 
                   project: str = "default",
                   priority: int = 5,
                   deadline: Optional[datetime] = None,
                   submitter_id: str = "user") -> str:
        """
        Submit a high-level goal to be executed by the swarm.
        
        Args:
            name: Goal name
            description: Detailed description of what needs to be accomplished
            project: Project identifier
            priority: Priority level (1-10, 1 being highest)
            deadline: Optional deadline for completion
            submitter_id: ID of the user or agent submitting the goal
            
        Returns:
            Goal ID
        """
        # Generate a goal ID
        goal_id = f"goal-{uuid.uuid4()}"
        
        # Create a Goal object
        goal = Goal(
            goal_id=goal_id,
            name=name,
            description=description,
            source=submitter_id
        )
        
        # Add priority and deadline if provided
        if priority is not None:
            goal.priority = priority
        
        if deadline is not None:
            goal.deadline = deadline.timestamp()
        
        # Store the goal
        self.active_goals[goal_id] = {
            "goal": goal,
            "project": project,
            "status": "planning",
            "created_at": datetime.now().isoformat(),
            "created_by": submitter_id,
            "tasks": [],
            "progress": 0.0,
            "result": None
        }
        
        # Log the submission
        self.logger.info(f"Goal '{name}' submitted with ID {goal_id}")
        
        # Announce goal submission to the swarm
        self.event_publisher.publish_system_event(
            event_type="goal_submitted",
            data={
                "goal_id": goal_id,
                "name": name,
                "description": description,
                "project": project,
                "priority": priority,
                "deadline": deadline.isoformat() if deadline else None,
                "submitter_id": submitter_id
            }
        )
        
        # Start planning in a separate thread to avoid blocking
        threading.Thread(
            target=self._plan_goal,
            args=(goal_id,),
            daemon=True,
            name=f"plan-goal-{goal_id}"
        ).start()
        
        return goal_id
    
    def _plan_goal(self, goal_id: str) -> None:
        """
        Plan a goal by breaking it down into smaller tasks.
        
        Args:
            goal_id: ID of the goal to plan
        """
        if goal_id not in self.active_goals:
            self.logger.error(f"Cannot plan unknown goal: {goal_id}")
            return
        
        goal_data = self.active_goals[goal_id]
        goal = goal_data["goal"]
        project = goal_data["project"]
        
        self.logger.info(f"Planning goal {goal_id}: {goal.name}")
        
        try:
            # Use the EvoGenesis TaskPlanner to decompose the goal
            # This calls the LLM to break down the goal into a task hierarchy
            root_task_ids = self.kernel.task_planner.decompose_goal(goal_id)
            
            # Flatten the task hierarchy
            all_tasks = self._get_all_tasks(root_task_ids)
            
            # Convert to TaskSpecs and determine dependencies
            task_specs = []
            for task_id, task in all_tasks.items():
                # Create a TaskSpec
                task_spec = self._convert_to_task_spec(task, project, goal_id)
                
                # Add to tracking
                self.active_tasks[task_spec.task_id] = task_spec
                task_specs.append(task_spec)
                
                # Track dependencies (tasks that depend on this one)
                if task.parent_id:
                    if task.parent_id not in self.task_dependencies:
                        self.task_dependencies[task.parent_id] = []
                    self.task_dependencies[task.parent_id].append(task_spec.task_id)
            
            # Update the goal data
            goal_data["tasks"] = [spec.task_id for spec in task_specs]
            goal_data["status"] = "executing"
            goal_data["task_count"] = len(task_specs)
            goal_data["completed_count"] = 0
            
            # Publish the tasks to the message bus
            self._publish_tasks(task_specs, project)
            
            self.logger.info(f"Goal {goal_id} planned: {len(task_specs)} tasks created")
            
        except Exception as e:
            self.logger.error(f"Error planning goal {goal_id}: {str(e)}")
            goal_data["status"] = "failed"
            goal_data["error"] = str(e)
            
            # Publish failure event
            self.event_publisher.publish_system_event(
                event_type="goal_planning_failed",
                data={
                    "goal_id": goal_id,
                    "error": str(e)
                }
            )
    
    def _get_all_tasks(self, root_task_ids: List[str]) -> Dict[str, Task]:
        """
        Get all tasks in a hierarchy, including subtasks.
        
        Args:
            root_task_ids: IDs of the root tasks
            
        Returns:
            Dictionary mapping task IDs to Task objects
        """
        tasks = {}
        
        def add_task_and_children(task_id):
            if task_id in tasks:
                return
            
            # Get the task from the task planner
            task = self.kernel.task_planner.tasks.get(task_id)
            if not task:
                return
            
            # Add it to our collection
            tasks[task_id] = task
            
            # Recursively add subtasks
            for subtask_id in task.subtasks:
                add_task_and_children(subtask_id)
        
        # Start with the root tasks
        for task_id in root_task_ids:
            add_task_and_children(task_id)
        
        return tasks
    
    def _convert_to_task_spec(self, task: Task, project: str, goal_id: str) -> TaskSpec:
        """
        Convert an EvoGenesis Task to a distributed TaskSpec.
        
        Args:
            task: The original Task object
            project: Project identifier
            goal_id: ID of the parent goal
            
        Returns:
            TaskSpec object
        """
        # Extract required capabilities
        required_capabilities = task.required_capabilities.copy() if hasattr(task, 'required_capabilities') else []
        
        # Create the TaskSpec
        task_spec = TaskSpec(
            task_id=task.task_id,
            name=task.name,
            description=task.description,
            parent_task_id=task.parent_id,
            priority=task.priority.value if hasattr(task.priority, 'value') else 5,  # Convert enum to int
            dependencies=[task.parent_id] if task.parent_id else [],
            required_capabilities=required_capabilities,
            memory_namespace=f"goal_{goal_id}",
            execution_context={
                "goal_id": goal_id,
                "project": project
            },
            created_by=self.coordinator_id
        )
        
        return task_spec
    
    def _publish_tasks(self, task_specs: List[TaskSpec], project: str) -> None:
        """
        Publish tasks to the message bus, respecting dependencies.
        
        Args:
            task_specs: TaskSpec objects to publish
            project: Project identifier
        """
        # Group tasks by their dependency level (0 = no dependencies)
        tasks_by_level = {}
        
        # First identify tasks with no dependencies
        for spec in task_specs:
            if not spec.dependencies:
                level = 0
            else:
                # We'll determine these later
                continue
            
            if level not in tasks_by_level:
                tasks_by_level[level] = []
            
            tasks_by_level[level].append(spec)
        
        # Now identify tasks with dependencies and their levels
        remaining_tasks = [spec for spec in task_specs if spec.dependencies]
        max_iterations = len(remaining_tasks)  # Prevent infinite loops
        
        for _ in range(max_iterations):
            if not remaining_tasks:
                break
            
            still_remaining = []
            for spec in remaining_tasks:
                # Determine the maximum level of dependencies
                max_dep_level = -1
                all_deps_found = True
                
                for dep_id in spec.dependencies:
                    # Find the level of this dependency
                    found = False
                    for level, specs in tasks_by_level.items():
                        if any(s.task_id == dep_id for s in specs):
                            max_dep_level = max(max_dep_level, level)
                            found = True
                            break
                    
                    if not found:
                        all_deps_found = False
                        break
                
                if all_deps_found:
                    # This task's level is one more than its highest dependency
                    level = max_dep_level + 1
                    
                    if level not in tasks_by_level:
                        tasks_by_level[level] = []
                    
                    tasks_by_level[level].append(spec)
                else:
                    still_remaining.append(spec)
            
            remaining_tasks = still_remaining
        
        # If there are still remaining tasks, they have circular dependencies
        # Put them at a high level
        if remaining_tasks:
            level = len(tasks_by_level) + 1
            tasks_by_level[level] = remaining_tasks
            self.logger.warning(f"Detected potential circular dependencies in {len(remaining_tasks)} tasks")
        
        # Now publish the tasks level by level
        published_count = 0
        
        # Sort the levels
        sorted_levels = sorted(tasks_by_level.keys())
        
        for level in sorted_levels:
            for spec in tasks_by_level[level]:
                try:
                    self.task_producer.publish_task(spec, project)
                    published_count += 1
                except Exception as e:
                    self.logger.error(f"Error publishing task {spec.task_id}: {str(e)}")
        
        self.logger.info(f"Published {published_count} tasks for project {project}")
    def _setup_event_listeners(self) -> None:
        """Set up event listeners for task updates and system events."""
        # Listen for task updates
        def handle_task_update(update):
            try:
                task_id = update.get("task_id")
                status = update.get("status")
                result = update.get("result")
                error = update.get("error")
                
                if not task_id or task_id not in self.active_tasks:
                    self.logger.warning(f"Received update for unknown task: {task_id}")
                    return
                
                self._process_task_update(task_id, status, result, error)
            except Exception as e:
                self.logger.error(f"Error handling task update: {str(e)}")
        
        # Subscribe to task updates for all projects
        self.event_subscriber.subscribe_to_system_events(
            callback=handle_task_update,
            event_types=["task_update", "task_completed", "task_failed", "task_started"]
        )
        
        # Listen for conflict resolution requests
        def handle_conflict_resolution(event):
            try:
                task_id = event.get("data", {}).get("task_id")
                results = event.get("data", {}).get("results", [])
                
                if not task_id or task_id not in self.active_tasks:
                    self.logger.warning(f"Received conflict resolution request for unknown task: {task_id}")
                    return
                
                self._resolve_conflict(task_id, results)
            except Exception as e:
                self.logger.error(f"Error handling conflict resolution: {str(e)}")
        
        self.event_subscriber.subscribe_to_system_events(
            callback=handle_conflict_resolution,
            event_types=["conflict_resolution_request"]
        )
        
        # Listen for worker node status updates
        def handle_worker_status(event):
            try:
                worker_id = event.get("data", {}).get("worker_id")
                status = event.get("data", {}).get("status")
                capabilities = event.get("data", {}).get("capabilities", [])
                
                self.logger.info(f"Worker {worker_id} status: {status}, capabilities: {capabilities}")
                
                # Handle worker joining/leaving the swarm
                if status == "online" and worker_id not in self.known_workers:
                    self.known_workers.add(worker_id)
                elif status == "offline" and worker_id in self.known_workers:
                    self.known_workers.remove(worker_id)
                    
                    # Reassign tasks from failed worker
                    self._handle_worker_failure(worker_id)
            except Exception as e:
                self.logger.error(f"Error handling worker status update: {str(e)}")
                
        self.event_subscriber.subscribe_to_system_events(
            callback=handle_worker_status,
            event_types=["worker_status_update", "worker_joined", "worker_left"]
        )
        
        # Listen for external goal submissions
        def handle_goal_submission(event):
            try:
                goal_data = event.get("data", {})
                if not goal_data.get("name") or not goal_data.get("description"):
                    self.logger.warning("Received invalid goal submission")
                    return
                
                # Create a new goal from the event data
                self.submit_goal(
                    name=goal_data.get("name"),
                    description=goal_data.get("description"),
                    project=goal_data.get("project", "default"),
                    priority=goal_data.get("priority", 5),
                    deadline=datetime.fromisoformat(goal_data.get("deadline")) if goal_data.get("deadline") else None,
                    submitter_id=goal_data.get("submitter_id", "external")
                )
            except Exception as e:
                self.logger.error(f"Error handling goal submission: {str(e)}")
                
        self.event_subscriber.subscribe_to_system_events(
            callback=handle_goal_submission,
            event_types=["external_goal_submission"]
        )
        
        # Listen for system-wide configuration changes
        def handle_config_update(event):
            try:
                config_data = event.get("data", {})
                
                if "coordination_strategy" in config_data:
                    new_strategy = config_data["coordination_strategy"]
                    if new_strategy in [s.value for s in CoordinationStrategy]:
                        self.coordination_strategy = CoordinationStrategy(new_strategy)
                        self.logger.info(f"Updated coordination strategy to {new_strategy}")
                
                if "conflict_resolution" in config_data:
                    new_resolution = config_data["conflict_resolution"]
                    if new_resolution in [s.value for s in ConflictResolutionStrategy]:
                        self.conflict_resolution = ConflictResolutionStrategy(new_resolution)
                        self.conflict_resolver.strategy = self.conflict_resolution
                        self.logger.info(f"Updated conflict resolution strategy to {new_resolution}")
                
                if "monitoring_interval" in config_data:
                    self.monitoring_interval = int(config_data["monitoring_interval"])
                    self.logger.info(f"Updated monitoring interval to {self.monitoring_interval} seconds")
            except Exception as e:
                self.logger.error(f"Error handling configuration update: {str(e)}")
                
        self.event_subscriber.subscribe_to_system_events(
            callback=handle_config_update,
            event_types=["system_configuration_update"]
        )
        
        self.logger.info(f"Event listeners set up for coordinator {self.coordinator_id}")
    def _process_task_update(self, 
                           task_id: str, 
                           status: str,
                           result: Optional[Dict[str, Any]] = None,
                           error: Optional[str] = None) -> None:
        """
        Process a task status update.
        
        Args:
            task_id: ID of the task
            status: New status
            result: Optional task result
            error: Optional error message
        """
        if task_id not in self.active_tasks:
            self.logger.warning(f"Received update for unknown task: {task_id}")
            return
        
        task = self.active_tasks[task_id]
        
        # Update the task status
        task.status = status
        task.updated_at = datetime.now().isoformat()
        
        # Process based on status
        if status == TaskStatus.COMPLETED:
            # Store the result
            if result:
                task.result = result
                
                # Check if we already have results for this task
                if task_id in self.task_results:
                    # We have multiple results, need conflict resolution
                    self.task_results[task_id].append(result)
                    
                    # If we have enough results for conflict resolution
                    if len(self.task_results[task_id]) >= 2:
                        self._resolve_conflict(task_id, self.task_results[task_id])
                else:
                    # First result for this task
                    self.task_results[task_id] = [result]
                    
                    # Check if any dependent tasks can now start
                    self._check_dependent_tasks(task_id)
                    
                    # Update goal progress
                    self._update_goal_progress(task)
            
        elif status == TaskStatus.FAILED:
            # Store the error
            if error:
                task.error = error
            
            # Decide whether to retry or mark as permanently failed
            if task.attempt < task.retries:
                # Retry the task
                task.attempt += 1
                task.status = TaskStatus.RETRY
                task.assigned_to = None
                
                # Re-publish the task
                try:
                    self.task_producer.publish_task(
                        task, 
                        task.execution_context.get("project", "default")
                    )
                    self.logger.info(f"Retrying failed task {task_id} (attempt {task.attempt}/{task.retries})")
                except Exception as e:
                    self.logger.error(f"Error re-publishing task {task_id}: {str(e)}")
            else:
                # Task has failed permanently
                self.logger.warning(f"Task {task_id} failed permanently after {task.attempt} attempts: {error}")
                
                # Update goal status
                self._handle_task_failure(task)
        
        # Log the update
        self.logger.info(f"Task {task_id} status updated to {status}")
    
    def _check_dependent_tasks(self, completed_task_id: str) -> None:
        """
        Check if any dependent tasks can now start.
        
        Args:
            completed_task_id: ID of the task that completed
        """
        # Check if this task has any dependent tasks
        if completed_task_id not in self.task_dependencies:
            return
        
        dependent_task_ids = self.task_dependencies[completed_task_id]
        
        for dep_id in dependent_task_ids:
            if dep_id in self.active_tasks:
                dependent_task = self.active_tasks[dep_id]
                
                # Update dependencies list
                if completed_task_id in dependent_task.dependencies:
                    dependent_task.dependencies.remove(completed_task_id)
                
                # If no more dependencies, publish the task
                if not dependent_task.dependencies and dependent_task.status == TaskStatus.PENDING:
                    try:
                        self.task_producer.publish_task(
                            dependent_task,
                            dependent_task.execution_context.get("project", "default")
                        )
                        self.logger.info(f"Published dependent task {dep_id} after dependency {completed_task_id} completed")
                    except Exception as e:
                        self.logger.error(f"Error publishing dependent task {dep_id}: {str(e)}")
    
    def _update_goal_progress(self, task: TaskSpec) -> None:
        """
        Update the progress of a goal based on task completion.
        
        Args:
            task: The task that was completed
        """
        # Find the goal this task belongs to
        goal_id = task.execution_context.get("goal_id")
        if not goal_id or goal_id not in self.active_goals:
            return
        
        goal_data = self.active_goals[goal_id]
        
        # Update completed count
        goal_data["completed_count"] = goal_data.get("completed_count", 0) + 1
        
        # Calculate progress
        if "task_count" in goal_data and goal_data["task_count"] > 0:
            goal_data["progress"] = goal_data["completed_count"] / goal_data["task_count"]
        
        # Check if all tasks are completed
        if goal_data["completed_count"] == goal_data["task_count"]:
            # All tasks completed, generate the final result
            self._complete_goal(goal_id)
    
    def _handle_task_failure(self, task: TaskSpec) -> None:
        """
        Handle permanent task failure.
        
        Args:
            task: The task that failed
        """
        # Find the goal this task belongs to
        goal_id = task.execution_context.get("goal_id")
        if not goal_id or goal_id not in self.active_goals:
            return
        
        goal_data = self.active_goals[goal_id]
        
        # Decide what to do based on the importance of the task
        if task.priority <= 3:  # High priority task
            # Critical task failed, mark the goal as failed
            goal_data["status"] = "failed"
            goal_data["error"] = f"Critical task {task.task_id} ({task.name}) failed: {task.error}"
            
            self.logger.error(f"Goal {goal_id} failed due to critical task failure: {task.error}")
            
            # Publish goal failure event
            self.event_publisher.publish_system_event(
                event_type="goal_failed",
                data={
                    "goal_id": goal_id,
                    "task_id": task.task_id,
                    "error": task.error
                }
            )
        else:
            # Non-critical task, just log it and continue
            self.logger.warning(f"Non-critical task {task.task_id} failed for goal {goal_id}")
            
            # Publish warning event
            self.event_publisher.publish_system_event(
                event_type="task_failed_non_critical",
                data={
                    "goal_id": goal_id,
                    "task_id": task.task_id,
                    "error": task.error
                }
            )
    
    def _complete_goal(self, goal_id: str) -> None:
        """
        Complete a goal by assembling all task results.
        
        Args:
            goal_id: ID of the goal
        """
        if goal_id not in self.active_goals:
            return
        
        goal_data = self.active_goals[goal_id]
        
        # Collect all task results
        results = {}
        for task_id in goal_data["tasks"]:
            if task_id in self.active_tasks and self.active_tasks[task_id].result:
                results[task_id] = self.active_tasks[task_id].result
        
        # Generate the final result using the kernel's LLM
        try:
            # Create a summary of the results
            prompt = f"""
            I have completed a complex goal: "{goal_data['goal'].name}"
            
            Here are the results of the individual tasks:
            {json.dumps(results, indent=2)}
            
            Please provide:
            1. A comprehensive summary of all the findings
            2. Key insights and conclusions
            3. Any recommended next steps
            
            Format your response as a JSON object with "summary", "insights", and "next_steps" keys.
            """
              # Get the result from the LLM
            # Use the LLM orchestrator to generate an insightful summary
            try:
                from evogenesis_core.modules.llm_orchestrator import LLMOrchestrator
                
                # Get or create LLM orchestrator instance
                llm_orchestrator = self.llm_orchestrator
                if not llm_orchestrator:
                    llm_orchestrator = LLMOrchestrator()
                
                # Prepare the context from task results
                context = "\n\n".join([
                    f"Task: {task_data.get('title', 'Unknown task')}\n"
                    f"Result: {task_data.get('result', {}).get('output', 'No output')}"
                    for task_id, task_data in results.items()
                ])
                
                # Build the prompt for summarization
                prompt = f"""
                You are analyzing the results of the goal: "{goal_data.get('title', 'Unknown goal')}"
                
                Context and task results:
                {context}
                
                Generate a comprehensive summary, key insights, and recommended next steps.
                """
                
                # Get LLM response
                llm_response = llm_orchestrator.generate_content(
                    prompt=prompt,
                    model=goal_data.get("summary_model", "default"),
                    response_format={"type": "json_object"}
                )
                
                # Parse the LLM response
                if isinstance(llm_response, str):
                    import json
                    try:
                        summary = json.loads(llm_response)
                    except json.JSONDecodeError:
                        # If not valid JSON, extract what we can with basic parsing
                        summary = {
                            "summary": llm_response[:500],
                            "insights": ["Could not parse LLM response as JSON"],
                            "next_steps": ["Review the goal results manually"]
                        }
                else:
                    # Handle structured response
                    summary = {
                        "summary": llm_response.get("summary", "No summary provided"),
                        "insights": llm_response.get("insights", []),
                        "next_steps": llm_response.get("next_steps", [])
                    }
                
                self.logger.info(f"Generated goal summary using LLM orchestrator")
                
            except Exception as e:
                self.logger.error(f"Error generating summary with LLM: {str(e)}")
                # Fallback summary in case of errors
                summary = {
                    "summary": f"Goal processing completed with {len(results)} tasks",
                    "insights": ["Error generating insights with LLM"],
                    "next_steps": ["Review raw task results", "Try regenerating summary"]
                }
            
            # Update the goal data
            goal_data["status"] = "completed"
            goal_data["completed_at"] = datetime.now().isoformat()
            goal_data["result"] = {
                "summary": summary,
                "task_results": results
            }
            
            self.logger.info(f"Goal {goal_id} completed successfully")
            
            # Publish goal completion event
            self.event_publisher.publish_system_event(
                event_type="goal_completed",
                data={
                    "goal_id": goal_id,
                    "result": summary
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error completing goal {goal_id}: {str(e)}")
            goal_data["status"] = "failed"
            goal_data["error"] = f"Error generating goal summary: {str(e)}"
            
            # Publish goal failure event
            self.event_publisher.publish_system_event(
                event_type="goal_failed",
                data={
                    "goal_id": goal_id,
                    "error": str(e)
                }
            )
    
    def _resolve_conflict(self, task_id: str, results: List[Dict[str, Any]]) -> None:
        """
        Resolve conflicts between multiple task results.
        
        Args:
            task_id: ID of the task
            results: List of conflicting results
        """
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        
        self.logger.info(f"Resolving conflicts for task {task_id} with {len(results)} results")
        
        try:
            # Use the conflict resolver
            resolved_result = self.conflict_resolver.resolve(
                task=task,
                results=results
            )
            
            # Update the task with the resolved result
            task.result = resolved_result
            
            # Check if any dependent tasks can now start
            self._check_dependent_tasks(task_id)
            
            # Update goal progress
            self._update_goal_progress(task)
            
            # Log the resolution
            self.logger.info(f"Conflict resolved for task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error resolving conflict for task {task_id}: {str(e)}")
            
            # Handle the conflict resolution failure
            task.status = TaskStatus.FAILED
            task.error = f"Conflict resolution failed: {str(e)}"
            
            # Handle the task failure
            self._handle_task_failure(task)
    
    def _monitor_tasks(self) -> None:
        """Periodically monitor task status and handle stalled tasks."""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check for stalled tasks (assigned but no updates for too long)
                for task_id, task in self.active_tasks.items():
                    if task.status != TaskStatus.ASSIGNED:
                        continue
                    
                    # Check if the task has been assigned for too long
                    updated_at = datetime.fromisoformat(task.updated_at)
                    time_since_update = (current_time - updated_at).total_seconds()
                    
                    if time_since_update > task.timeout_seconds:
                        # Task has stalled, reset it to pending
                        self.logger.warning(f"Task {task_id} has stalled, reassigning")
                        
                        task.status = TaskStatus.PENDING
                        task.assigned_to = None
                        task.updated_at = current_time.isoformat()
                        
                        # Re-publish the task
                        try:
                            self.task_producer.publish_task(
                                task,
                                task.execution_context.get("project", "default")
                            )
                        except Exception as e:
                            self.logger.error(f"Error re-publishing stalled task {task_id}: {str(e)}")
                
                # Sleep until the next monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in task monitoring: {str(e)}")
                time.sleep(5)  # Shorter sleep on error to avoid prolonged issues
    
    def get_goal_status(self, goal_id: str) -> Dict[str, Any]:
        """
        Get the status of a goal.
        
        Args:
            goal_id: ID of the goal
            
        Returns:
            Status information for the goal
        """
        if goal_id not in self.active_goals:
            return {"error": "Goal not found"}
        
        goal_data = self.active_goals[goal_id]
        
        # Create a status summary
        status = {
            "goal_id": goal_id,
            "name": goal_data["goal"].name,
            "status": goal_data["status"],
            "progress": goal_data.get("progress", 0.0),
            "created_at": goal_data["created_at"],
            "tasks": {
                "total": goal_data.get("task_count", 0),
                "completed": goal_data.get("completed_count", 0)
            }
        }
        
        # Add completion time if available
        if "completed_at" in goal_data:
            status["completed_at"] = goal_data["completed_at"]
        
        # Add error if available
        if "error" in goal_data:
            status["error"] = goal_data["error"]
        
        # Add result summary if available
        if "result" in goal_data and goal_data["result"]:
            status["result_summary"] = goal_data["result"].get("summary")
        
        return status
    
    def list_goals(self, 
                 status_filter: Optional[str] = None,
                 project_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all goals with optional filtering.
        
        Args:
            status_filter: Optional status to filter by
            project_filter: Optional project to filter by
            
        Returns:
            List of goal summaries
        """
        goals = []
        
        for goal_id, goal_data in self.active_goals.items():
            # Apply filters
            if status_filter and goal_data["status"] != status_filter:
                continue
            
            if project_filter and goal_data.get("project") != project_filter:
                continue
            
            # Create a goal summary
            goal_summary = {
                "goal_id": goal_id,
                "name": goal_data["goal"].name,
                "status": goal_data["status"],
                "progress": goal_data.get("progress", 0.0),
                "project": goal_data.get("project", "default"),
                "created_at": goal_data["created_at"],
                "task_count": goal_data.get("task_count", 0)
            }
            
            goals.append(goal_summary)
        
        return goals


class SwarmConflictResolver:
    """
    Resolves conflicts between multiple results for the same task.
    
    This class handles various strategies for conflict resolution, such as
    majority voting, confidence-based selection, result merging, and human review.
    """
    
    def __init__(self, 
                 kernel,
                 strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.CONFIDENCE):
        """
        Initialize the conflict resolver.
        
        Args:
            kernel: EvoGenesis kernel instance
            strategy: Strategy for resolving conflicts
        """
        self.kernel = kernel
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
    
    def resolve(self, 
               task: TaskSpec, 
               results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve conflicts between multiple task results.
        
        Args:
            task: The task with conflicting results
            results: List of conflicting results
            
        Returns:
            Resolved result
        """
        if not results:
            raise ValueError("No results provided for conflict resolution")
        
        if len(results) == 1:
            return results[0]  # No conflict to resolve
        
        self.logger.info(f"Resolving conflict for task {task.task_id} using strategy: {self.strategy}")
        
        if self.strategy == ConflictResolutionStrategy.VOTING:
            return self._resolve_by_voting(results)
        
        elif self.strategy == ConflictResolutionStrategy.CONFIDENCE:
            return self._resolve_by_confidence(results)
        
        elif self.strategy == ConflictResolutionStrategy.MERGE:
            return self._resolve_by_merging(task, results)
        
        elif self.strategy == ConflictResolutionStrategy.HUMAN_REVIEW:
            return self._resolve_by_human_review(task, results)
        
        elif self.strategy == ConflictResolutionStrategy.RERUN:
            raise NotImplementedError("Re-running the task must be handled by the coordinator")
        
        else:
            raise ValueError(f"Unknown conflict resolution strategy: {self.strategy}")
    def _resolve_by_voting(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve conflicts by majority voting.
        
        Args:
            results: List of conflicting results
            
        Returns:
            The result with the most votes
        """
        self.logger.info(f"Resolving conflict using voting strategy with {len(results)} results")
        
        if not results:
            raise ValueError("No results provided for voting-based conflict resolution")
        
        if len(results) == 1:
            return results[0]
        
        # Extract the core content for comparison
        # This handles different result structures more robustly
        def extract_content(result):
            # Try different common fields where the main content might be
            for field in ["content", "output", "result", "data", "value"]:
                if field in result and result[field]:
                    return json.dumps(result[field], sort_keys=True)
            # Fallback to the whole result minus metadata fields
            filtered = {k: v for k, v in result.items() 
                      if not k.startswith("_") and k not in ["metadata", "timestamp", "worker_id"]}
            return json.dumps(filtered, sort_keys=True)
        
        # Map each result to its content for voting
        content_map = {}
        for i, result in enumerate(results):
            content = extract_content(result)
            content_map[i] = content
        
        # Count votes
        from collections import Counter
        vote_counts = Counter(content_map.values())
        most_common = vote_counts.most_common()
        
        # Handle ties by using confidence scores if available
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            self.logger.info("Detected tie in voting, using confidence as tiebreaker")
            
            # Find indices of results that tied for first place
            tied_content = most_common[0][0]
            tied_indices = [i for i, content in content_map.items() if content == tied_content]
            
            # Try to break the tie using confidence scores
            max_confidence = -1
            selected_index = tied_indices[0]  # Default to first tied result
            
            for idx in tied_indices:
                result = results[idx]
                confidence = None
                
                # Look for confidence in various locations
                if "confidence" in result:
                    confidence = result["confidence"]
                elif "metadata" in result and isinstance(result["metadata"], dict):
                    confidence = result["metadata"].get("confidence")
                elif "result" in result and isinstance(result["result"], dict):
                    confidence = result["result"].get("confidence")
                    
                if confidence is not None and isinstance(confidence, (int, float)) and confidence > max_confidence:
                    max_confidence = confidence
                    selected_index = idx
            
            selected_result = results[selected_index]
        else:
            # Get the winning content and find its original result
            winning_content = most_common[0][0]
            winning_indices = [i for i, content in content_map.items() if content == winning_content]
            selected_result = results[winning_indices[0]]
        
        # Add metadata about the voting process
        vote_summary = {str(i): count for i, (_, count) in enumerate(most_common)}
        selected_result["_conflict_resolution"] = {
            "strategy": "voting",
            "vote_counts": vote_summary,
            "total_votes": len(results),
            "winning_vote_count": most_common[0][1]
        }
        
        self.logger.info(f"Voting complete. Selected result with {most_common[0][1]} out of {len(results)} votes")
        return selected_result
    def _resolve_by_confidence(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve conflicts by selecting the result with highest confidence.
        
        Args:
            results: List of conflicting results
            
        Returns:
            The result with the highest confidence score
        """
        # Look for confidence scores in the results
        confidence_scores = []
        
        for i, result in enumerate(results):
            # Check for confidence in different possible locations
            confidence = None
            
            # Option 1: Direct confidence field
            if "confidence" in result:
                confidence = result["confidence"]
            
            # Option 2: Nested in metadata
            elif "metadata" in result and isinstance(result["metadata"], dict):
                if "confidence" in result["metadata"]:
                    confidence = result["metadata"]["confidence"]
            
            # Option 3: Nested in result
            elif "result" in result and isinstance(result["result"], dict):
                if "confidence" in result["result"]:
                    confidence = result["result"]["confidence"]
            
            # If we found a confidence value, add it to our list
            if confidence is not None and isinstance(confidence, (int, float)):
                confidence_scores.append((i, confidence))
        
        # If we have confidence scores, select the highest
        if confidence_scores:
            confidence_scores.sort(key=lambda x: x[1], reverse=True)
            selected_index = confidence_scores[0][0]
            selected_result = results[selected_index]
            
            # Add metadata about the selection
            selected_result["_conflict_resolution"] = {
                "strategy": "confidence",
                "confidence": confidence_scores[0][1],
                "all_confidence_scores": [score for _, score in confidence_scores]
            }
            
            return selected_result
        
        # If no confidence scores, fall back to the first result
        results[0]["_conflict_resolution"] = {
            "strategy": "confidence",
            "note": "No confidence scores found, selected first result"
        }
        
        return results[0]
    
    def _resolve_by_merging(self, task: TaskSpec, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve conflicts by merging results using an LLM.
        
        Args:
            task: The task with conflicting results
            results: List of conflicting results
            
        Returns:
            Merged result
        """        # Use the LLM orchestrator to intelligently merge conflicting results
        try:
            from evogenesis_core.modules.llm_orchestrator import LLMOrchestrator
            
            # Get or create LLM orchestrator instance
            llm_orchestrator = self.llm_orchestrator
            if not llm_orchestrator:
                llm_orchestrator = LLMOrchestrator()
            
            # Create a detailed prompt for the LLM to merge results
            prompt = """
            You need to merge multiple conflicting results into a single coherent result.
            
            Here are the results to merge:
            """
            
            # Add each result to the prompt
            for i, result in enumerate(results):
                content = result.get("content", result.get("output", "No content"))
                if isinstance(content, dict):
                    import json
                    content = json.dumps(content, indent=2)
                    
                prompt += f"\n--- Result {i+1} ---\n{content}\n"
            
            prompt += """
            Please merge these results into a single coherent result that preserves all important information.
            Structure your response as a JSON object with a "merged_content" field containing the merged result.
            If there are irreconcilable conflicts, explain them in a "_conflicts" field.
            """
            
            # Get LLM response for merging
            llm_response = llm_orchestrator.generate_content(
                prompt=prompt,
                model=task.get("merger_model", "default"),
                response_format={"type": "json_object"}
            )
            
            # Parse the LLM response
            merged_content = "Failed to merge results"
            conflicts = []
            
            if isinstance(llm_response, str):
                import json
                try:
                    response_data = json.loads(llm_response)
                    merged_content = response_data.get("merged_content", "LLM provided no merged content")
                    conflicts = response_data.get("_conflicts", [])
                except json.JSONDecodeError:
                    # If not valid JSON, use the raw response
                    merged_content = llm_response
            else:
                # Handle structured response
                merged_content = llm_response.get("merged_content", "LLM provided no merged content")
                conflicts = llm_response.get("_conflicts", [])
            
            merged_result = {
                "merged_content": merged_content,
                "_conflict_resolution": {
                    "strategy": "llm_merge",
                    "sources": len(results),
                    "conflicts": conflicts
                }
            }
            
            self.logger.info(f"Successfully merged {len(results)} conflicting results using LLM")
            
        except Exception as e:
            self.logger.error(f"Error merging results with LLM: {str(e)}")
            # Fallback merged result in case of errors
            merged_result = {
                "merged_content": f"Failed to merge {len(results)} results due to error: {str(e)}",
                "_conflict_resolution": {
                    "strategy": "error",
                    "sources": len(results)
                }
            }
            
        return merged_result
    def _resolve_by_human_review(self, task: TaskSpec, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve conflicts by requesting human review.
        
        Args:
            task: The task with conflicting results
            results: List of conflicting results
            
        Returns:
            Result selected by human review
        """
        self.logger.info(f"Requesting human review for task {task.task_id} with {len(results)} conflicting results")
        
        # Create a unique ID for this review request
        review_id = f"review-{uuid.uuid4()}"
        
        # Prepare data for human review
        review_data = {
            "review_id": review_id,
            "task_id": task.task_id,
            "task_name": task.name,
            "task_description": task.description,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create an event to publish
        human_review_event = {
            "type": "human_review_request",
            "data": review_data
        }
        
        # Create a threading Event to wait for the response
        response_event = threading.Event()
        selected_result = None
        
        # Set up a response handler
        def handle_review_response(event):
            if event.get("type") == "human_review_response" and event.get("data", {}).get("review_id") == review_id:
                nonlocal selected_result
                response_data = event.get("data", {})
                selected_index = response_data.get("selected_index")
                
                if selected_index is not None and 0 <= selected_index < len(results):
                    selected_result = results[selected_index]
                    selected_result["_conflict_resolution"] = {
                        "strategy": "human_review",
                        "reviewer_id": response_data.get("reviewer_id", "unknown"),
                        "review_notes": response_data.get("notes", "")
                    }
                    response_event.set()
        
        # Subscribe to human review responses
        self.kernel.event_subscriber.subscribe_to_system_events(
            callback=handle_review_response,
            event_types=["human_review_response"]
        )
        
        try:
            # Publish the request
            self.kernel.event_publisher.publish_system_event(
                event_type="human_review_request",
                data=review_data
            )
            
            # Wait for response with timeout (3 minutes)
            review_timeout = 180
            response_received = response_event.wait(timeout=review_timeout)
            
            if response_received and selected_result is not None:
                self.logger.info(f"Human review completed for task {task.task_id}")
                return selected_result
            else:
                # Timeout or error occurred, fall back to confidence-based resolution
                self.logger.warning(f"Human review timeout for task {task.task_id}, falling back to confidence strategy")
                fallback_result = self._resolve_by_confidence(results)
                fallback_result["_conflict_resolution"] = {
                    "strategy": "human_review_timeout",
                    "fallback_strategy": "confidence",
                    "timeout_seconds": review_timeout
                }
                return fallback_result
                
        finally:
            # Unsubscribe from the response events
            self.kernel.event_subscriber.unsubscribe_from_system_events(
                callback=handle_review_response,
                event_types=["human_review_response"]
            )
