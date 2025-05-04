"""
Mission Scheduler Module - Manages long-running and recurring tasks in EvoGenesis.

This module enables the creation and management of persistent "watchdog" tasks
that can run for extended periods (months or years) with a stable agent identity.
The scheduler uses cron-style patterns (RRULE) to define recurring tasks and
maintains persistent state across system restarts.
"""

from typing import Dict, Any, List, Optional, Union
import uuid
import time
from datetime import datetime, timedelta
import json
import os
import re
from dateutil import rrule

class MissionSchedule:
    """Represents a schedule for a recurring mission."""
    
    def __init__(self, 
                 schedule_id: Optional[str] = None,
                 name: str = "Generic Schedule",
                 rrule_pattern: Optional[str] = None,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 timezone: str = "UTC",
                 max_missed_executions: int = 3):
        """
        Initialize a mission schedule.
        
        Args:
            schedule_id: Unique identifier for the schedule
            name: Human-readable name for the schedule
            rrule_pattern: RFC 5545 RRULE pattern (e.g. 'FREQ=MINUTELY;INTERVAL=5')
            start_date: When the schedule starts
            end_date: When the schedule ends (None for indefinite)
            timezone: Timezone for the schedule
            max_missed_executions: Maximum number of missed executions before alerting
        """
        self.schedule_id = schedule_id or str(uuid.uuid4())
        self.name = name
        self.rrule_pattern = rrule_pattern or "FREQ=DAILY;INTERVAL=1"
        self.start_date = start_date or datetime.now()
        self.end_date = end_date
        self.timezone = timezone
        self.max_missed_executions = max_missed_executions
        
        # Execution history
        self.last_execution = None
        self.next_execution = self._calculate_next_execution()
        self.missed_executions = 0
        self.total_executions = 0
        self.execution_stats = {
            "success_count": 0,
            "failure_count": 0,
            "average_duration": 0
        }
    
    def _calculate_next_execution(self) -> datetime:
        """Calculate the next execution time based on the RRULE pattern."""
        start_time = self.last_execution or self.start_date
        
        # Parse the RRULE pattern
        if not self.rrule_pattern.startswith("RRULE:"):
            rule_str = f"RRULE:{self.rrule_pattern}"
        else:
            rule_str = self.rrule_pattern
            
        # Create a rule with dateutil.rrule
        rule = rrule.rrulestr(rule_str, dtstart=start_time)
        
        # Find the next occurrence after the last execution
        next_time = rule.after(start_time)
        
        return next_time
    
    def update_execution_stats(self, start_time: datetime, end_time: datetime, success: bool):
        """Update execution statistics after a task execution."""
        duration = (end_time - start_time).total_seconds()
        
        if success:
            self.execution_stats["success_count"] += 1
        else:
            self.execution_stats["failure_count"] += 1
        
        total_executions = (self.execution_stats["success_count"] + 
                           self.execution_stats["failure_count"])
        
        # Update average duration
        if total_executions > 1:
            prev_avg = self.execution_stats["average_duration"]
            prev_count = total_executions - 1
            self.execution_stats["average_duration"] = (
                (prev_avg * prev_count + duration) / total_executions
            )
        else:
            self.execution_stats["average_duration"] = duration
        
        # Update execution tracking
        self.last_execution = end_time
        self.total_executions += 1
        self.next_execution = self._calculate_next_execution()
    
    def is_due(self, current_time: Optional[datetime] = None) -> bool:
        """Check if the schedule is due for execution."""
        current_time = current_time or datetime.now()
        return self.next_execution and current_time >= self.next_execution
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the schedule to a dictionary for serialization."""
        return {
            "schedule_id": self.schedule_id,
            "name": self.name,
            "rrule_pattern": self.rrule_pattern,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "timezone": self.timezone,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "next_execution": self.next_execution.isoformat() if self.next_execution else None,
            "missed_executions": self.missed_executions,
            "total_executions": self.total_executions,
            "execution_stats": self.execution_stats,
            "max_missed_executions": self.max_missed_executions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MissionSchedule':
        """Create a schedule from a dictionary."""
        schedule = cls(
            schedule_id=data["schedule_id"],
            name=data["name"],
            rrule_pattern=data["rrule_pattern"],
            timezone=data["timezone"],
            max_missed_executions=data.get("max_missed_executions", 3)
        )
        
        # Convert ISO format strings to datetime objects
        if data.get("start_date"):
            schedule.start_date = datetime.fromisoformat(data["start_date"])
        
        if data.get("end_date"):
            schedule.end_date = datetime.fromisoformat(data["end_date"])
        
        if data.get("last_execution"):
            schedule.last_execution = datetime.fromisoformat(data["last_execution"])
        
        if data.get("next_execution"):
            schedule.next_execution = datetime.fromisoformat(data["next_execution"])
        
        schedule.missed_executions = data.get("missed_executions", 0)
        schedule.total_executions = data.get("total_executions", 0)
        schedule.execution_stats = data.get("execution_stats", {
            "success_count": 0,
            "failure_count": 0,
            "average_duration": 0
        })
        
        return schedule


class PersistentMission:
    """Represents a long-running mission that persists across system restarts."""
    
    def __init__(self,
                 mission_id: Optional[str] = None,
                 name: str = "Persistent Mission",
                 description: str = "",
                 schedule: Optional[MissionSchedule] = None,
                 agent_id: Optional[str] = None,
                 task_template: Optional[Dict[str, Any]] = None,
                 status: str = "active"):
        """
        Initialize a persistent mission.
        
        Args:
            mission_id: Unique identifier for the mission
            name: Human-readable name for the mission
            description: Detailed description of the mission
            schedule: The execution schedule for the mission
            agent_id: ID of the agent assigned to this mission
            task_template: Template for generating tasks for this mission
            status: Current status of the mission ("active", "paused", "completed")
        """
        self.mission_id = mission_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.schedule = schedule
        self.agent_id = agent_id
        self.task_template = task_template or {}
        self.status = status
        
        # Mission state
        self.creation_time = datetime.now()
        self.last_modified = self.creation_time
        self.execution_history = []  # List of execution timestamps and results
        self.state_data = {}  # Persistent state data for the mission
    
    def add_execution_record(self, 
                            execution_time: datetime, 
                            task_id: str, 
                            success: bool, 
                            result: Dict[str, Any] = None):
        """Add a record of mission execution."""
        record = {
            "execution_time": execution_time,
            "task_id": task_id,
            "success": success,
            "result": result or {}
        }
        
        self.execution_history.append(record)
        self.last_modified = datetime.now()
        
        # Update schedule statistics if available
        if self.schedule:
            start_time = execution_time - timedelta(
                seconds=self.schedule.execution_stats.get("average_duration", 0)
            )
            self.schedule.update_execution_stats(start_time, execution_time, success)
    
    def update_state(self, state_update: Dict[str, Any]):
        """Update the persistent state data for the mission."""
        self.state_data.update(state_update)
        self.last_modified = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the mission to a dictionary for serialization."""
        return {
            "mission_id": self.mission_id,
            "name": self.name,
            "description": self.description,
            "schedule": self.schedule.to_dict() if self.schedule else None,
            "agent_id": self.agent_id,
            "task_template": self.task_template,
            "status": self.status,
            "creation_time": self.creation_time.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "execution_history": [
                {
                    "execution_time": record["execution_time"].isoformat(),
                    "task_id": record["task_id"],
                    "success": record["success"],
                    "result": record["result"]
                }
                for record in self.execution_history
            ],
            "state_data": self.state_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistentMission':
        """Create a mission from a dictionary."""
        mission = cls(
            mission_id=data["mission_id"],
            name=data["name"],
            description=data["description"],
            agent_id=data["agent_id"],
            task_template=data["task_template"],
            status=data["status"]
        )
        
        # Convert ISO format strings to datetime objects
        mission.creation_time = datetime.fromisoformat(data["creation_time"])
        mission.last_modified = datetime.fromisoformat(data["last_modified"])
        
        # Load schedule if available
        if data.get("schedule"):
            mission.schedule = MissionSchedule.from_dict(data["schedule"])
        
        # Load execution history
        mission.execution_history = [
            {
                "execution_time": datetime.fromisoformat(record["execution_time"]),
                "task_id": record["task_id"],
                "success": record["success"],
                "result": record["result"]
            }
            for record in data.get("execution_history", [])
        ]
        
        # Load state data
        mission.state_data = data.get("state_data", {})
        
        return mission


class MissionScheduler:
    """
    Manages long-running and recurring mission schedules in the EvoGenesis framework.
    
    Responsible for:
    - Creating and managing recurring task patterns
    - Maintaining persistent agent identities
    - Tracking execution of scheduled missions
    - Ensuring mission continuity across system restarts
    """
    
    def __init__(self, kernel):
        """
        Initialize the Mission Scheduler.
        
        Args:
            kernel: The EvoGenesis kernel instance
        """
        self.kernel = kernel
        self.missions = {}  # mission_id -> PersistentMission
        self.schedules = {}  # schedule_id -> MissionSchedule
        self.persistent_agents = {}  # agent_id -> metadata
        
        # Scheduler settings
        self.scheduler_settings = {
            "enabled": True,
            "check_interval": 60,  # seconds
            "max_concurrent_missions": 50,
            "storage_path": os.path.join("data", "missions"),
            "gitops_sync": False,  # Whether to sync with GitOps repo
            "gitops_repo_path": ""
        }
        
        # Common schedules
        self.common_schedule_patterns = {
            "every_5_minutes": "FREQ=MINUTELY;INTERVAL=5",
            "hourly": "FREQ=HOURLY;INTERVAL=1",
            "daily": "FREQ=DAILY;INTERVAL=1",
            "weekly": "FREQ=WEEKLY;INTERVAL=1",
            "monthly": "FREQ=MONTHLY;INTERVAL=1",
            "weekdays_9am": "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9;BYMINUTE=0",
            "business_hours": "FREQ=HOURLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9,10,11,12,13,14,15,16,17"
        }
        
        # Last check time for schedules
        self.last_check_time = datetime.now()
        
        # Ensure storage path exists
        if not os.path.exists(self.scheduler_settings["storage_path"]):
            os.makedirs(self.scheduler_settings["storage_path"], exist_ok=True)
    
    def start(self):
        """Start the Mission Scheduler."""
        self.load_missions_from_storage()
        self.kernel.logger.info("Mission Scheduler started")
    
    def stop(self):
        """Stop the Mission Scheduler."""
        self.save_missions_to_storage()
        self.kernel.logger.info("Mission Scheduler stopped")
    
    def load_missions_from_storage(self):
        """Load missions from persistent storage."""
        storage_path = self.scheduler_settings["storage_path"]
        if not os.path.exists(storage_path):
            return
        
        mission_files = [f for f in os.listdir(storage_path) if f.endswith(".json")]
        for file_name in mission_files:
            try:
                with open(os.path.join(storage_path, file_name), 'r') as f:
                    mission_data = json.load(f)
                    mission = PersistentMission.from_dict(mission_data)
                    self.missions[mission.mission_id] = mission
                    
                    # Also register the schedule
                    if mission.schedule:
                        self.schedules[mission.schedule.schedule_id] = mission.schedule
            except Exception as e:
                self.kernel.logger.error(f"Error loading mission from {file_name}: {e}")
    
    def save_missions_to_storage(self):
        """Save missions to persistent storage."""
        storage_path = self.scheduler_settings["storage_path"]
        if not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)
        
        for mission_id, mission in self.missions.items():
            try:
                file_path = os.path.join(storage_path, f"{mission_id}.json")
                with open(file_path, 'w') as f:
                    json.dump(mission.to_dict(), f, indent=2)
            except Exception as e:
                self.kernel.logger.error(f"Error saving mission {mission_id}: {e}")
    
    def create_mission_schedule(self, 
                              name: str, 
                              rrule_pattern: str,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> MissionSchedule:
        """
        Create a new mission schedule.
        
        Args:
            name: Human-readable name for the schedule
            rrule_pattern: RFC 5545 RRULE pattern (e.g. 'FREQ=MINUTELY;INTERVAL=5')
            start_date: When the schedule starts (defaults to now)
            end_date: When the schedule ends (None for indefinite)
            
        Returns:
            The newly created schedule
        """
        # If a common pattern name is provided, use its pattern
        if rrule_pattern in self.common_schedule_patterns:
            rrule_pattern = self.common_schedule_patterns[rrule_pattern]
        
        schedule = MissionSchedule(
            name=name,
            rrule_pattern=rrule_pattern,
            start_date=start_date,
            end_date=end_date
        )
        
        self.schedules[schedule.schedule_id] = schedule
        return schedule
    
    def create_persistent_mission(self,
                                name: str,
                                description: str,
                                schedule: Union[str, MissionSchedule],
                                task_template: Dict[str, Any],
                                agent_spec: Optional[Dict[str, Any]] = None) -> PersistentMission:
        """
        Create a new persistent mission.
        
        Args:
            name: Human-readable name for the mission
            description: Detailed description of the mission
            schedule: Schedule for the mission (ID, pattern name, or MissionSchedule object)
            task_template: Template for generating tasks for this mission
            agent_spec: Specification for the agent to create/use (None to use existing)
            
        Returns:
            The newly created mission
        """
        # Resolve the schedule
        mission_schedule = None
        if isinstance(schedule, str):
            # Check if it's a schedule ID
            if schedule in self.schedules:
                mission_schedule = self.schedules[schedule]
            # Check if it's a common pattern name
            elif schedule in self.common_schedule_patterns:
                mission_schedule = self.create_mission_schedule(
                    name=f"{name} Schedule",
                    rrule_pattern=self.common_schedule_patterns[schedule]
                )
            # Assume it's a raw RRULE pattern
            else:
                mission_schedule = self.create_mission_schedule(
                    name=f"{name} Schedule",
                    rrule_pattern=schedule
                )
        else:
            mission_schedule = schedule
        
        # Create or get the persistent agent
        agent_id = None
        if agent_spec:
            # Check if we should create a persistent agent
            if agent_spec.get("persistent_identity"):
                agent_id = self._get_or_create_persistent_agent(agent_spec)
            else:
                # Create a normal agent
                agent = self.kernel.agent_factory.create_agent( # Changed from agent_manager
                    agent_type=agent_spec.get("agent_type", "generic"),
                    name=agent_spec.get("name", name),
                    capabilities=agent_spec.get("capabilities", []),
                    domain=agent_spec.get("domain"),
                    persistent_mission=True
                )
                agent_id = agent.agent_id
        
        # Create the mission
        mission = PersistentMission(
            name=name,
            description=description,
            schedule=mission_schedule,
            agent_id=agent_id,
            task_template=task_template
        )
        
        self.missions[mission.mission_id] = mission
        
        # Save the mission to storage
        self._save_mission(mission)
        
        return mission
    
    def _get_or_create_persistent_agent(self, agent_spec: Dict[str, Any]) -> str:
        """
        Get an existing persistent agent or create a new one.
        
        Args:
            agent_spec: Specification for the agent
            
        Returns:
            The agent ID
        """
        # Check if we already have a persistent agent with this identity
        identity = agent_spec.get("persistent_identity")
        if not identity:
            identity = f"persistent_{str(uuid.uuid4())}"
        
        # Normalize the identity string
        identity = re.sub(r'[^a-zA-Z0-9_-]', '_', identity).lower()
        
        # Check if agent exists
        for agent_id, agent_meta in self.persistent_agents.items():
            if agent_meta.get("persistent_identity") == identity:
                return agent_id
        
        # Create a new persistent agent
        memory_namespace = f"persistent_{identity}"
        agent = self.kernel.agent_manager.create_agent(
            agent_type=agent_spec.get("agent_type", "generic"),
            name=agent_spec.get("name", f"Persistent Agent {identity}"),
            capabilities=agent_spec.get("capabilities", []),
            domain=agent_spec.get("domain"),
            persistent_mission=True,
            persistent_identity=identity,
            memory_namespace=memory_namespace
        )
        
        # Register as persistent agent
        self.persistent_agents[agent.agent_id] = {
            "persistent_identity": identity,
            "name": agent.name,
            "creation_time": datetime.now().isoformat(),
            "memory_namespace": memory_namespace
        }
        
        return agent.agent_id
    
    def check_due_schedules(self):
        """Check for and trigger scheduled missions that are due for execution."""
        current_time = datetime.now()
        
        # Don't check too frequently (lightweight approach)
        if (current_time - self.last_check_time).total_seconds() < self.scheduler_settings["check_interval"]:
            return
        
        self.last_check_time = current_time
        
        # Check each mission
        for mission_id, mission in list(self.missions.items()):
            if mission.status != "active" or not mission.schedule:
                continue
                
            if mission.schedule.is_due(current_time):
                self._trigger_mission(mission)
    
    def _trigger_mission(self, mission: PersistentMission):
        """
        Trigger the execution of a mission.
        
        Args:
            mission: The mission to trigger
        """
        # Generate a task from the template
        task_data = mission.task_template.copy()
        
        # Add context from mission state
        task_data["mission_id"] = mission.mission_id
        task_data["mission_state"] = mission.state_data
        task_data["execution_count"] = len(mission.execution_history)
        
        # Create the task using the task planner
        task = self.kernel.task_planner.create_task(
            name=f"{mission.name} - Execution {len(mission.execution_history) + 1}",
            description=mission.description,
            metadata={
                "mission_id": mission.mission_id,
                "is_scheduled": True,
                "schedule_id": mission.schedule.schedule_id if mission.schedule else None
            },
            **task_data
        )
        
        # Assign to the persistent agent if specified
        if mission.agent_id:
            self.kernel.task_planner.assign_task(task.task_id, mission.agent_id)
        
        # Update the mission schedule
        execution_time = datetime.now()
        mission.add_execution_record(
            execution_time=execution_time,
            task_id=task.task_id,
            success=True,  # Initially mark as successful (will be updated on completion)
            result={"status": "initiated"}
        )
        
        # Save the updated mission
        self._save_mission(mission)
        
        self.kernel.logger.info(f"Triggered mission {mission.name} (ID: {mission.mission_id})")
    
    def update_mission_result(self, task_id: str, success: bool, result: Dict[str, Any]):
        """
        Update a mission with the results of a task execution.
        
        Args:
            task_id: The ID of the task that was executed
            success: Whether the task was successful
            result: The result data from the task
        """
        # Find the mission that contains this task
        for mission_id, mission in self.missions.items():
            for record in mission.execution_history:
                if record.get("task_id") == task_id:
                    # Update the record
                    record["success"] = success
                    record["result"] = result
                    
                    # Extract state updates from the result if available
                    if "mission_state_updates" in result:
                        mission.update_state(result["mission_state_updates"])
                    
                    # Save the updated mission
                    self._save_mission(mission)
                    return
    
    def _save_mission(self, mission: PersistentMission):
        """Save a single mission to storage."""
        storage_path = self.scheduler_settings["storage_path"]
        if not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)
        
        try:
            file_path = os.path.join(storage_path, f"{mission.mission_id}.json")
            with open(file_path, 'w') as f:
                json.dump(mission.to_dict(), f, indent=2)
        except Exception as e:
            self.kernel.logger.error(f"Error saving mission {mission.mission_id}: {e}")
    
    def pause_mission(self, mission_id: str) -> bool:
        """
        Pause a mission.
        
        Args:
            mission_id: The ID of the mission to pause
            
        Returns:
            True if successful, False otherwise
        """
        if mission_id in self.missions:
            mission = self.missions[mission_id]
            mission.status = "paused"
            self._save_mission(mission)
            return True
        return False
    
    def resume_mission(self, mission_id: str) -> bool:
        """
        Resume a paused mission.
        
        Args:
            mission_id: The ID of the mission to resume
            
        Returns:
            True if successful, False otherwise
        """
        if mission_id in self.missions:
            mission = self.missions[mission_id]
            mission.status = "active"
            self._save_mission(mission)
            return True
        return False
    
    def delete_mission(self, mission_id: str) -> bool:
        """
        Delete a mission.
        
        Args:
            mission_id: The ID of the mission to delete
            
        Returns:
            True if successful, False otherwise
        """
        if mission_id in self.missions:
            # Remove from memory
            mission = self.missions.pop(mission_id)
            
            # Remove from storage
            storage_path = self.scheduler_settings["storage_path"]
            file_path = os.path.join(storage_path, f"{mission_id}.json")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    self.kernel.logger.error(f"Error deleting mission file {mission_id}: {e}")
                    return False
            
            # Remove associated schedule if not used by other missions
            if mission.schedule:
                schedule_id = mission.schedule.schedule_id
                used_by_others = False
                for other_mission in self.missions.values():
                    if (other_mission.schedule and 
                        other_mission.schedule.schedule_id == schedule_id):
                        used_by_others = True
                        break
                
                if not used_by_others and schedule_id in self.schedules:
                    del self.schedules[schedule_id]
            
            return True
        return False
    
    def get_all_missions(self) -> List[Dict[str, Any]]:
        """
        Get a list of all missions.
        
        Returns:
            List of mission dictionaries
        """
        return [
            {
                "mission_id": mission.mission_id,
                "name": mission.name,
                "description": mission.description,
                "status": mission.status,
                "schedule": mission.schedule.name if mission.schedule else "None",
                "agent_id": mission.agent_id,
                "executions": len(mission.execution_history),
                "last_modified": mission.last_modified.isoformat()
            }
            for mission in self.missions.values()
        ]
    
    def get_mission_details(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a mission.
        
        Args:
            mission_id: The ID of the mission
            
        Returns:
            Mission details dictionary or None if not found
        """
        if mission_id in self.missions:
            mission = self.missions[mission_id]
            
            # Get associated agent info
            agent_info = {}
            if mission.agent_id and mission.agent_id in self.kernel.agent_manager.agents:
                agent = self.kernel.agent_manager.agents[mission.agent_id]
                agent_info = {
                    "name": agent.name,
                    "status": agent.status,
                    "capabilities": agent.capabilities
                }
            
            return {
                "mission_id": mission.mission_id,
                "name": mission.name,
                "description": mission.description,
                "status": mission.status,
                "agent": agent_info,
                "schedule": mission.schedule.to_dict() if mission.schedule else None,
                "creation_time": mission.creation_time.isoformat(),
                "last_modified": mission.last_modified.isoformat(),
                "execution_history": mission.execution_history[:10],  # Limit to last 10 executions
                "state_data": mission.state_data
            }
        return None
    
    def get_mission_by_task(self, task_id: str) -> Optional[PersistentMission]:
        """
        Get the mission associated with a task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            The associated mission or None if not found
        """
        for mission in self.missions.values():
            for record in mission.execution_history:
                if record.get("task_id") == task_id:
                    return mission
        return None
