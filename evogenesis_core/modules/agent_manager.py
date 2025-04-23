"""
Agent Management Module - Handles the lifecycle of agents and teams in EvoGenesis.

This module is responsible for creating, monitoring, and managing agents and teams,
handling inter-agent communication, and tracking resource usage.
"""

from typing import Dict, Any, List, Optional
import uuid
import time

class Agent:
    """Base class for an EvoGenesis agent."""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = "Generic Agent", 
                 capabilities: Optional[List[str]] = None, **kwargs):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.capabilities = capabilities or []
        self.status = "initialized"
        self.attributes = kwargs
        self.assigned_tasks = []
        
        # Performance tracking
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_completion_time": 0,
            "success_rate": 1.0,
            "last_active": time.time()
        }
        
        # Resource usage tracking
        self.resource_usage = {
            "tokens_consumed": 0,
            "api_calls": 0,
            "estimated_cost": 0.0
        }
        
        # Learning and adaptation
        self.knowledge_base = {}  # Domain-specific knowledge accumulated
        self.skill_improvements = []  # Record of skill improvements over time
    
    def __str__(self):
        return f"Agent({self.name}, id={self.agent_id}, status={self.status})"
    
    def update_performance(self, task_completed: bool, completion_time: float = 0):
        """Update performance metrics based on task execution."""
        if task_completed:
            self.performance_metrics["tasks_completed"] += 1
            # Update average completion time
            prev_avg = self.performance_metrics["average_completion_time"]
            prev_count = self.performance_metrics["tasks_completed"] - 1
            
            if prev_count > 0:
                self.performance_metrics["average_completion_time"] = (
                    (prev_avg * prev_count + completion_time) / 
                    self.performance_metrics["tasks_completed"]
                )
            else:
                self.performance_metrics["average_completion_time"] = completion_time
        else:
            self.performance_metrics["tasks_failed"] += 1
        
        total_tasks = (self.performance_metrics["tasks_completed"] + 
                      self.performance_metrics["tasks_failed"])
        
        if total_tasks > 0:
            self.performance_metrics["success_rate"] = (
                self.performance_metrics["tasks_completed"] / total_tasks
            )
        
        self.performance_metrics["last_active"] = time.time()
    
    def add_capability(self, capability: str):
        """Add a new capability to the agent."""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
    
    def remove_capability(self, capability: str):
        """Remove a capability from the agent."""
        if capability in self.capabilities:
            self.capabilities.remove(capability)
    
    def update_resource_usage(self, tokens: int = 0, api_calls: int = 0, cost: float = 0.0):
        """Update resource usage metrics."""
        self.resource_usage["tokens_consumed"] += tokens
        self.resource_usage["api_calls"] += api_calls
        self.resource_usage["estimated_cost"] += cost


class Team:
    """A team of agents working together on related tasks."""
    
    def __init__(self, team_id: Optional[str] = None, name: str = "Generic Team"):
        self.team_id = team_id or str(uuid.uuid4())
        self.name = name
        self.agents = {}  # agent_id -> Agent
        self.status = "initialized"
        self.goals = []
        
        # Team performance metrics
        self.performance = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_response_time": 0,
            "creation_time": time.time(),
            "last_active": time.time()
        }
        
        # Team structure
        self.hierarchy = {}  # Role hierarchy (e.g., manager -> workers)
        self.communication_channels = []  # Defined communication paths
        
        # Team learning
        self.shared_knowledge = {}  # Knowledge shared across team members
        self.team_improvements = []  # Record of team improvements
    
    def add_agent(self, agent: Agent, role: str = "member"):
        """Add an agent to the team with a specific role."""
        self.agents[agent.agent_id] = {
            "agent": agent,
            "role": role,
            "join_time": time.time(),
            "contribution_score": 0.0
        }
        
        # Update hierarchy if needed
        if role == "manager" or role == "lead":
            self.hierarchy[agent.agent_id] = []
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the team."""
        if agent_id in self.agents:
            # Update hierarchy if needed
            for manager_id, subordinates in self.hierarchy.items():
                if agent_id in subordinates:
                    subordinates.remove(agent_id)
            
            if agent_id in self.hierarchy:
                del self.hierarchy[agent_id]
            
            del self.agents[agent_id]
    
    def establish_hierarchy(self, manager_id: str, subordinate_ids: List[str]):
        """Establish a management hierarchy within the team."""
        if manager_id in self.agents and all(sid in self.agents for sid in subordinate_ids):
            self.hierarchy[manager_id] = subordinate_ids
            
            # Update roles
            self.agents[manager_id]["role"] = "manager"
            for sid in subordinate_ids:
                self.agents[sid]["role"] = "subordinate"
    
    def update_performance(self, task_completed: bool, response_time: float = 0):
        """Update team performance metrics based on task execution."""
        if task_completed:
            self.performance["tasks_completed"] += 1
            
            # Update average response time
            prev_avg = self.performance["average_response_time"]
            prev_count = self.performance["tasks_completed"] - 1
            
            if prev_count > 0:
                self.performance["average_response_time"] = (
                    (prev_avg * prev_count + response_time) / 
                    self.performance["tasks_completed"]
                )
            else:
                self.performance["average_response_time"] = response_time
        else:
            self.performance["tasks_failed"] += 1
        
        self.performance["last_active"] = time.time()
    
    def add_shared_knowledge(self, key: str, value: Any):
        """Add knowledge that is shared across all team members."""
        self.shared_knowledge[key] = value
    
    def get_team_efficiency(self) -> float:
        """Calculate team efficiency score based on various metrics."""
        if self.performance["tasks_completed"] + self.performance["tasks_failed"] == 0:
            return 0.0
        
        # Calculate success rate
        success_rate = (self.performance["tasks_completed"] / 
                        (self.performance["tasks_completed"] + self.performance["tasks_failed"]))
        
        # Calculate agent diversity score (higher diversity is better up to a point)
        unique_capabilities = set()
        for agent_info in self.agents.values():
            unique_capabilities.update(agent_info["agent"].capabilities)
        
        diversity_score = min(1.0, len(unique_capabilities) / 10)  # Cap at 1.0
        
        # Calculate coordination score based on hierarchy
        coordination_score = min(1.0, len(self.hierarchy) / max(1, len(self.agents)))
        
        # Combine metrics
        return (success_rate * 0.6 + diversity_score * 0.2 + coordination_score * 0.2)


class AgentManager:
    """
    Manages the lifecycle of agents and teams in the EvoGenesis framework.
    
    Responsible for:
    - Creating and terminating agents
    - Forming and managing teams
    - Handling inter-agent communication
    - Tracking resource usage of agents
    - Monitoring agent and team performance
    - Dynamically scaling agent resources based on demand
    """
    
    def __init__(self, kernel):
        """
        Initialize the Agent Manager.
        
        Args:
            kernel: The EvoGenesis kernel instance
        """
        self.kernel = kernel
        self.agents = {}  # agent_id -> Agent
        self.teams = {}   # team_id -> Team
        self.resource_usage = {}  # agent_id -> resource stats
        
        # Agent types registry
        self.agent_types = {
            "generic": {"base_capabilities": []},
            "researcher": {"base_capabilities": ["search", "analyze", "summarize"]},
            "planner": {"base_capabilities": ["plan", "decompose", "prioritize"]},
            "executor": {"base_capabilities": ["execute", "debug", "monitor"]},
            "critic": {"base_capabilities": ["evaluate", "suggest", "improve"]},
            "coordinator": {"base_capabilities": ["coordinate", "communicate", "delegate"]}
        }
        
        # Communication protocols
        self.communication_protocols = {
            "direct": lambda sender, receiver, message: self._direct_communication(sender, receiver, message),
            "broadcast": lambda sender, receivers, message: self._broadcast_communication(sender, receivers, message),
            "hierarchical": lambda manager, team, message: self._hierarchical_communication(manager, team, message),
            "mcp": lambda sender, receiver, message: self._mcp_communication(sender, receiver, message)
        }
        
        # Performance monitoring
        self.performance_thresholds = {
            "min_success_rate": 0.6,
            "max_response_time": 60,  # seconds
            "cost_per_task_limit": 0.5,  # dollars
            "idle_timeout": 3600  # seconds
        }
        
        # Resource scaling settings
        self.scaling_settings = {
            "enable_auto_scaling": True,
            "min_agents": 1,
            "max_agents": 100,
            "scale_up_threshold": 0.8,  # 80% resource utilization
            "scale_down_threshold": 0.3  # 30% resource utilization
        }
    
    def start(self):
        """Start the Agent Manager module."""
        # Initialize core system agents if configured
        if self.kernel.config.get("create_system_agents", True):
            self._initialize_system_agents()
    
    def stop(self):
        """Stop the Agent Manager module."""
        # Terminate all agents
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            self.terminate_agent(agent_id)
        
        # Clear teams
        self.teams.clear()
    
    def get_status(self):
        """Get the current status of the Agent Manager."""
        active_agents = sum(1 for a in self.agents.values() if a.status == "active")
        idle_agents = sum(1 for a in self.agents.values() if a.status == "idle")
        total_cost = sum(a.resource_usage["estimated_cost"] for a in self.agents.values())
        
        return {
            "status": "active",
            "agent_count": len(self.agents),
            "active_agents": active_agents,
            "idle_agents": idle_agents,
            "team_count": len(self.teams),
            "total_resource_cost": total_cost
        }
    
    def _initialize_system_agents(self):
        """Initialize core system agents needed for basic operation."""
        # Create a system coordinator agent
        coordinator = self.create_agent(
            agent_type="coordinator",
            name="System Coordinator",
            capabilities=["system_coordination", "agent_management", "process_monitoring"],
            system_agent=True
        )
        
        # Create a system planner agent
        planner = self.create_agent(
            agent_type="planner",
            name="System Planner",
            capabilities=["strategic_planning", "task_decomposition", "resource_allocation"],
            system_agent=True
        )
        
        # Create a system monitor agent
        monitor = self.create_agent(
            agent_type="critic",
            name="System Monitor",
            capabilities=["performance_monitoring", "error_detection", "quality_control"],
            system_agent=True
        )
        
        # Form a core system team
        system_team = self.create_team(
            name="System Core Team",
            goal="Ensure efficient and effective operation of the EvoGenesis framework",
            agent_roles={
                "coordinator": coordinator.agent_id,
                "planner": planner.agent_id,
                "monitor": monitor.agent_id
            }
        )
        
        # Establish team hierarchy
        system_team.establish_hierarchy(
            coordinator.agent_id,
            [planner.agent_id, monitor.agent_id]
        )
    
    def get_agent_types(self) -> List[str]:
        """Get a list of available agent types."""
        return list(self.agent_types.keys())
    
    def create_agent(self, agent_type: str, name: str, 
                     capabilities: Optional[List[str]] = None, **kwargs) -> Agent:
        """
        Create a new agent with the specified type and capabilities.
        
        Args:
            agent_type: The type of agent to create
            name: The name for the new agent
            capabilities: List of capability identifiers
            **kwargs: Additional agent-specific parameters
            
        Returns:
            The newly created agent instance
        """
        # Merge base capabilities with provided capabilities
        all_capabilities = self.agent_types.get(agent_type, {}).get("base_capabilities", []).copy()
        if capabilities:
            for cap in capabilities:
                if cap not in all_capabilities:
                    all_capabilities.append(cap)
        
        # Use LLM Orchestrator to select appropriate model
        llm_config = self.kernel.llm_orchestrator.select_model(
            task_type="agent_execution",
            capabilities=all_capabilities,
            agent_type=agent_type
        )
        
        # Create the agent
        agent = Agent(
            name=name,
            capabilities=all_capabilities,
            llm_config=llm_config,
            agent_type=agent_type,
            **kwargs
        )
        
        # Notify HITL interface if it's not a system agent
        if not kwargs.get("system_agent", False):
            self.kernel.hitl_interface.notify_agent_created(agent.agent_id, agent.name, agent_type)
        
        self.agents[agent.agent_id] = agent
        return agent
    
    def create_team(self, name: str, goal: str, 
                    agent_roles: Optional[Dict[str, Any]] = None) -> Team:
        """
        Create a new team of agents with specified roles.
        
        Args:
            name: The name for the new team
            goal: The primary goal of the team
            agent_roles: Dictionary mapping role names to either capability lists or existing agent IDs
            
        Returns:
            The newly created team instance
        """
        team = Team(name=name)
        team.goals.append(goal)
        
        if agent_roles:
            for role, value in agent_roles.items():
                # If value is a string, treat it as an agent_id
                if isinstance(value, str) and value in self.agents:
                    agent = self.agents[value]
                    team.add_agent(agent, role=role)
                # If value is a list, treat it as capabilities and create a new agent
                elif isinstance(value, list):
                    agent = self.create_agent(
                        agent_type=role,
                        name=f"{name} {role}",
                        capabilities=value,
                        team_id=team.team_id
                    )
                    team.add_agent(agent, role=role)
        
        self.teams[team.team_id] = team
        
        # Notify HITL interface
        self.kernel.hitl_interface.notify_team_created(team.team_id, team.name, goal)
        
        return team
    
    def assign_role(self, agent_id: str, role: str, team_id: Optional[str] = None) -> bool:
        """
        Assign a role to an agent, optionally within a specific team.
        
        Args:
            agent_id: The ID of the agent
            role: The role to assign
            team_id: Optional team context for the role
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            return False
        
        # If team specified, update role within team
        if team_id and team_id in self.teams:
            team = self.teams[team_id]
            if agent_id in team.agents:
                team.agents[agent_id]["role"] = role
                return True
            return False
        
        # Otherwise, update agent's global role
        self.agents[agent_id].attributes["global_role"] = role
        return True
    
    def find_agents_for_task(self, task_description: str, 
                           task_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Find suitable agents for a task based on capabilities and performance.
        
        Args:
            task_description: Description of the task
            task_metadata: Additional metadata about the task
            
        Returns:
            List of dictionaries with agent_id and match_score
        """
        # Extract required capabilities from task description
        required_capabilities = self._extract_capabilities_from_task(
            task_description, task_metadata
        )
        
        # Score each agent based on capability match and performance
        agent_scores = []
        for agent_id, agent in self.agents.items():
            if agent.status == "terminated":
                continue
                
            # Calculate capability match score
            capability_match = sum(1 for cap in required_capabilities if cap in agent.capabilities)
            capability_score = capability_match / max(1, len(required_capabilities))
            
            # Get performance score
            performance_score = agent.performance_metrics["success_rate"]
            
            # Combine scores (weighted)
            combined_score = capability_score * 0.7 + performance_score * 0.3
            
            agent_scores.append({
                "agent_id": agent_id,
                "match_score": combined_score,
                "capability_match": capability_score,
                "performance_score": performance_score
            })
        
        # Sort by combined score
        agent_scores.sort(key=lambda x: x["match_score"], reverse=True)
        return agent_scores
    
    def _extract_capabilities_from_task(self, task_description: str, 
                                      task_metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Extract required capabilities from a task description.
        
        Args:
            task_description: Description of the task
            task_metadata: Additional metadata about the task
            
        Returns:
            List of required capability strings
        """
        # If metadata explicitly defines required capabilities, use those
        if task_metadata and "required_capabilities" in task_metadata:
            return task_metadata["required_capabilities"]
        
        # Otherwise, use LLM to extract capabilities from description
        capabilities_extraction = self.kernel.llm_orchestrator.execute_prompt(
            task_type="capability_extraction",
            prompt_template="extract_capabilities.jinja2",
            params={
                "task_description": task_description,
                "available_capabilities": [cap for agent_type in self.agent_types.values() 
                                          for cap in agent_type.get("base_capabilities", [])]
            }
        )
        
        return capabilities_extraction.get("capabilities", [])
    
    def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an agent and clean up its resources.
        
        Args:
            agent_id: The ID of the agent to terminate
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # First change status to terminating
            agent.status = "terminating"
            
            # Finalize any pending tasks
            for task_id in agent.assigned_tasks:
                self.kernel.task_planner.cancel_task(task_id)
            
            # Remove from any teams
            for team in self.teams.values():
                if agent_id in team.agents:
                    team.remove_agent(agent_id)
            
            # Archive agent data in memory
            self.kernel.memory_manager.archive_agent(
                agent_id=agent_id,
                agent_data={
                    "name": agent.name,
                    "capabilities": agent.capabilities,
                    "performance": agent.performance_metrics,
                    "resource_usage": agent.resource_usage,
                    "knowledge": agent.knowledge_base
                }
            )
            
            # Clean up resources
            if agent_id in self.resource_usage:
                del self.resource_usage[agent_id]
            
            # Mark as terminated but keep in registry for reference
            agent.status = "terminated"
            
            # Notify HITL interface
            self.kernel.hitl_interface.notify_agent_terminated(agent_id, agent.name)
            
            return True
        
        return False
    
    def monitor_agent_performance(self):
        """
        Monitor agent performance and take actions for underperforming agents.
        """
        current_time = time.time()
        
        for agent_id, agent in list(self.agents.items()):
            if agent.status == "terminated":
                continue
                
            # Check for idle timeout
            idle_time = current_time - agent.performance_metrics["last_active"]
            if idle_time > self.performance_thresholds["idle_timeout"]:
                # Terminate idle agent
                self.terminate_agent(agent_id)
                continue
            
            # Check success rate
            if (agent.performance_metrics["tasks_completed"] + 
                agent.performance_metrics["tasks_failed"] >= 5):  # Minimum sample size
                
                if agent.performance_metrics["success_rate"] < self.performance_thresholds["min_success_rate"]:
                    # Try to improve agent first
                    self._attempt_agent_improvement(agent_id)
                    
                    # If still underperforming after improvement attempt, consider replacement
                    if agent.performance_metrics["success_rate"] < self.performance_thresholds["min_success_rate"]:
                        self._replace_underperforming_agent(agent_id)
    
    def _attempt_agent_improvement(self, agent_id: str):
        """
        Attempt to improve an underperforming agent.
        
        Args:
            agent_id: The ID of the agent to improve
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return
        
        # Use LLM to suggest improvements
        improvement_suggestion = self.kernel.llm_orchestrator.execute_prompt(
            task_type="agent_improvement",
            prompt_template="improve_agent.jinja2",
            params={
                "agent_name": agent.name,
                "agent_type": agent.attributes.get("agent_type", "generic"),
                "current_capabilities": agent.capabilities,
                "performance_metrics": agent.performance_metrics,
                "failed_tasks": self.kernel.memory_manager.get_agent_failed_tasks(agent_id)
            }
        )
        
        # Implement suggested improvements
        if "add_capabilities" in improvement_suggestion:
            for capability in improvement_suggestion["add_capabilities"]:
                agent.add_capability(capability)
        
        if "update_llm" in improvement_suggestion and improvement_suggestion["update_llm"]:
            # Update LLM configuration
            new_llm_config = self.kernel.llm_orchestrator.select_model(
                task_type="agent_execution",
                capabilities=agent.capabilities,
                agent_type=agent.attributes.get("agent_type", "generic"),
                performance_requirements={
                    "priority": "quality"  # Prioritize quality over cost
                }
            )
            agent.attributes["llm_config"] = new_llm_config
        
        # Record improvement attempt
        agent.skill_improvements.append({
            "timestamp": time.time(),
            "reason": "performance_improvement",
            "changes": improvement_suggestion
        })
    
    def _replace_underperforming_agent(self, agent_id: str):
        """
        Replace an underperforming agent with a new one.
        
        Args:
            agent_id: The ID of the agent to replace
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return
        
        # Create replacement agent
        replacement = self.create_agent(
            agent_type=agent.attributes.get("agent_type", "generic"),
            name=f"{agent.name} (Replacement)",
            capabilities=agent.capabilities,
            replacing_agent_id=agent_id
        )
        
        # Transfer team memberships
        for team_id, team in self.teams.items():
            if agent_id in team.agents:
                role = team.agents[agent_id]["role"]
                team.remove_agent(agent_id)
                team.add_agent(replacement, role=role)
        
        # Transfer assigned tasks
        for task_id in agent.assigned_tasks:
            if self.kernel.task_planner.assign_task(task_id, replacement.agent_id):
                agent.assigned_tasks.remove(task_id)
                replacement.assigned_tasks.append(task_id)
        
        # Terminate underperforming agent
        self.terminate_agent(agent_id)
        
        # Notify HITL interface
        self.kernel.hitl_interface.notify_agent_replaced(
            old_agent_id=agent_id,
            new_agent_id=replacement.agent_id,
            reason="Underperforming agent replacement"
        )
    
    def scale_agents_based_on_demand(self, current_task_load: int):
        """
        Scale up or down the number of agents based on task load.
        
        Args:
            current_task_load: The current number of pending tasks
        """
        if not self.scaling_settings["enable_auto_scaling"]:
            return
        
        active_agents = sum(1 for a in self.agents.values() 
                           if a.status != "terminated" and not a.attributes.get("system_agent", False))
        
        # Calculate utilization
        utilization = current_task_load / max(1, active_agents)
        
        # Scale up if needed
        if (utilization > self.scaling_settings["scale_up_threshold"] and 
            active_agents < self.scaling_settings["max_agents"]):
            
            # Determine agent types needed based on task analysis
            needed_agent_types = self._analyze_task_agent_needs()
            
            # Create new agents
            for agent_type, count in needed_agent_types.items():
                for i in range(count):
                    self.create_agent(
                        agent_type=agent_type,
                        name=f"Auto-Scaled {agent_type.capitalize()} {i+1}",
                        auto_scaled=True
                    )
        
        # Scale down if needed
        elif (utilization < self.scaling_settings["scale_down_threshold"] and 
              active_agents > self.scaling_settings["min_agents"]):
            
            # Find auto-scaled agents with low utilization
            auto_scaled_agents = [
                agent_id for agent_id, agent in self.agents.items()
                if agent.attributes.get("auto_scaled", False) and len(agent.assigned_tasks) == 0
            ]
            
            # Sort by lowest performance
            auto_scaled_agents.sort(
                key=lambda id: self.agents[id].performance_metrics["success_rate"]
            )
            
            # Calculate how many to remove
            to_remove = min(
                len(auto_scaled_agents),
                active_agents - self.scaling_settings["min_agents"]
            )
            
            # Remove agents
            for i in range(to_remove):
                self.terminate_agent(auto_scaled_agents[i])
    
    def _analyze_task_agent_needs(self) -> Dict[str, int]:
        """
        Analyze pending tasks to determine agent types needed.
        
        Returns:
            Dictionary mapping agent types to counts needed
        """
        pending_tasks = self.kernel.task_planner.get_pending_tasks()
        
        # Use LLM to analyze task requirements
        analysis = self.kernel.llm_orchestrator.execute_prompt(
            task_type="task_agent_analysis",
            prompt_template="analyze_agent_needs.jinja2",
            params={
                "pending_tasks": [
                    {"id": task.task_id, "name": task.name, "description": task.description}
                    for task in pending_tasks
                ],
                "available_agent_types": self.get_agent_types()
            }
        )
        
        return analysis.get("agent_needs", {})
    
    def send_message(self, from_agent_id: str, to_agent_id: str, 
                     message: Dict[str, Any], protocol: str = "direct") -> bool:
        """
        Send a message from one agent to another using specified protocol.
        
        Args:
            from_agent_id: The ID of the sending agent
            to_agent_id: The ID of the receiving agent
            message: The message content
            protocol: The communication protocol to use
            
        Returns:
            True if successful, False otherwise
        """
        if protocol not in self.communication_protocols:
            return False
        
        # Add message to memory for tracking
        self.kernel.memory_manager.store_message(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            message=message,
            protocol=protocol
        )
        
        # Use the specified protocol
        return self.communication_protocols[protocol](from_agent_id, to_agent_id, message)
    
    def _direct_communication(self, sender_id: str, receiver_id: str, message: Dict[str, Any]) -> bool:
        """Direct point-to-point communication protocol."""
        if sender_id not in self.agents or receiver_id not in self.agents:
            return False
        
        # In a real implementation, this would handle the actual message delivery mechanism
        return True
    
    def _broadcast_communication(self, sender_id: str, receiver_ids: List[str], 
                              message: Dict[str, Any]) -> bool:
        """Broadcast communication protocol for one-to-many messaging."""
        if sender_id not in self.agents:
            return False
        
        success = True
        for receiver_id in receiver_ids:
            if receiver_id in self.agents:
                success = success and self._direct_communication(sender_id, receiver_id, message)
        
        return success
    
    def _hierarchical_communication(self, manager_id: str, team_id: str, 
                                 message: Dict[str, Any]) -> bool:
        """Hierarchical communication following team structure."""
        if manager_id not in self.agents or team_id not in self.teams:
            return False
        
        team = self.teams[team_id]
        
        # Verify sender is a manager in the team
        if manager_id not in team.hierarchy:
            return False
        
        # Send to all subordinates
        success = True
        for subordinate_id in team.hierarchy[manager_id]:
            success = success and self._direct_communication(manager_id, subordinate_id, message)
        
        return success
    
    def _mcp_communication(self, sender_id: str, receiver_id: str, 
                        message: Dict[str, Any]) -> bool:
        """Model Context Protocol communication with structured format."""
        if sender_id not in self.agents or receiver_id not in self.agents:
            return False
        
        # Ensure message follows MCP format
        if not self._validate_mcp_message(message):
            return False
        
        # Process MCP message
        return self._direct_communication(sender_id, receiver_id, message)
    
    def _validate_mcp_message(self, message: Dict[str, Any]) -> bool:
        """Validate if a message follows the Model Context Protocol format."""
        required_fields = ["type", "content", "timestamp"]
        return all(field in message for field in required_fields)
    
    def get_team_agents(self, team_id: str) -> List[Agent]:
        """
        Get all agents in a specific team.
        
        Args:
            team_id: The ID of the team
            
        Returns:
            List of agents in the team
        """
        if team_id in self.teams:
            return [info["agent"] for info in self.teams[team_id].agents.values()]
        return []
