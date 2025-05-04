"""
Agent Factory Module - Handles the lifecycle of agents and teams in EvoGenesis.

This module is responsible for creating, monitoring, and managing agents and teams,
handling inter-agent communication, and tracking resource usage. It includes a
prompt compiler that can generate specialized prompts from a canonical template.
"""

from typing import Dict, Any, List, Optional
import uuid
import time
from functools import lru_cache
import os, json

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
        
        # Persistence attributes
        self.persistent_identity = kwargs.get("persistent_identity", None)
        self.memory_namespace = kwargs.get("memory_namespace", None)
        self.is_persistent = bool(self.persistent_identity) or kwargs.get("persistent_mission", False)
        self.last_checkpoint = None
        
        # Performance tracking
        self.metrics = {
            "tasks_completed": 0,
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "total_execution_time": 0.0
        }
        
        # System properties
        self.creation_time = time.time()
        self.last_active_time = self.creation_time

class AgentFactory:
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
        Initialize the Agent Factory.
        
        Args:
            kernel: The EvoGenesis kernel instance
        """
        self.kernel = kernel
        self.agents = {}  # agent_id -> Agent
        self.teams = {}   # team_id -> Team
        self.resource_usage = {}  # agent_id -> resource stats
        self.logger = kernel.logger # Initialize logger from kernel
        
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
            "direct": "Direct message between two agents",
            "broadcast": "Message to all agents in a team",
            "hierarchical": "Message passed through team hierarchy"
        }
        
        # Prompt templates for different agent types
        self.prompt_templates = {
            "generic": """You are a versatile AI agent named {name} with the following capabilities: {capabilities}.
                       Your current status is: {status}.""",
            "researcher": """You are a research specialist AI agent named {name}. 
                          You excel at gathering and analyzing information on {domain}.
                          Use these capabilities to provide thorough research: {capabilities}.""",
            "planner": """You are a strategic planning AI agent named {name}.
                       Your purpose is to create well-structured plans for {domain} tasks.
                       Leverage these capabilities to develop effective plans: {capabilities}."""
        }
    
    @lru_cache(maxsize=32)
    def _get_domain_knowledge(self, domain: str) -> str:
        """
        Retrieve domain-specific knowledge snippets from a knowledge base.
        First attempts to fetch from the kernel's memory_manager (if available),
        then falls back to loading a file from the configured knowledge_base_path.
        Caches results in-memory for performance.
        """
        kb_text = ""

        # 1. Try memory manager
        try:
            mm = getattr(self.kernel, "memory_manager", None)
            if mm and hasattr(mm, "get_knowledge"):
                kb_text = mm.get_knowledge(domain)
                if kb_text:
                    return kb_text
        except Exception as e:
            self.kernel.logger.warning(f"Memory manager lookup failed for '{domain}': {e}")

        # 2. Try file-based KB
        base_path = self.kernel.config.get("knowledge_base_path", "")
        if base_path:
            file_path = os.path.join(base_path, f"{domain}.md")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    kb_text = f.read().strip()
                    if kb_text:
                        return kb_text
            except FileNotFoundError:
                self.kernel.logger.info(f"No KB file found for domain '{domain}' at {file_path}")
            except Exception as e:
                self.kernel.logger.warning(f"Error reading KB file for '{domain}': {e}")

        # 3. Fallback to built-in snippets
        defaults = {
            "finance": "Finance domain includes knowledge of markets, investments, and regulations.",
            "healthcare": "Healthcare domain includes medical terminology and protocols.",
            "legal": "Legal domain includes case law, statutes, and legal procedures."
        }
        return defaults.get(domain, "")
    
    def start(self):
        """Start the Agent Factory module."""
        # Restore persistent agents if any exist
        self._restore_persistent_agents()
        
        # Initialize core system agents if configured
        if self.kernel.config.get("create_system_agents", True):
            self._initialize_system_agents()
    def _restore_persistent_agents(self):
        """Attempt to restore persistent agents from long-term memory."""
        if not hasattr(self.kernel, 'memory_manager') or not self.kernel.memory_manager.long_term:
            self.logger.warning("Long-term memory store not available for restoring agents.")
            return

        self.logger.info("Attempting to restore persistent agents...")
        try:
            # Assuming persistent agents are stored in a specific namespace, e.g., 'persistent_agents'
            # And assuming they can be retrieved by searching metadata.
            # The exact method and parameters depend heavily on the LongTermMemoryStore implementation.
            # Replacing the non-existent 'list_items' with a plausible 'search' call.
            # We search with an empty query embedding and filter by metadata.
            # This might need adjustment based on the actual memory store capabilities.
            agent_data_list = self.kernel.memory_manager.long_term.search(
                namespace="persistent_agents",
                query_embedding=[], # Pass empty or None if the backend allows metadata-only search
                limit=1000, # Adjust limit as needed
                filter_metadata={"item_type": "persistent_agent"} # Example filter
            )

            if not agent_data_list:
                self.logger.info("No persistent agents found to restore.")
                return

            restored_count = 0
            for agent_data in agent_data_list:
                metadata = agent_data.get('metadata', {})
                agent_id = agent_data.get('id')
                # Extract necessary info from metadata to recreate the agent
                name = metadata.get('name', f"RestoredAgent_{agent_id[:8]}")
                agent_type = metadata.get('agent_type', 'generic')
                capabilities = metadata.get('capabilities', [])
                # Add other relevant attributes from metadata...

                if agent_id and agent_id not in self.agents:
                    agent = self.create_agent(
                        agent_type=agent_type,
                        name=name,
                        capabilities=capabilities,
                        agent_id=agent_id, # Use the original ID
                        persistent_identity=metadata.get('persistent_identity'),
                        memory_namespace=metadata.get('memory_namespace'),
                        # Restore other attributes from metadata...
                        status="restored" # Set status
                    )
                    self.logger.info(f"Restored persistent agent: {agent.name} (ID: {agent.agent_id})")
                    restored_count += 1
                elif agent_id in self.agents:
                     self.logger.warning(f"Agent with ID {agent_id} already exists, skipping restoration.")

            self.logger.info(f"Successfully restored {restored_count} persistent agents.")

        except AttributeError as ae:
             self.logger.error(f"Memory store interface mismatch during agent restoration: {ae}. The 'search' method might not support metadata filtering as expected.")
             # Log specific error about search if it fails
        except Exception as e:
            # Log the specific exception 'e'
            self.logger.error(f"Error during persistent agent restoration: {e}", exc_info=True)
    def stop(self):
        """Stop the Agent Factory module."""
        # Save or checkpoint any persistent agents
        for agent_id, agent in self.agents.items():
            if agent.is_persistent:
                self._checkpoint_agent(agent)
        
        # Terminate all agents
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            self.terminate_agent(agent_id)
        
        # Clear teams
        self.teams.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Agent Factory.
        
        Returns:
            A dictionary containing status information
        """
        active_agents = sum(1 for a in self.agents.values() if a.status == "active")
        active_teams = sum(1 for t in self.teams.values() if hasattr(t, "status") and t.status == "active")
        
        return {
            "status": "active",
            "active_agents": active_agents,
            "total_agents": len(self.agents),
            "active_teams": active_teams,
            "total_teams": len(self.teams),
            "agent_types": list(self.agent_types.keys())
        }
        
    def list_agents(self) -> Dict[str, Any]:
        """
        List all agents managed by this factory.
        
        Returns:
            A dictionary of agent_id -> Agent
        """
        return self.agents
    
    def _initialize_system_agents(self):
        """Initialize core system agents needed for basic operation."""
        # Create a coordinator agent that will manage the coordination between other agents
        coordinator = self.create_agent(
            agent_type="coordinator",
            name="System Coordinator",
            capabilities=["coordination", "delegation", "monitoring"],
            system_agent=True
        )
        
        # Create a planning agent for decomposing user requests
        planner = self.create_agent(
            agent_type="planner",
            name="System Planner",
            capabilities=["planning", "decomposition", "prioritization"],
            system_agent=True
        )
        
        # Create a monitoring agent to track system health
        monitor = self.create_agent(
            agent_type="critic",
            name="System Monitor",
            capabilities=["monitoring", "evaluation", "reporting"],
            system_agent=True
        )
        
        # Form a system team with these agents
        system_team = self.create_team(
            name="System Team",
            members={
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
        Create a new agent with specified type and capabilities.
        
        Args:
            agent_type: Type of agent to create
            name: Name of the agent
            capabilities: List of capabilities the agent should have
            **kwargs: Additional agent attributes
            
        Returns:
            The created Agent instance
        """
        # Get base capabilities for this agent type
        base_capabilities = self.agent_types.get(agent_type, {}).get("base_capabilities", [])
        
        # Merge with provided capabilities
        all_capabilities = list(set(base_capabilities + (capabilities or [])))
        
        # Create the agent
        agent = Agent(
            name=name,
            capabilities=all_capabilities,
            **kwargs
        )
        
        # Store the agent
        self.agents[agent.agent_id] = agent
        
        # Initialize agent's resource tracking
        self.resource_usage[agent.agent_id] = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "api_calls": 0,
            "tokens_used": 0
        }
        
        return agent
    
    def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an agent by its ID.
        
        Args:
            agent_id: ID of the agent to terminate
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            return False
            
        # Save persistent agent state if needed
        agent = self.agents[agent_id]
        if agent.is_persistent:
            self._checkpoint_agent(agent)
            
        # Remove from resource tracking
        if agent_id in self.resource_usage:
            del self.resource_usage[agent_id]
            
        # Remove from teams
        for team_id, team in self.teams.items():
            if agent_id in team.members.values():
                team.remove_member(agent_id)
                
        # Delete the agent
        del self.agents[agent_id]
        return True
    def _checkpoint_agent(self, agent: Agent):
        """Save agent state for persistent agents."""
        agent.last_checkpoint = time.time()
        # Build agent state
        state = {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "capabilities": agent.capabilities,
            "status": agent.status,
            "assigned_tasks": agent.assigned_tasks,
            "metrics": agent.metrics,
            "creation_time": agent.creation_time,
            "last_active_time": agent.last_active_time,
            "last_checkpoint": agent.last_checkpoint,
            "attributes": agent.attributes,
            "persistent_identity": agent.persistent_identity,
            "memory_namespace": agent.memory_namespace,
            "persistent_mission": agent.is_persistent,
            "resource_usage": self.resource_usage.get(agent.agent_id, {})
        }
        try:
            mm = getattr(self.kernel, "memory_manager", None)
            # Prefer memory manager
            if mm and hasattr(mm, "save_agent_state"):
                mm.save_agent_state(agent.agent_id, state)
                self.kernel.logger.info(
                    f"Persisted agent {agent.name} ({agent.agent_id}) via memory_manager"
                )
            else:
                # Fallback to file-based storage
                path = self.kernel.config.get("agent_state_path", "")
                if not path:
                    raise RuntimeError("No memory_manager and no agent_state_path configured")
                os.makedirs(path, exist_ok=True)
                file_path = os.path.join(path, f"{agent.agent_id}.json")
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                self.kernel.logger.info(
                    f"Saved agent state to file {file_path}"
                )
        except Exception as e:
            self.kernel.logger.error(
                f"Failed to checkpoint agent {agent.agent_id}: {e}"
            )
    def create_team(self, name: str, members: Dict[str, str], **kwargs) -> 'Team':
        """Create a new team of agents."""
        # Check if Swarm Coordinator is enabled, as teams are related to swarm functionality
        swarm_coordinator = self.kernel.get_module("swarm_coordinator")
        if not swarm_coordinator:
             self.logger.warning("Swarm Coordinator not available or enabled. Team functionality might be limited.")
             # Decide if team creation should proceed or fail.
             # For now, let's allow creation but log the warning.
             # If teams strictly require the coordinator, raise an error:
             # raise RuntimeError("Swarm Coordinator is required to create teams.")

        team_id = str(uuid.uuid4())
        team = Team(team_id, name, members, **kwargs)
        self.teams[team_id] = team
        self.logger.info(f"Created team '{name}' ({team_id}) with members: {members}")
        # Log activity
        if hasattr(self.kernel, 'log_activity'):
            self.kernel.log_activity(
                activity_type="team.create",
                title=f"Team Created: {name}",
                message=f"Team '{name}' ({team_id}) created.",
                data={"team_id": team_id, "name": name, "members": members}
            )
        return team
    
    def dissolve_team(self, team_id: str) -> bool:
        """
        Dissolve a team by its ID.
        
        Args:
            team_id: ID of the team to dissolve
            
        Returns:
            True if successful, False otherwise
        """
        if team_id not in self.teams:
            return False
            
        del self.teams[team_id]
        return True
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def get_team(self, team_id: str) -> Optional['Team']:
        """Get team by ID."""
        return self.teams.get(team_id)
    
    def compile_prompt(self, agent_id: str, prompt_template: str, context: Dict[str, Any]) -> str:
        """
        Compile a prompt for an agent using a template and context.
        
        Args:
            agent_id: ID of the agent
            prompt_template: Base template for the prompt
            context: Context variables to fill into the template
            
        Returns:
            The compiled prompt
        """
        agent = self.get_agent(agent_id)
        if not agent:
            return ""
            
        # Get agent info
        agent_spec = {
            "name": agent.name,
            "capabilities": ", ".join(agent.capabilities),
            "status": agent.status
        }
        
        # Merge with context
        prompt_context = {**agent_spec, **context}
        
        # Format the template
        prompt = prompt_template.format(**prompt_context)
        
        # Add domain knowledge if specified
        if "domain" in context:
            domain_knowledge = self._get_domain_knowledge(context["domain"])
            if domain_knowledge:
                prompt += f"\n\nDomain Knowledge:\n{domain_knowledge}"
                
        # Add allowlist of tools based on permissions
        allowed_tools = agent_spec.get("allowed_tools", [])
        if allowed_tools:
            prompt += f"\n<tool_whitelist>\n{', '.join(allowed_tools)}\n</tool_whitelist>"
        
        return prompt

class Team:
    """Represents a team of agents working together."""
    
    def __init__(self, team_id: str, name: str, members: Dict[str, str], **kwargs):
        self.team_id = team_id
        self.name = name
        self.members = members  # role -> agent_id
        self.hierarchy = kwargs.get("hierarchy", {})  # agent_id -> [agent_ids under it]
        self.communication_protocol = kwargs.get("communication_protocol", "direct")
        self.creation_time = time.time()
        self.attributes = kwargs
    
    def add_member(self, role: str, agent_id: str) -> bool:
        """Add a member to the team."""
        if role in self.members:
            return False
            
        self.members[role] = agent_id
        return True
    
    def remove_member(self, agent_id: str) -> bool:
        """Remove a member from the team by agent ID."""
        for role, member_id in list(self.members.items()):
            if member_id == agent_id:
                del self.members[role]
                
                # Also remove from hierarchy
                if agent_id in self.hierarchy:
                    del self.hierarchy[agent_id]
                    
                for manager_id, subordinates in self.hierarchy.items():
                    if agent_id in subordinates:
                        subordinates.remove(agent_id)
                        
                return True
                
        return False
    
    def establish_hierarchy(self, manager_id: str, subordinate_ids: List[str]) -> bool:
        """Set up hierarchical relationships between team members."""
        if manager_id not in self.members.values():
            return False
            
        # Validate all subordinates are team members
        for sub_id in subordinate_ids:
            if sub_id not in self.members.values():
                return False
                
        # Set the hierarchy
        self.hierarchy[manager_id] = subordinate_ids
        return True
    
    def get_members(self) -> Dict[str, str]:
        """Get all team members."""
        return self.members.copy()

# Export the AgentFactory class for importing
__all__ = ["AgentFactory", "Agent", "Team"]
