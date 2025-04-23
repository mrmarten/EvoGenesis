"""
AutoGen Adapter - Implements adapter for Microsoft AutoGen framework.

This adapter enables EvoGenesis to use Microsoft AutoGen for multi-agent systems,
mapping EvoGenesis concepts to AutoGen agents, conversations, and execution patterns.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Union, Callable
import uuid
import os

# Conditionally import AutoGen
try:
    import autogen
    from autogen import Agent, AssistantAgent, UserProxyAgent, config_list_from_json
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

from evogenesis_core.adapters.base_adapter import AgentExecutionAdapter


class AutoGenAdapter(AgentExecutionAdapter):
    """
    Adapter for Microsoft AutoGen framework.
    
    Maps EvoGenesis agents and tasks to AutoGen concepts:
    - Agents → AutoGen AssistantAgent or UserProxyAgent
    - Team → AutoGen agent group with conversation
    - Tasks → AutoGen conversation/task executions
    """
    
    def __init__(self):
        """Initialize the AutoGen adapter."""
        if not AUTOGEN_AVAILABLE:
            raise ImportError("AutoGen is not available. Install with 'pip install pyautogen'")
        
        self.agents = {}  # agent_id -> AutoGen agent
        self.teams = {}   # team_id -> Dict with team info
        self.agent_configs = {}  # agent_id -> config
        self.active_tasks = {}  # task_id -> task_info
        self.agent_status = {}  # agent_id -> status
        self.conversations = {}  # conversation_id -> conversation object
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the AutoGen adapter with configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.global_config = config
            
            # Set up LLM config list for AutoGen
            self.config_list = []
            
            # If a specific config file is provided, use it
            config_file = config.get("autogen_config_file")
            if config_file and os.path.exists(config_file):
                self.config_list = config_list_from_json(config_file)
            else:
                # Otherwise, create from provided API keys
                for provider, key_info in config.get("api_keys", {}).items():
                    if provider == "openai" and key_info.get("api_key"):
                        self.config_list.append({
                            "model": "gpt-4o",
                            "api_key": key_info["api_key"]
                        })
                    elif provider == "anthropic" and key_info.get("api_key"):
                        self.config_list.append({
                            "model": "claude-3-opus-20240229",
                            "api_key": key_info["api_key"]
                        })
            
            logging.info("AutoGen adapter initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize AutoGen adapter: {str(e)}")
            return False
    
    async def create_agent(self, agent_spec: Dict[str, Any]) -> str:
        """
        Create an agent using AutoGen.
        
        Args:
            agent_spec: Specification of the agent to create
            
        Returns:
            Agent ID
        """
        try:
            # Generate an ID for this agent
            agent_id = str(uuid.uuid4())
            
            # Extract agent configuration
            agent_name = agent_spec.get("name", f"Agent-{agent_id[:8]}")
            agent_type = agent_spec.get("type", "assistant")
            agent_description = agent_spec.get("description", "A helpful assistant")
            
            # Get LLM config
            llm_config = agent_spec.get("llm_config", {})
            model = llm_config.get("model_name", "gpt-4o")
            
            # Prepare AutoGen config
            llm_config = {
                "config_list": self.config_list,
                "temperature": llm_config.get("temperature", 0.7),
                "max_tokens": llm_config.get("max_tokens", 2000)
            }
            
            # Create the appropriate agent type
            if agent_type == "assistant":
                # Create an AssistantAgent
                autogen_agent = AssistantAgent(
                    name=agent_name,
                    system_message=agent_description,
                    llm_config=llm_config
                )
            
            elif agent_type == "user_proxy":
                # Create a UserProxyAgent
                exec_mode = agent_spec.get("execution_mode", "local")
                
                # Determine which tools to enable
                tools = []
                if agent_spec.get("enable_file_tools", False):
                    tools.append("file")
                if agent_spec.get("enable_web_tools", False):
                    tools.append("web")
                if agent_spec.get("enable_code_execution", False):
                    tools.append("code")
                
                autogen_agent = UserProxyAgent(
                    name=agent_name,
                    human_input_mode="NEVER",  # Automated mode for EvoGenesis
                    system_message=agent_description,
                    code_execution_config={
                        "executor": exec_mode,
                        "use_docker": agent_spec.get("use_docker", False),
                        "work_dir": agent_spec.get("work_dir", "autogen_output"),
                        "last_n_messages": agent_spec.get("last_n_messages", 3),
                        "tools": tools
                    } if agent_spec.get("enable_code_execution", False) else None
                )
            
            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")
            
            # Store the agent
            self.agents[agent_id] = autogen_agent
            self.agent_configs[agent_id] = agent_spec
            self.agent_status[agent_id] = {
                "status": "initialized",
                "tasks_completed": 0,
                "current_task": None,
                "last_active": asyncio.get_event_loop().time()
            }
            
            logging.info(f"Created AutoGen agent {agent_id} of type {agent_type}")
            return agent_id
            
        except Exception as e:
            logging.error(f"Failed to create AutoGen agent: {str(e)}")
            raise
    
    async def run_agent_task(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task using AutoGen.
        
        Args:
            agent_id: ID of the agent to use
            task: Task specification
            
        Returns:
            Task execution results
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        task_id = task.get("task_id", str(uuid.uuid4()))
        
        try:
            # Update agent status
            self.agent_status[agent_id]["status"] = "running"
            self.agent_status[agent_id]["current_task"] = task_id
            self.agent_status[agent_id]["last_active"] = asyncio.get_event_loop().time()
            
            # Store task information
            self.active_tasks[task_id] = {
                "agent_id": agent_id,
                "task": task,
                "status": "running",
                "start_time": asyncio.get_event_loop().time()
            }
            
            # Extract task information
            task_type = task.get("type", "general")
            task_content = task.get("content", "")
            
            # Create a conversation log for capturing output
            conversation_id = f"conv-{task_id}"
            self.conversations[conversation_id] = {
                "messages": [],
                "summary": None
            }
            
            # Execute based on task type
            if task_type == "direct":
                # Simple message to an agent
                if isinstance(agent, AssistantAgent):
                    # For an AssistantAgent, we need a UserProxyAgent to talk to it
                    proxy = UserProxyAgent(
                        name="TaskUser",
                        human_input_mode="NEVER"
                    )
                    response = await asyncio.to_thread(
                        proxy.initiate_chat, 
                        agent,
                        message=task_content
                    )
                    chat_history = agent.chat_messages[proxy]
                    
                else:
                    # For a UserProxyAgent, which typically executes code
                    response = await asyncio.to_thread(
                        agent.generate_reply, 
                        messages=[{"role": "user", "content": task_content}]
                    )
                    chat_history = [{"role": "user", "content": task_content}, 
                                    {"role": "assistant", "content": response}]
            
            elif task_type == "group":
                # Group conversation requires a specified team
                team_id = task.get("team_id")
                if not team_id or team_id not in self.teams:
                    raise ValueError(f"Valid team_id required for group task")
                
                team = self.teams[team_id]
                initiator_id = task.get("initiator_id", team["manager_id"])
                responder_ids = task.get("responder_ids", [a_id for a_id in team["agents"] if a_id != initiator_id])
                
                # Get the actual agents
                initiator = self.agents[initiator_id]
                responders = [self.agents[a_id] for a_id in responder_ids]
                
                # Initiate group chat
                chat_result = await asyncio.to_thread(
                    initiator.initiate_chat,
                    responders,
                    message=task_content,
                    max_turns=task.get("max_turns", 10)
                )
                
                # Extract conversation history
                chat_history = []
                for agent_id in [initiator_id] + responder_ids:
                    if agent_id in chat_result:
                        chat_history.extend(chat_result[agent_id])
                
                response = "Group conversation completed"
                
            else:
                raise ValueError(f"Unsupported task type for AutoGen: {task_type}")
            
            # Store conversation history
            self.conversations[conversation_id]["messages"] = chat_history
            
            # Create a summary if multiple messages
            if len(chat_history) > 1:
                # Get the final output (last assistant message)
                for msg in reversed(chat_history):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        final_output = msg.get("content", "")
                        break
                    elif hasattr(msg, "role") and msg.role == "assistant":
                        final_output = msg.content
                        break
                else:
                    final_output = str(chat_history[-1])
            else:
                final_output = response
            
            # Update task and agent status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["end_time"] = asyncio.get_event_loop().time()
            self.active_tasks[task_id]["result"] = final_output
            
            self.agent_status[agent_id]["status"] = "idle"
            self.agent_status[agent_id]["current_task"] = None
            self.agent_status[agent_id]["tasks_completed"] += 1
            self.agent_status[agent_id]["last_active"] = asyncio.get_event_loop().time()
            
            # Return result
            return {
                "task_id": task_id,
                "agent_id": agent_id,
                "conversation_id": conversation_id,
                "status": "completed",
                "result": final_output,
                "execution_time": self.active_tasks[task_id]["end_time"] - self.active_tasks[task_id]["start_time"]
            }
            
        except Exception as e:
            # Update task and agent status on error
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["error"] = str(e)
                self.active_tasks[task_id]["end_time"] = asyncio.get_event_loop().time()
            
            self.agent_status[agent_id]["status"] = "error"
            self.agent_status[agent_id]["current_task"] = None
            self.agent_status[agent_id]["last_active"] = asyncio.get_event_loop().time()
            
            logging.error(f"Failed to execute task {task_id} with agent {agent_id}: {str(e)}")
            
            return {
                "task_id": task_id,
                "agent_id": agent_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the current status of an AutoGen agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with agent status information
        """
        if agent_id not in self.agent_status:
            raise ValueError(f"Agent {agent_id} not found")
        
        status_info = self.agent_status[agent_id].copy()
        
        # Add some AutoGen-specific information
        agent = self.agents[agent_id]
        status_info["agent_name"] = agent.name
        status_info["agent_type"] = "assistant" if isinstance(agent, AssistantAgent) else "user_proxy"
        
        return status_info
    
    async def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an AutoGen agent.
        
        Args:
            agent_id: ID of the agent to terminate
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            return False
        
        try:
            # Remove the agent
            agent = self.agents[agent_id]
            
            # Clean up any resources
            agent.reset()
            
            # Remove from our records
            del self.agents[agent_id]
            del self.agent_configs[agent_id]
            self.agent_status[agent_id] = {"status": "terminated"}
            
            # Remove from any teams
            for team_id, team in self.teams.items():
                if agent_id in team["agents"]:
                    team["agents"].remove(agent_id)
            
            return True
        except Exception as e:
            logging.error(f"Failed to terminate agent {agent_id}: {str(e)}")
            return False
    
    async def pause_agent(self, agent_id: str) -> bool:
        """
        Pause an AutoGen agent (note: limited concept in AutoGen).
        
        Args:
            agent_id: ID of the agent to pause
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agent_status:
            return False
        
        # AutoGen doesn't have direct pause/resume, but we can track status
        self.agent_status[agent_id]["status"] = "paused"
        return True
    
    async def resume_agent(self, agent_id: str) -> bool:
        """
        Resume a paused AutoGen agent.
        
        Args:
            agent_id: ID of the agent to resume
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agent_status:
            return False
        
        if self.agent_status[agent_id]["status"] == "paused":
            self.agent_status[agent_id]["status"] = "idle"
            return True
        
        return False
    
    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an AutoGen agent's configuration.
        
        Args:
            agent_id: ID of the agent to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            return False
        
        try:
            agent = self.agents[agent_id]
            
            # Update the agent configuration
            self.agent_configs[agent_id].update(updates)
            
            # Apply applicable updates
            if "name" in updates:
                agent.name = updates["name"]
            
            if "system_message" in updates:
                agent.system_message = updates["system_message"]
            
            if "llm_config" in updates and isinstance(agent, AssistantAgent):
                # Update LLM configuration
                llm_config = updates["llm_config"]
                new_config = {
                    "config_list": self.config_list,
                    "temperature": llm_config.get("temperature", 0.7),
                    "max_tokens": llm_config.get("max_tokens", 2000)
                }
                agent.llm_config = new_config
            
            return True
        except Exception as e:
            logging.error(f"Failed to update agent {agent_id}: {str(e)}")
            return False
    
    async def create_team(self, team_spec: Dict[str, Any]) -> str:
        """
        Create a team of AutoGen agents.
        
        Args:
            team_spec: Specification of the team to create
            
        Returns:
            Team ID
        """
        team_id = str(uuid.uuid4())
        
        try:
            # Create a team configuration
            team_name = team_spec.get("name", f"Team-{team_id[:8]}")
            member_specs = team_spec.get("members", [])
            
            # Create or use existing agents
            team_agents = []
            for member_spec in member_specs:
                if "agent_id" in member_spec and member_spec["agent_id"] in self.agents:
                    # Use existing agent
                    team_agents.append(member_spec["agent_id"])
                else:
                    # Create a new agent
                    agent_id = await self.create_agent(member_spec)
                    team_agents.append(agent_id)
            
            # Identify manager (defaults to first agent)
            manager_id = team_spec.get("manager_id", team_agents[0] if team_agents else None)
            
            # Store team information
            self.teams[team_id] = {
                "id": team_id,
                "name": team_name,
                "agents": team_agents,
                "manager_id": manager_id,
                "creation_time": asyncio.get_event_loop().time(),
                "description": team_spec.get("description", "")
            }
            
            logging.info(f"Created AutoGen team {team_id} with {len(team_agents)} agents")
            return team_id
            
        except Exception as e:
            logging.error(f"Failed to create team: {str(e)}")
            raise
    
    async def get_framework_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of AutoGen.
        
        Returns:
            Dictionary describing AutoGen's capabilities
        """
        return {
            "name": "AutoGen",
            "version": autogen.__version__,
            "features": {
                "multi_agent": True,
                "code_execution": True,
                "file_operations": True,
                "web_search": True,
                "group_chat": True,
                "tool_use": True,
                "human_feedback": True
            },
            "agent_types": ["assistant", "user_proxy"],
            "supported_models": ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"]
        }
    
    async def shutdown(self) -> bool:
        """
        Shut down the AutoGen adapter cleanly.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Reset all agents
            for agent_id, agent in self.agents.items():
                try:
                    agent.reset()
                except:
                    pass
            
            # Clear all data structures
            self.agents.clear()
            self.teams.clear()
            self.agent_configs.clear()
            self.agent_status.clear()
            self.active_tasks.clear()
            self.conversations.clear()
            
            return True
        except Exception as e:
            logging.error(f"Error shutting down AutoGen adapter: {str(e)}")
            return False
