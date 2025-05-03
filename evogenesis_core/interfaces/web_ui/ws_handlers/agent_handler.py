"""
Agent Factory WebSocket Handler for EvoGenesis Web UI

This module connects the Agent Factory to the WebSocketManager for real-time updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

class AgentFactoryWebSocketHandler:
    """
    Handles WebSocket events for the Agent Factory.
    
    This class bridges the Agent Factory with the WebSocketManager to provide
    real-time updates about agents and teams to the web UI.
    """
    
    def __init__(self, agent_factory, ws_manager):
        """
        Initialize the Agent Factory WebSocket Handler.
        
        Args:
            agent_factory: The Agent Factory instance
            ws_manager: The WebSocket Manager instance
        """
        self.agent_factory = agent_factory
        self.ws_manager = ws_manager
        self.logger = logging.getLogger(__name__)
        
        # Track the last status to avoid sending duplicates
        self.last_status = None
        self.last_agents_hash = None
        self.last_teams_hash = None
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start the status update task
        asyncio.create_task(self._status_update_task())
    
    def _register_event_handlers(self):
        """Register handlers for agent-related WebSocket events."""
        # Register component-specific handlers
        self.ws_manager.register_component_handler("agents.action", self._handle_agent_action)
        self.ws_manager.register_component_handler("teams.action", self._handle_team_action)
    
    async def _handle_agent_action(self, message: Dict[str, Any]):
        """
        Handle agent action messages from the WebSocket.
        
        Args:
            message: The action message
        """
        if not isinstance(message, dict):
            return
        
        action = message.get("action")
        agent_id = message.get("agent_id")
        
        if not action or not agent_id:
            return
        
        try:
            if action == "start":
                success = self.agent_factory.start_agent(agent_id)
                if success:
                    await self._broadcast_agent_update(agent_id, "started")
            
            elif action == "stop":
                success = self.agent_factory.stop_agent(agent_id)
                if success:
                    await self._broadcast_agent_update(agent_id, "stopped")
            
            elif action == "pause":
                success = self.agent_factory.pause_agent(agent_id)
                if success:
                    await self._broadcast_agent_update(agent_id, "paused")
            
            elif action == "resume":
                success = self.agent_factory.resume_agent(agent_id)
                if success:
                    await self._broadcast_agent_update(agent_id, "resumed")
            
            elif action == "delete":
                success = self.agent_factory.delete_agent(agent_id)
                if success:
                    await self._broadcast_agent_update(agent_id, "deleted")
        
        except Exception as e:
            self.logger.error(f"Error handling agent action {action} for agent {agent_id}: {str(e)}")
            await self.ws_manager.broadcast_to_topic("agents.errors", {
                "error": "action_failed",
                "action": action,
                "agent_id": agent_id,
                "message": str(e)
            })
    
    async def _handle_team_action(self, message: Dict[str, Any]):
        """
        Handle team action messages from the WebSocket.
        
        Args:
            message: The action message
        """
        if not isinstance(message, dict):
            return
        
        action = message.get("action")
        team_id = message.get("team_id")
        
        if not action or not team_id:
            return
        
        try:
            if action == "start":
                success = self.agent_factory.start_team(team_id)
                if success:
                    await self._broadcast_team_update(team_id, "started")
            
            elif action == "stop":
                success = self.agent_factory.stop_team(team_id)
                if success:
                    await self._broadcast_team_update(team_id, "stopped")
            
            elif action == "delete":
                success = self.agent_factory.delete_team(team_id)
                if success:
                    await self._broadcast_team_update(team_id, "deleted")
            
            elif action == "add_agent":
                agent_id = message.get("agent_id")
                if agent_id:
                    success = self.agent_factory.add_agent_to_team(agent_id, team_id)
                    if success:
                        await self._broadcast_team_update(team_id, "agent_added", {"agent_id": agent_id})
            
            elif action == "remove_agent":
                agent_id = message.get("agent_id")
                if agent_id:
                    success = self.agent_factory.remove_agent_from_team(agent_id, team_id)
                    if success:
                        await self._broadcast_team_update(team_id, "agent_removed", {"agent_id": agent_id})
        
        except Exception as e:
            self.logger.error(f"Error handling team action {action} for team {team_id}: {str(e)}")
            await self.ws_manager.broadcast_to_topic("teams.errors", {
                "error": "action_failed",
                "action": action,
                "team_id": team_id,
                "message": str(e)
            })
    
    async def _broadcast_agent_update(self, agent_id, event_type, extra_data=None):
        """Broadcast an agent update event."""
        try:
            # Get the agent data
            agent = self.agent_factory.get_agent(agent_id)
            
            if not agent and event_type != "deleted":
                return
            
            # Create agent data dict
            agent_data = {
                "id": agent_id,
                "event": f"agent_{event_type}",
                "timestamp": time.time()
            }
            
            if agent and event_type != "deleted":
                agent_data.update({
                    "name": agent.name if hasattr(agent, "name") else "Unknown",
                    "type": agent.attributes.get("agent_type", "Unknown") if hasattr(agent, "attributes") else "Unknown",
                    "status": agent.status if hasattr(agent, "status") else "Unknown",
                    "capabilities": agent.capabilities if hasattr(agent, "capabilities") else [],
                    "team_id": agent.attributes.get("team_id") if hasattr(agent, "attributes") else None
                })
            
            # Add extra data if provided
            if extra_data:
                agent_data.update(extra_data)
            
            # Broadcast to agents topic
            await self.ws_manager.broadcast_to_topic("agents", agent_data)
            
            # Also broadcast to specific agent topic
            await self.ws_manager.broadcast_to_topic(f"agents.{agent_id}", agent_data)
            
            # Broadcast to agent status topic
            await self.ws_manager.broadcast_to_topic("agents.status", {
                "event": f"agent_{event_type}",
                "agent_id": agent_id,
                "status": agent.status if agent and hasattr(agent, "status") else "deleted",
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.error(f"Error broadcasting agent update for {agent_id}: {str(e)}")
    
    async def _broadcast_team_update(self, team_id, event_type, extra_data=None):
        """Broadcast a team update event."""
        try:
            # Get the team data
            team = self.agent_factory.get_team(team_id)
            
            if not team and event_type != "deleted":
                return
            
            # Create team data dict
            team_data = {
                "id": team_id,
                "event": f"team_{event_type}",
                "timestamp": time.time()
            }
            
            if team and event_type != "deleted":
                team_data.update({
                    "name": team.name if hasattr(team, "name") else "Unknown",
                    "goal": team.goals[0] if hasattr(team, "goals") and team.goals else "Unknown",
                    "agent_count": len(team.agents) if hasattr(team, "agents") else 0,
                    "status": team.status if hasattr(team, "status") else "Unknown"
                })
            
            # Add extra data if provided
            if extra_data:
                team_data.update(extra_data)
            
            # Broadcast to teams topic
            await self.ws_manager.broadcast_to_topic("teams", team_data)
            
            # Also broadcast to specific team topic
            await self.ws_manager.broadcast_to_topic(f"teams.{team_id}", team_data)
            
            # Broadcast to team status topic
            await self.ws_manager.broadcast_to_topic("teams.status", {
                "event": f"team_{event_type}",
                "team_id": team_id,
                "status": team.status if team and hasattr(team, "status") else "deleted",
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.error(f"Error broadcasting team update for {team_id}: {str(e)}")
    
    async def _status_update_task(self):
        """Background task to update agent factory status and broadcast it."""
        while True:
            try:
                # Get current status
                status = self.agent_factory.get_status()
                
                # Check if status has changed to avoid spamming
                if status != self.last_status:
                    # Broadcast to subscribers
                    await self.ws_manager.broadcast_to_topic("agents.factory.status", status)
                    self.last_status = status
                
                # Update agent and team lists if changed
                await self._update_agents_if_changed()
                await self._update_teams_if_changed()
                
                # Wait before checking again
                await asyncio.sleep(3)  # Updates for agents/teams
            except Exception as e:
                self.logger.error(f"Error in agent factory status update task: {str(e)}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _update_agents_if_changed(self):
        """Update and broadcast agent list if it has changed."""
        try:
            # Get current agents
            agents = self.agent_factory.agents
            
            # Generate a simple hash to check for changes
            agents_hash = hash(frozenset([
                (agent_id, 
                 getattr(agent, 'status', 'unknown'),
                 getattr(agent, 'updated_at', 0) if hasattr(agent, 'updated_at') else 0)
                for agent_id, agent in agents.items()
            ]))
            
            if agents_hash != self.last_agents_hash:
                # Format agent data for the frontend
                agent_list = []
                for agent_id, agent in agents.items():
                    agent_list.append({
                        "id": agent_id,
                        "name": agent.name if hasattr(agent, "name") else "Unknown",
                        "type": agent.attributes.get("agent_type", "Unknown") if hasattr(agent, "attributes") else "Unknown",
                        "status": agent.status if hasattr(agent, "status") else "Unknown",
                        "capabilities": agent.capabilities if hasattr(agent, "capabilities") else [],
                        "team_id": agent.attributes.get("team_id") if hasattr(agent, "attributes") else None
                    })
                
                # Broadcast agent list
                await self.ws_manager.broadcast_to_topic("agents.list", {
                    "event": "agents_updated",
                    "agents": agent_list,
                    "timestamp": time.time()
                })
                
                # Update the hash
                self.last_agents_hash = agents_hash
        except Exception as e:
            self.logger.error(f"Error updating agents: {str(e)}")
    
    async def _update_teams_if_changed(self):
        """Update and broadcast team list if it has changed."""
        try:
            # Get current teams
            teams = self.agent_factory.teams
            
            # Generate a simple hash to check for changes
            teams_hash = hash(frozenset([
                (team_id, 
                 getattr(team, 'status', 'unknown'),
                 len(team.agents) if hasattr(team, 'agents') else 0,
                 getattr(team, 'updated_at', 0) if hasattr(team, 'updated_at') else 0)
                for team_id, team in teams.items()
            ]))
            
            if teams_hash != self.last_teams_hash:
                # Format team data for the frontend
                team_list = []
                for team_id, team in teams.items():
                    team_list.append({
                        "id": team_id,
                        "name": team.name if hasattr(team, "name") else "Unknown",
                        "goal": team.goals[0] if hasattr(team, "goals") and team.goals else "Unknown",
                        "agent_count": len(team.agents) if hasattr(team, "agents") else 0,
                        "status": team.status if hasattr(team, "status") else "Unknown"
                    })
                
                # Broadcast team list
                await self.ws_manager.broadcast_to_topic("teams.list", {
                    "event": "teams_updated",
                    "teams": team_list,
                    "timestamp": time.time()
                })
                
                # Update the hash
                self.last_teams_hash = teams_hash
        except Exception as e:
            self.logger.error(f"Error updating teams: {str(e)}")

def connect_agent_factory(agent_factory, ws_manager):
    """
    Connect the Agent Factory to the WebSocket Manager.
    
    Args:
        agent_factory: The Agent Factory instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        The handler instance
    """
    handler = AgentFactoryWebSocketHandler(agent_factory, ws_manager)
    return handler
