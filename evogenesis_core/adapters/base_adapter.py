"""
Base Adapter Module - Defines the interface for external agent framework adapters.

This module provides the abstract base class that all framework adapters must implement
to integrate external agent frameworks with EvoGenesis.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable


class AgentExecutionAdapter(ABC):
    """
    Abstract base class for adapters to external agent frameworks.
    
    All adapters must implement these methods to ensure consistent 
    interaction between EvoGenesis and external frameworks.
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the external framework with configuration.
        
        Args:
            config: Configuration dictionary for the framework
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def create_agent(self, agent_spec: Dict[str, Any]) -> str:
        """
        Create an agent in the external framework.
        
        Args:
            agent_spec: Specification of the agent to create
            
        Returns:
            Agent ID in the external framework
        """
        pass
    
    @abstractmethod
    async def run_agent_task(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task using the specified agent.
        
        Args:
            agent_id: ID of the agent to use
            task: Task specification containing input, context, and requirements
            
        Returns:
            Task execution results
        """
        pass
    
    @abstractmethod
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the current status of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with agent status information
        """
        pass
    
    @abstractmethod
    async def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an agent in the external framework.
        
        Args:
            agent_id: ID of the agent to terminate
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def pause_agent(self, agent_id: str) -> bool:
        """
        Pause an agent's execution (if supported).
        
        Args:
            agent_id: ID of the agent to pause
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def resume_agent(self, agent_id: str) -> bool:
        """
        Resume a paused agent's execution.
        
        Args:
            agent_id: ID of the agent to resume
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an agent's configuration or state.
        
        Args:
            agent_id: ID of the agent to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def create_team(self, team_spec: Dict[str, Any]) -> str:
        """
        Create a team of agents if supported by the framework.
        
        Args:
            team_spec: Specification of the team to create
            
        Returns:
            Team ID in the external framework
        """
        pass
    
    @abstractmethod
    async def get_framework_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the framework.
        
        Returns:
            Dictionary describing the framework's capabilities
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Shut down the framework adapter cleanly.
        
        Returns:
            True if successful, False otherwise
        """
        pass
