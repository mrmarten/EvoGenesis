"""
Remote Kernel Client - Enables communication with distributed backup kernels.

This module provides client functionality for connecting to remote EvoGenesis kernels,
enabling distributed deployments with high availability.
"""

import logging
import requests
import time
from typing import Dict, Any, Optional

class RemoteKernelClient:
    """
    Client for communicating with remote EvoGenesis kernel instances.
    
    Supports heartbeat, status checks, state synchronization and failover
    coordination between primary and backup kernels.
    """
    
    def __init__(
        self, 
        address: str, 
        port: int, 
        api_key: Optional[str] = None,
        connection_timeout: int = 30
    ):
        """
        Initialize the remote kernel client.
        
        Args:
            address: Address of the remote kernel
            port: Port the remote kernel is listening on
            api_key: Optional API key for authentication
            connection_timeout: Connection timeout in seconds
        """
        self.address = address
        self.port = port
        self.api_key = api_key
        self.connection_timeout = connection_timeout
        self.base_url = f"http://{address}:{port}/api"
        self.logger = logging.getLogger(__name__)
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the remote kernel.
        
        Returns:
            Status information dictionary
        """
        try:
            response = self._make_request("GET", "/status")
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting remote kernel status: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def configure_as_backup(self, primary_id: str) -> bool:
        """
        Configure the remote kernel as a backup for this primary.
        
        Args:
            primary_id: ID of the primary kernel
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {"primary_id": primary_id, "role": "backup"}
            response = self._make_request("POST", "/configure", json=data)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Error configuring remote kernel as backup: {str(e)}")
            return False
    
    def receive_state_update(self, state: Dict[str, Any]) -> bool:
        """
        Send state update to the remote kernel.
        
        Args:
            state: State data to synchronize
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self._make_request("POST", "/state", json=state)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Error sending state update: {str(e)}")
            return False
            
    def shutdown(self, reason: str = "api_request") -> bool:
        """
        Request remote kernel to shut down.
        
        Args:
            reason: Reason for shutdown
            
        Returns:
            True if request was successful, False otherwise
        """
        try:
            data = {"reason": reason}
            response = self._make_request("POST", "/shutdown", json=data)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Error requesting remote kernel shutdown: {str(e)}")
            return False
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make an HTTP request to the remote kernel API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            Exception: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        headers = kwargs.pop("headers", {})
        
        # Add authentication
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        # Add timeout
        timeout = kwargs.pop("timeout", self.connection_timeout)
        
        # Make request with retries
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.request(
                    method, 
                    url, 
                    headers=headers, 
                    timeout=timeout,
                    **kwargs
                )
                
                if response.status_code in (429, 503):  # Rate limit or service unavailable
                    retry_delay = min(retry_delay * 2, 10)  # Exponential backoff
                    time.sleep(retry_delay)
                    continue
                    
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    retry_delay = min(retry_delay * 2, 10)
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"Request failed after {max_retries} attempts: {str(e)}")
        
        # This should be unreachable
        raise Exception("Request failed with unknown error")