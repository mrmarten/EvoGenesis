"""
Perception-Action Tooling Layer for EvoGenesis

This module implements the Perception-Action Tooling layer that enables EvoGenesis
to dynamically manufacture the right control method for remote machines - API, 
command shell, GUI automation, or lights-out KVM.

Key components:
1. Remote Target Discovery - Inspects targets to determine available control methods
2. Decision Engine - Chooses the safest/cheapest adapter based on weighted scoring
3. Tool Generator - Auto-generates remote control tools with appropriate adapters
4. Audit Logging - Records all remote control operations for replay and audit
"""

import os
import re
import json
import time
import logging
import asyncio
from enum import Enum, auto
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import core EvoGenesis components
from evogenesis_core.modules.tooling_system import ToolScope, ToolStatus, SandboxType, Tool
import nmap
import socket
import requests
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.desktopvirtualization import DesktopVirtualizationMgmtClient


class RemoteAdapterType(str, Enum):
    """Types of adapters for remote machine control."""
    GRAPH_CLOUDPC = "graph_cloudpc"    # Microsoft Graph API for Cloud PCs
    DEV_BOX = "dev_box"                # Dev Box REST API
    AVD_REST = "avd_rest"              # Azure Virtual Desktop REST API
    ARC_COMMAND = "arc_command"        # Azure Arc Run Command
    RDP = "rdp"                        # Remote Desktop Protocol
    VNC = "vnc"                        # Virtual Network Computing
    SSH = "ssh"                        # Secure Shell
    AMT_KVM = "amt_kvm"                # Intel AMT KVM
    VISION_FALLBACK = "vision"         # Vision-based fallback (GPT-4o/Microsoft computer-use)


class RemoteTargetInfo:
    """Information about a remote machine target."""
    
    def __init__(self, 
                host_id: str, 
                hostname: str, 
                ip_address: Optional[str] = None,
                os_type: Optional[str] = None,
                available_adapters: Optional[List[RemoteAdapterType]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize remote target information.
        
        Args:
            host_id: Unique identifier for the host
            hostname: Human-readable hostname
            ip_address: IP address if available
            os_type: Operating system type if known
            available_adapters: List of available adapter types
            metadata: Additional metadata about the target
        """
        self.host_id = host_id
        self.hostname = hostname
        self.ip_address = ip_address
        self.os_type = os_type
        self.available_adapters = available_adapters or []
        self.metadata = metadata or {}
        self.last_discovery = datetime.now()
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "host_id": self.host_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "os_type": self.os_type,
            "available_adapters": [adapter.value for adapter in self.available_adapters],
            "metadata": self.metadata,
            "last_discovery": self.last_discovery.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RemoteTargetInfo':
        """Create from dictionary representation."""
        target = cls(
            host_id=data.get("host_id", ""),
            hostname=data.get("hostname", ""),
            ip_address=data.get("ip_address"),
            os_type=data.get("os_type"),
            available_adapters=[RemoteAdapterType(a) for a in data.get("available_adapters", [])],
            metadata=data.get("metadata", {})
        )
        
        if "last_discovery" in data:
            target.last_discovery = datetime.fromisoformat(data["last_discovery"])
        
        return target


class RemoteDiscoveryService:
    """
    Service that discovers capabilities of remote targets.
    
    Responsible for:
    - Probing remote machines for available APIs
    - Port scanning for remote control protocols
    - Checking for cloud-specific capabilities
    - Maintaining a registry of machine capabilities
    """
    
    def __init__(self, kernel):
        """
        Initialize the remote discovery service.
        
        Args:
            kernel: Reference to the EvoGenesis kernel
        """
        self.kernel = kernel
        self.config = kernel.config.get("remote_discovery", {})
        self.target_registry = {}  # host_id -> RemoteTargetInfo
        
        # Initialize network scanning and discovery tools
        self.scanner_initialized = False
        
        # Check if scanning is explicitly disabled in config
        if self.config.get("disable_scanning", False):
            logging.info("Network scanning explicitly disabled in configuration")
            return
            
        try:
            # Import required libraries for network scanning
            
            # Initialize scanning components
            self.nmap_scanner = nmap.PortScanner()
            self.azure_credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
            
            # Test connectivity to ensure scanner works
            socket.gethostbyname("localhost")  # Basic connectivity test
            
            # Mark scanner as initialized
            self.scanner_initialized = True
            logging.info("Network scanning tools initialized successfully")
        except ImportError as e:
            logging.warning(f"Network scanning tools not available: {str(e)}")
            logging.info("Remote discovery will rely on API-based detection only")
        except Exception as e:
            logging.error(f"Error initializing network scanning tools: {str(e)}")
            logging.info("Remote discovery will operate in limited mode")
    
    async def discover_target(self, host_id: str, hostname: str, 
                           ip_address: Optional[str] = None) -> RemoteTargetInfo:
        """
        Discover capabilities of a remote target.
        
        Args:
            host_id: Unique identifier for the host
            hostname: Hostname or FQDN
            ip_address: IP address if available
            
        Returns:
            RemoteTargetInfo with discovered capabilities
        """
        # Check if we already have this target
        if host_id in self.target_registry:
            # If recent enough, return cached info
            target = self.target_registry[host_id]
            if (datetime.now() - target.last_discovery).total_seconds() < self.config.get("cache_ttl", 3600):
                return target
        
        # Create a new target info object
        target = RemoteTargetInfo(
            host_id=host_id,
            hostname=hostname,
            ip_address=ip_address
        )
        
        # Discover available adapters
        available_adapters = []
        
        # 1. Check for cloud APIs
        cloud_adapters = await self._check_cloud_apis(target)
        available_adapters.extend(cloud_adapters)
        
        # 2. Port scan if allowed and needed
        if not available_adapters and self.config.get("enable_port_scan", False):
            protocol_adapters = await self._scan_protocols(target)
            available_adapters.extend(protocol_adapters)
        
        # 3. Check for out-of-band management
        oob_adapters = await self._check_oob_management(target)
        available_adapters.extend(oob_adapters)
        
        # 4. Always add vision fallback as last resort
        if RemoteAdapterType.VISION_FALLBACK not in available_adapters:
            available_adapters.append(RemoteAdapterType.VISION_FALLBACK)
        
        # Update target with discovered adapters
        target.available_adapters = available_adapters
        target.last_discovery = datetime.now()
        
        # Store in registry
        self.target_registry[host_id] = target
        
        return target
    
    async def _check_cloud_apis(self, target: RemoteTargetInfo) -> List[RemoteAdapterType]:
        """Check for cloud-specific APIs."""
        adapters = []
        
        # Check for Microsoft Graph API for Cloud PCs
        if await self._probe_graph_cloudpc(target):
            adapters.append(RemoteAdapterType.GRAPH_CLOUDPC)
        
        # Check for Dev Box API
        if await self._probe_dev_box(target):
            adapters.append(RemoteAdapterType.DEV_BOX)
        
        # Check for Azure Virtual Desktop
        if await self._probe_avd(target):
            adapters.append(RemoteAdapterType.AVD_REST)
        
        # Check for Azure Arc
        if await self._probe_azure_arc(target):
            adapters.append(RemoteAdapterType.ARC_COMMAND)
        
        return adapters
    
    async def _scan_protocols(self, target: RemoteTargetInfo) -> List[RemoteAdapterType]:
        """Scan for remote control protocols."""
        adapters = []
        
        # Check if scanner is properly initialized
        if not self.scanner_initialized:
            logging.info(f"Network scanning tools not initialized, skipping protocol scan for {target.hostname}")
            return adapters
        # Ensure we have an IP to scan
        if not target.ip_address:
            try:
                # Try to resolve hostname to IP
                target.ip_address = socket.gethostbyname(target.hostname)
                logging.info(f"Resolved {target.hostname} to {target.ip_address}")
            except socket.gaierror:
                logging.warning(f"Could not resolve hostname {target.hostname} to IP address")
                return adapters
        
        try:
            # Scan common remote control protocol ports
            ports_to_scan = "22,3389,5900-5910"  # SSH, RDP, VNC range
            logging.info(f"Scanning {target.ip_address} for protocols on ports {ports_to_scan}")
            
            # Use nmap for comprehensive port scanning with service detection
            self.nmap_scanner.scan(target.ip_address, ports_to_scan, arguments='-sV --open')
            
            # Process scan results if the host was successfully scanned
            if target.ip_address in self.nmap_scanner.all_hosts():
                for proto in self.nmap_scanner[target.ip_address].all_protocols():
                    ports = sorted(self.nmap_scanner[target.ip_address][proto].keys())
                    for port in ports:
                        service = self.nmap_scanner[target.ip_address][proto][port]['name']
                        logging.debug(f"Found service {service} on port {port}")
                        
                        # Add appropriate adapter based on detected service
                        if port == 22 and service in ['ssh', 'openssh']:
                            adapters.append(RemoteAdapterType.SSH)
                        elif port == 3389 and service in ['ms-wbt-server', 'rdp']:
                            adapters.append(RemoteAdapterType.RDP)
                        elif 5900 <= port <= 5910 and service in ['vnc', 'realvnc', 'tightvnc']:
                            adapters.append(RemoteAdapterType.VNC)
            
            logging.info(f"Protocol scan complete for {target.hostname}, found: {adapters}")
        except Exception as e:
            logging.error(f"Error during protocol scan for {target.hostname}: {str(e)}")
        
        return adapters
    async def _check_oob_management(self, target: RemoteTargetInfo) -> List[RemoteAdapterType]:
        """Check for out-of-band management capabilities."""
        adapters = []
        
        # Check for Intel AMT/IPMI
        if await self._probe_amt(target):
            adapters.append(RemoteAdapterType.AMT_KVM)
        
        return adapters
    async def _probe_graph_cloudpc(self, target: RemoteTargetInfo) -> bool:
        """
        Probe for Microsoft Graph API for Cloud PCs.
        
        Attempts to query the Microsoft Graph API to determine if the target
        is a Cloud PC that can be managed via Graph API.
        """
        try:
            if not hasattr(self, 'azure_credential'):
                logging.debug(f"Azure credential not available, skipping Graph API probe for {target.hostname}")
                return False
                
            # Check if we have the target in Azure
            # Use Microsoft Graph API to query for Cloud PCs
            graph_endpoint = "https://graph.microsoft.com/v1.0/deviceManagement/virtualEndpoint/cloudPCs"
            headers = {
                "Authorization": f"Bearer {await self._get_graph_token()}",
                "Content-Type": "application/json"
            }
            
            # Query for Cloud PCs matching this hostname
            params = {
                "$filter": f"displayName eq '{target.hostname}' or managedDeviceName eq '{target.hostname}'"
            }
            
            async with self.kernel.get_http_client() as client:
                response = await client.get(
                    graph_endpoint, 
                    headers=headers, 
                    params=params,
                    timeout=self.config.get("api_timeout", 10)
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # If we found any matching Cloud PCs
                    if data.get("value") and len(data["value"]) > 0:
                        logging.info(f"Found Cloud PC for {target.hostname} via Graph API")
                        
                        # Store Cloud PC ID in metadata for future use
                        cloud_pc = data["value"][0]
                        target.metadata["cloud_pc_id"] = cloud_pc.get("id")
                        target.metadata["cloud_pc_status"] = cloud_pc.get("status")
                        
                        return True
                
                logging.debug(f"No Cloud PC found for {target.hostname} via Graph API")
                return False
                
        except Exception as e:
            logging.warning(f"Error probing Graph API for {target.hostname}: {str(e)}")
            return False
    
    async def _probe_dev_box(self, target: RemoteTargetInfo) -> bool:
        """
        Probe for Dev Box API.
        
        Checks if the target is a Dev Box that can be managed via the Dev Box API.
        """
        try:
            if not hasattr(self, 'azure_credential'):
                logging.debug(f"Azure credential not available, skipping Dev Box probe for {target.hostname}")
                return False
            
            # Dev Box API endpoint structure
            subscription_ids = self.config.get("azure_subscription_ids", [])
            if not subscription_ids:
                logging.debug("No Azure subscription IDs configured for Dev Box probing")
                return False
            
            # Check each subscription
            for subscription_id in subscription_ids:
                dev_box_endpoint = f"https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.DevCenter/devcenters?api-version=2023-04-01"
                
                headers = {
                    "Authorization": f"Bearer {await self._get_azure_token('https://management.azure.com/')}",
                    "Content-Type": "application/json"
                }
                
                async with self.kernel.get_http_client() as client:
                    # First get available Dev Centers
                    response = await client.get(
                        dev_box_endpoint, 
                        headers=headers,
                        timeout=self.config.get("api_timeout", 10)
                    )
                    
                    if response.status_code != 200:
                        continue
                    
                    dev_centers = response.json().get("value", [])
                    
                    # For each Dev Center, check Dev Boxes
                    for center in dev_centers:
                        center_name = center.get("name")
                        resource_group = center.get("id", "").split("/resourceGroups/")[1].split("/")[0]
                        
                        # Query for Dev Boxes in this center
                        boxes_endpoint = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.DevCenter/projects?api-version=2023-04-01"
                        
                        projects_response = await client.get(
                            boxes_endpoint, 
                            headers=headers,
                            timeout=self.config.get("api_timeout", 10)
                        )
                        
                        if projects_response.status_code != 200:
                            continue
                        
                        projects = projects_response.json().get("value", [])
                        
                        # Check each project for Dev Boxes
                        for project in projects:
                            project_name = project.get("name")
                            
                            # Query for Dev Boxes in this project
                            dev_boxes_endpoint = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.DevCenter/projects/{project_name}/devboxes?api-version=2023-04-01"
                            
                            boxes_response = await client.get(
                                dev_boxes_endpoint, 
                                headers=headers,
                                timeout=self.config.get("api_timeout", 10)
                            )
                            
                            if boxes_response.status_code != 200:
                                continue
                            
                            boxes = boxes_response.json().get("value", [])
                            
                            # Check if any box matches our target
                            for box in boxes:
                                box_name = box.get("name")
                                box_hostname = box.get("properties", {}).get("uniqueName")
                                
                                if box_hostname == target.hostname or box_name == target.hostname:
                                    logging.info(f"Found Dev Box for {target.hostname}")
                                    
                                    # Store Dev Box details in metadata
                                    target.metadata["dev_box_id"] = box.get("id")
                                    target.metadata["dev_box_resource_group"] = resource_group
                                    target.metadata["dev_box_project"] = project_name
                                    target.metadata["dev_box_subscription"] = subscription_id
                                    
                                    return True
            
            logging.debug(f"No Dev Box found for {target.hostname}")
            return False
                
        except Exception as e:
            logging.warning(f"Error probing Dev Box API for {target.hostname}: {str(e)}")
            return False
    
    async def _probe_avd(self, target: RemoteTargetInfo) -> bool:
        """
        Probe for Azure Virtual Desktop.
        
        Checks if the target is an Azure Virtual Desktop session host.
        """
        try:
            if not hasattr(self, 'azure_credential'):
                logging.debug(f"Azure credential not available, skipping AVD probe for {target.hostname}")
                return False
            
            subscription_ids = self.config.get("azure_subscription_ids", [])
            if not subscription_ids:
                logging.debug("No Azure subscription IDs configured for AVD probing")
                return False
                
            # Check each subscription for AVD resources
            for subscription_id in subscription_ids:
                try:
                    # Initialize AVD client
                    avd_client = DesktopVirtualizationMgmtClient(
                        credential=self.azure_credential,
                        subscription_id=subscription_id
                    )
                    
                    # Get host pools
                    host_pools = list(avd_client.host_pools.list_by_subscription())
                    
                    for pool in host_pools:
                        # Get resource group from the pool ID
                        resource_group = pool.id.split("/resourceGroups/")[1].split("/")[0]
                        
                        # Get session hosts in this pool
                        session_hosts = list(avd_client.session_hosts.list_by_host_pool(
                            resource_group_name=resource_group,
                            host_pool_name=pool.name
                        ))
                        
                        # Check if any session host matches our target
                        for host in session_hosts:
                            # Extract hostname from FQDN
                            host_name = host.name.split("/")[0]
                            host_fqdn = host.session_host_name
                            
                            if host_name == target.hostname or host_fqdn == target.hostname:
                                logging.info(f"Found AVD session host for {target.hostname}")
                                
                                # Store AVD details in metadata
                                target.metadata["avd_host_pool"] = pool.name
                                target.metadata["avd_resource_group"] = resource_group
                                target.metadata["avd_subscription"] = subscription_id
                                target.metadata["avd_session_host"] = host.name
                                
                                return True
                
                except Exception as sub_e:
                    logging.warning(f"Error checking AVD in subscription {subscription_id}: {str(sub_e)}")
                    continue
            
            logging.debug(f"No AVD session host found for {target.hostname}")
            return False
                
        except Exception as e:
            logging.warning(f"Error probing AVD API for {target.hostname}: {str(e)}")
            return False
    
    async def _probe_azure_arc(self, target: RemoteTargetInfo) -> bool:
        """
        Probe for Azure Arc.
        
        Checks if the target is an Azure Arc-enabled server.
        """
        try:
            if not hasattr(self, 'azure_credential'):
                logging.debug(f"Azure credential not available, skipping Arc probe for {target.hostname}")
                return False
            
            subscription_ids = self.config.get("azure_subscription_ids", [])
            if not subscription_ids:
                logging.debug("No Azure subscription IDs configured for Arc probing")
                return False
            
            # Check each subscription for Arc-enabled machines
            for subscription_id in subscription_ids:
                try:
                    # Initialize Compute client for Arc resources
                    compute_client = ComputeManagementClient(
                        credential=self.azure_credential,
                        subscription_id=subscription_id
                    )
                    
                    # List all Arc machines in the subscription
                    arc_machines = compute_client.hybrid_compute_machines.list_by_subscription()
                    
                    # Check if any matches our target
                    for machine in arc_machines:
                        machine_name = machine.name
                        machine_fqdn = machine.os_profile.computer_name if machine.os_profile else None
                        
                        if machine_name == target.hostname or machine_fqdn == target.hostname:
                            logging.info(f"Found Arc-enabled server for {target.hostname}")
                            
                            # Store Arc details in metadata
                            resource_group = machine.id.split("/resourceGroups/")[1].split("/")[0]
                            target.metadata["arc_machine_id"] = machine.id
                            target.metadata["arc_resource_group"] = resource_group
                            target.metadata["arc_subscription"] = subscription_id
                            target.metadata["arc_os_type"] = machine.os_type
                            
                            # Update target OS type if available
                            if not target.os_type and machine.os_type:
                                target.os_type = machine.os_type
                            
                            return True
                
                except Exception as sub_e:
                    logging.warning(f"Error checking Arc in subscription {subscription_id}: {str(sub_e)}")
                    continue
            
            logging.debug(f"No Arc-enabled server found for {target.hostname}")
            return False
                
        except Exception as e:
            logging.warning(f"Error probing Azure Arc for {target.hostname}: {str(e)}")
            return False
    
    async def _scan_port(self, target: RemoteTargetInfo, port: int) -> bool:
        """
        Scan for open port on target.
        
        Args:
            target: The target to scan
            port: The port number to check
            
        Returns:
            True if port is open, False otherwise
        """
        if not target.ip_address:
            try:
                # Try to resolve hostname to IP
                target.ip_address = socket.gethostbyname(target.hostname)
                logging.info(f"Resolved {target.hostname} to {target.ip_address}")
            except socket.gaierror:
                logging.warning(f"Could not resolve hostname {target.hostname} to IP address")
                return False
        
        try:
            # Use non-blocking socket with timeout
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.config.get("port_scan_timeout", 1.0))
            
            # Connect to the port
            result = s.connect_ex((target.ip_address, port))
            s.close()
            
            # If result is 0, port is open
            is_open = (result == 0)
            
            if is_open:
                logging.info(f"Port {port} is open on {target.hostname} ({target.ip_address})")
            else:
                logging.debug(f"Port {port} is closed on {target.hostname} ({target.ip_address})")
                
            return is_open
            
        except Exception as e:
            logging.warning(f"Error scanning port {port} on {target.hostname}: {str(e)}")
            return False
    
    async def _probe_amt(self, target: RemoteTargetInfo) -> bool:
        """
        Probe for Intel AMT.
        
        Checks if the target has Intel AMT/IPMI capabilities for out-of-band management.
        """
        try:
            if not target.ip_address:
                try:
                    # Try to resolve hostname to IP
                    target.ip_address = socket.gethostbyname(target.hostname)
                    logging.info(f"Resolved {target.hostname} to {target.ip_address}")
                except socket.gaierror:
                    logging.warning(f"Could not resolve hostname {target.hostname} to IP address")
                    return False
            
            # Check common AMT ports
            amt_ports = [16992, 16993, 623, 664]  # 16992/16993 for AMT web, 623/664 for IPMI
            
            for port in amt_ports:
                is_open = await self._scan_port(target, port)
                if is_open:
                    # For AMT web ports, try to verify it's actually AMT
                    if port in [16992, 16993]:
                        protocol = "http" if port == 16992 else "https"
                        url = f"{protocol}://{target.ip_address}:{port}/amt-tls"
                        
                        try:
                            async with self.kernel.get_http_client() as client:
                                response = await client.get(
                                    url, 
                                    timeout=self.config.get("api_timeout", 5),
                                    verify=False  # AMT often uses self-signed certs
                                )
                                
                                # If we get specific AMT headers or content
                                if response.status_code == 200 or "Intel(R) AMT" in response.text:
                                    logging.info(f"Verified Intel AMT on {target.hostname} port {port}")
                                    target.metadata["amt_port"] = port
                                    target.metadata["amt_protocol"] = protocol
                                    return True
                        except Exception as req_err:
                            logging.debug(f"Error verifying AMT on {target.hostname}: {str(req_err)}")
                    
                    # For IPMI port, just the open port is a good indicator
                    if port in [623, 664]:
                        logging.info(f"Detected possible IPMI on {target.hostname} port {port}")
                        target.metadata["ipmi_port"] = port
                        return True
            
            logging.debug(f"No Intel AMT/IPMI capabilities detected for {target.hostname}")
            return False
                
        except Exception as e:
            logging.warning(f"Error probing for Intel AMT on {target.hostname}: {str(e)}")
            return False
            
    async def _get_graph_token(self) -> str:
        """Helper method to get Microsoft Graph API access token."""
        try:
            token = await self.azure_credential.get_token("https://graph.microsoft.com/.default")
            return token.token
        except Exception as e:
            logging.error(f"Error getting Graph API token: {str(e)}")
            raise
            
    async def _get_azure_token(self, scope: str) -> str:
        """Helper method to get Azure API access token for specific scope."""
        try:
            token = await self.azure_credential.get_token(scope)
            return token.token
        except Exception as e:
            logging.error(f"Error getting Azure token for scope {scope}: {str(e)}")
            raise


class RemoteAdapterDecisionEngine:
    """
    Decision engine that selects the best adapter for a remote control task.
    
    Uses a weighted scoring system based on:
    - Security (higher is better)
    - Cost (lower is better)
    - Latency (lower is better)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the decision engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default weights
        self.weights = self.config.get("weights", {
            "security": 0.5,
            "cost": 0.3,
            "latency": 0.2
        })
        
        # Default adapter scores
        self.adapter_scores = self.config.get("adapter_scores", {
            RemoteAdapterType.GRAPH_CLOUDPC: {"security": 0.9, "cost": 0.2, "latency": 0.5},
            RemoteAdapterType.DEV_BOX: {"security": 0.9, "cost": 0.2, "latency": 0.5},
            RemoteAdapterType.AVD_REST: {"security": 0.8, "cost": 0.3, "latency": 0.6},
            RemoteAdapterType.ARC_COMMAND: {"security": 0.8, "cost": 0.3, "latency": 0.7},
            RemoteAdapterType.RDP: {"security": 0.7, "cost": 0.4, "latency": 0.3},
            RemoteAdapterType.VNC: {"security": 0.6, "cost": 0.4, "latency": 0.3},
            RemoteAdapterType.SSH: {"security": 0.8, "cost": 0.1, "latency": 0.2},
            RemoteAdapterType.AMT_KVM: {"security": 0.7, "cost": 0.5, "latency": 0.7},
            RemoteAdapterType.VISION_FALLBACK: {"security": 0.6, "cost": 0.8, "latency": 0.8}
        })
    
    def select_adapter(self, available_adapters: List[RemoteAdapterType], 
                      operation_type: str = "general") -> RemoteAdapterType:
        """
        Select the best adapter for an operation.
        
        Args:
            available_adapters: List of available adapters
            operation_type: Type of operation to perform 
                        (e.g., 'power', 'script', 'gui')
            
        Returns:
            The selected adapter type
        """
        if not available_adapters:
            raise ValueError("No adapters available")
        
        # Adjust weights based on operation type
        weights = self.weights.copy()
        if operation_type == "power":
            # For power operations, security is most important
            weights["security"] = 0.7
            weights["cost"] = 0.2
            weights["latency"] = 0.1
        elif operation_type == "script":
            # For script execution, balance security and latency
            weights["security"] = 0.6
            weights["cost"] = 0.2
            weights["latency"] = 0.2
        elif operation_type == "gui":
            # For GUI operations, latency is more important
            weights["security"] = 0.4
            weights["cost"] = 0.2
            weights["latency"] = 0.4
        
        # Calculate score for each adapter
        scores = {}
        for adapter in available_adapters:
            if adapter not in self.adapter_scores:
                scores[adapter] = 0.0
                continue
                
            adapter_score = self.adapter_scores[adapter]
            
            # Calculate weighted score
            # Security is better when higher, cost and latency are better when lower
            score = (
                adapter_score["security"] * weights["security"] +
                (1 - adapter_score["cost"]) * weights["cost"] +
                (1 - adapter_score["latency"]) * weights["latency"]
            )
            
            scores[adapter] = score
        
        # Select the adapter with the highest score
        selected_adapter = max(scores.items(), key=lambda x: x[1])[0]
        
        return selected_adapter


class RemoteControlModule:
    """
    Main module for remote machine control in EvoGenesis.
    
    Integrates with the existing ToolingSystem to provide:
    - Remote target discovery
    - Adapter selection
    - Tool generation
    - Secure execution
    - Audit logging
    """
    
    def __init__(self, kernel):
        """
        Initialize the remote control module.
        
        Args:
            kernel: Reference to the EvoGenesis kernel
        """
        self.kernel = kernel
        self.tooling_system = kernel.get_module("tooling_system")
        self.llm_orchestrator = kernel.llm_orchestrator
        self.config = kernel.config.get("remote_control", {})
        
        # Initialize discovery service and decision engine
        self.discovery_service = RemoteDiscoveryService(kernel)
        self.decision_engine = RemoteAdapterDecisionEngine(
            self.config.get("decision_engine", {})
        )
        
        # Remote control audit log
        self.audit_log = []
    
    async def discover_target(self, host_id: str, hostname: str, 
                           ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Discover capabilities of a remote target.
        
        Args:
            host_id: Unique identifier for the host
            hostname: Hostname or FQDN
            ip_address: IP address if available
            
        Returns:
            Dictionary with target information and available adapters
        """
        target = await self.discovery_service.discover_target(
            host_id=host_id,
            hostname=hostname,
            ip_address=ip_address
        )
        
        return target.as_dict()
    
    async def generate_tool(self, 
                         host_id: str,
                         hostname: str,
                         description: str,
                         operation_type: str = "general",
                         parameters: Dict[str, Dict[str, Any]] = None,
                         returns: Dict[str, Any] = None,
                         ip_address: Optional[str] = None) -> str:
        """
        Generate a tool for remote machine control.
        
        Args:
            host_id: Unique identifier for the host
            hostname: Hostname or FQDN
            description: Description of what the tool should do
            operation_type: Type of operation ('power', 'script', 'gui')
            parameters: Dictionary of parameter specifications
            returns: Specification of what the tool should return
            ip_address: IP address if available
            
        Returns:
            ID of the generated tool
        """
        # Discover target capabilities if not already done
        target = await self.discovery_service.discover_target(
            host_id=host_id,
            hostname=hostname,
            ip_address=ip_address
        )
        
        if not target.available_adapters:
            raise ValueError(f"No adapters available for {hostname}")
        
        # Select best adapter
        adapter_type = self.decision_engine.select_adapter(
            available_adapters=target.available_adapters,
            operation_type=operation_type
        )
        
        # Generate the prompt template
        adapter_templates = {
            RemoteAdapterType.GRAPH_CLOUDPC: "Generate a Python function that uses Microsoft Graph API to {description} on Cloud PC {hostname}.",
            RemoteAdapterType.DEV_BOX: "Generate a Python function that uses Dev Box REST API to {description} on Dev Box {hostname}.",
            RemoteAdapterType.AVD_REST: "Generate a Python function that uses Azure Virtual Desktop REST API to {description} on AVD host {hostname}.",
            RemoteAdapterType.ARC_COMMAND: "Generate a Python function that uses Azure Arc Run Command to {description} on Arc-enabled server {hostname}.",
            RemoteAdapterType.RDP: "Generate a Python function using Playwright to {description} over RDP on host {hostname}.",
            RemoteAdapterType.VNC: "Generate a Python function using Playwright to {description} over VNC on host {hostname}.",
            RemoteAdapterType.SSH: "Generate a Python function using paramiko to {description} over SSH on host {hostname}.",
            RemoteAdapterType.AMT_KVM: "Generate a Python function using Intel AMT SDK to {description} on host {hostname} using out-of-band management.",
            RemoteAdapterType.VISION_FALLBACK: "Generate a Python function using Playwright and Vision API to {description} on host {hostname} by analyzing screenshots and performing actions."
        }
        
        template = adapter_templates.get(adapter_type, 
            "Generate a Python function that {description} on host {hostname}.")
        
        prompt = template.format(
            description=description,
            hostname=hostname
        )
        
        # Add parameter and return information
        if parameters:
            prompt += f"\n\nThe function should accept these parameters: {json.dumps(parameters, indent=2)}"
        
        if returns:
            prompt += f"\n\nThe function should return: {json.dumps(returns, indent=2)}"
        
        # Add adapter-specific details
        prompt += self._get_adapter_details(adapter_type, target)
        
        # Add security and audit requirements
        prompt += """
        
The code must follow these security requirements:
1. Use secure authentication methods with credentials from environment variables
2. Include proper error handling and logging
3. Record all actions for audit (timestamps, commands executed, results)
4. Implement timeouts for all remote operations
5. Sanitize all inputs to prevent injection attacks
6. Return a complete audit record with the results
"""
        
        # Generate the code using LLM
        response = await self.llm_orchestrator.execute_prompt_async(
            task_type="code_generation",
            prompt_template="direct",
            params={"prompt": prompt},
            model_selection={
                "model_name": "gpt-4o",  # Use a strong model for security-critical code
                "provider": "openai"
            },
            max_tokens=2500
        )
        
        # Extract code from the response
        code = response.get("result", "")
        
        # If the result is a string, try to extract a code block
        if isinstance(code, str):
            # Try to extract code block if present
            code_match = re.search(r'```python\s*(.*?)\s*```', code, re.DOTALL)
            if code_match:
                code = code_match.group(1)
        
        # Generate a name for the tool based on the description and host
        tool_name = self._generate_tool_name(
            description=description,
            hostname=hostname,
            adapter_type=adapter_type
        )
        
        # Create the tool
        tool = Tool(
            name=tool_name,
            description=f"{description} on {hostname}",
            function=code,
            scope=ToolScope.REMOTE,  # Use REMOTE scope for all remote control tools
            parameters=parameters or {},
            returns=returns or {"type": "object"},
            metadata={
                "host_id": host_id,
                "hostname": hostname,
                "adapter_type": adapter_type.value,
                "operation_type": operation_type,
                "target_info": target.as_dict(),
                "generation_prompt": prompt,
                "has_audit_log": True,
                "requires_vision": adapter_type == RemoteAdapterType.VISION_FALLBACK
            },
            sandbox_type=SandboxType.DOCKER,  # Run in Docker for security
            auto_generated=True
        )
        
        # Register the tool with the tooling system
        tool_id = self.tooling_system.register_tool(tool)
        
        # Log the tool generation
        logging.info(f"Generated remote control tool: {tool_name} (ID: {tool_id}) for {hostname} using {adapter_type}")
        
        return tool_id
    
    def _generate_tool_name(self, 
                          description: str, 
                          hostname: str,
                          adapter_type: RemoteAdapterType) -> str:
        """
        Generate a name for the remote control tool.
        
        Args:
            description: Description of what the tool does
            hostname: Hostname or FQDN
            adapter_type: Type of adapter
            
        Returns:
            Tool name
        """
        # Extract key action from description
        action_words = ["start", "stop", "restart", "reboot", "shutdown", "connect", 
                      "execute", "run", "get", "monitor", "update", "install", "remove"]
        
        action = next((word for word in action_words if word in description.lower()), "control")
        
        # Clean hostname for use in function name
        host_part = re.sub(r'[^a-zA-Z0-9]', '_', hostname.split('.')[0]).lower()
        
        # Add adapter suffix
        adapter_suffix = adapter_type.value.lower()
        
        # Combine parts
        return f"{action}_{host_part}_{adapter_suffix}"
    
    def _get_adapter_details(self, adapter_type: RemoteAdapterType, target: RemoteTargetInfo) -> str:
        """
        Get adapter-specific details to include in the prompt.
        
        Args:
            adapter_type: Type of adapter
            target: Target information
            
        Returns:
            String with adapter details
        """
        details = "\n\nImplementation details:"
        
        if adapter_type == RemoteAdapterType.GRAPH_CLOUDPC:
            details += """
- Use Microsoft Graph API to connect to the Cloud PC
- Authenticate using OAuth and Microsoft Identity client library
- Include proper error handling for Graph API responses
- Documentation: https://learn.microsoft.com/en-us/graph/api/resources/cloudpc"""
            
        elif adapter_type == RemoteAdapterType.DEV_BOX:
            details += """
- Use Dev Box REST API to manage the Dev Box
- Authenticate using OAuth and Microsoft Identity client library
- Handle common API errors and retry patterns
- Documentation: https://learn.microsoft.com/en-us/rest/api/devbox/"""
            
        elif adapter_type == RemoteAdapterType.AVD_REST:
            details += """
- Use Azure Virtual Desktop REST API
- Authenticate using Azure Identity SDK
- Implement appropriate session management
- Documentation: https://learn.microsoft.com/en-us/rest/api/avd/"""
            
        elif adapter_type == RemoteAdapterType.ARC_COMMAND:
            details += """
- Use Azure Arc Run Command feature
- Authenticate using Azure Identity SDK
- Support both Windows and Linux command execution
- Documentation: https://learn.microsoft.com/en-us/azure/azure-arc/servers/run-command-overview"""
            
        elif adapter_type == RemoteAdapterType.RDP:
            details += """
- Use Playwright to connect to RDP session and control the UI
- Take screenshots for verification
- Implement retry logic for UI operations
- Include functions to find and interact with UI elements"""
            
        elif adapter_type == RemoteAdapterType.VNC:
            details += """
- Use Playwright to connect to VNC session and control the UI
- Implement VNC-specific protocol handling
- Take screenshots for verification
- Include functions to find and interact with UI elements"""
            
        elif adapter_type == RemoteAdapterType.SSH:
            details += """
- Use Paramiko for SSH connection and command execution
- Implement key-based or password authentication
- Handle terminal output parsing
- Include timeouts and error handling"""
            
        elif adapter_type == RemoteAdapterType.AMT_KVM:
            details += """
- Use Intel AMT SDK for out-of-band management
- Implement KVM functionality for remote console
- Support power operations (on/off/reset)
- Documentation: https://www.intel.com/content/www/us/en/developer/articles/technical/getting-started-with-intel-amt.html"""
            
        elif adapter_type == RemoteAdapterType.VISION_FALLBACK:
            details += """
- Use a combination of Playwright for remote desktop and OpenAI GPT-4 Vision API
- Take screenshots and analyze them with the Vision API
- Convert natural language commands to UI actions
- Implement verification by taking screenshots after actions
- Use vision API to locate UI elements and guide interactions"""
            
        return details
    
    async def execute_tool(self, 
                        tool_id: str, 
                        args: Dict[str, Any],
                        record_video: bool = True,
                        timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a remote control tool with extra safety and logging.
        
        Args:
            tool_id: ID of the tool to execute
            args: Arguments for the tool
            record_video: Whether to record video/screenshots of the operation
            timeout: Optional timeout override
            
        Returns:
            Dictionary with execution results and audit information
        """
        # Get the tool
        tool = self.tooling_system.get_tool(tool_id)
        if not tool:
            raise ValueError(f"Tool not found: {tool_id}")
        
        # Verify it's a remote tool
        if tool.scope != ToolScope.REMOTE:
            raise ValueError(f"Tool {tool_id} is not a remote control tool")
        
        # Create audit record
        audit_record = {
            "tool_id": tool_id,
            "tool_name": tool.name,
            "host_id": tool.metadata.get("host_id"),
            "hostname": tool.metadata.get("hostname"),
            "adapter_type": tool.metadata.get("adapter_type"),
            "operation_type": tool.metadata.get("operation_type"),
            "args": args,
            "timestamp_start": datetime.now().isoformat(),
            "video_recording": None,
            "screenshots": []
        }
        # Start recording if needed
        recording_path = None
        if record_video and tool.metadata.get("adapter_type") in [
            RemoteAdapterType.RDP.value, 
            RemoteAdapterType.VNC.value, 
            RemoteAdapterType.VISION_FALLBACK.value
        ]:
            try:
                # Create recordings directory if it doesn't exist
                recordings_dir = os.path.join(
                    self.config.get("recordings_path", "recordings"),
                    datetime.now().strftime("%Y%m%d")
                )
                os.makedirs(recordings_dir, exist_ok=True)
                
                # Generate a unique filename for this recording
                timestamp = datetime.now().strftime("%H%M%S")
                host_part = tool.metadata.get("hostname", "unknown").split('.')[0]
                recording_filename = f"{timestamp}_{host_part}_{tool_id[-8:]}.mp4"
                recording_path = os.path.join(recordings_dir, recording_filename)
                
                # Set up recording configuration in the execution environment
                args["__recording_config"] = {
                    "enabled": True,
                    "path": recording_path,
                    "format": "mp4",
                    "fps": self.config.get("recording_fps", 5),
                    "resolution": self.config.get("recording_resolution", "1280x720")
                }
                
                # Update audit record with recording info
                audit_record["video_recording"] = {
                    "path": recording_path,
                    "format": "mp4",
                    "start_time": datetime.now().isoformat()
                }
                
                logging.info(f"Starting video recording for remote session at {recording_path}")
            except Exception as e:
                logging.error(f"Failed to set up video recording: {str(e)}")
                # Continue execution even if recording setup fails
                audit_record["video_recording"] = {
                    "error": f"Failed to set up recording: {str(e)}"
                }
        
        # Execute the tool with enhanced security
        result = await self.tooling_system.execute_tool_safely(
            tool_id=tool_id, 
            args=args,
            timeout=timeout or 120.0,  # Default 2-minute timeout for remote operations
            sandbox_override=SandboxType.DOCKER  # Always use Docker sandbox for remote ops
        )
        
        # Update audit record
        audit_record.update({
            "timestamp_end": datetime.now().isoformat(),
            "success": result.success,
            "error": result.error,
            "execution_time": result.execution_time
        })
        
        # Store screenshot if available in result
        if result.success and result.output and "screenshot" in result.output:
            audit_record["screenshots"].append({
                "timestamp": datetime.now().isoformat(),
                "data": result.output["screenshot"]
            })
        
        # Store audit record
        self.audit_log.append(audit_record)
        
        # Return result with audit information
        result_dict = {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "execution_time": result.execution_time,
            "audit_id": len(self.audit_log) - 1
        }
        
        return result_dict
    
    def get_audit_log(self, 
                    audit_id: Optional[int] = None, 
                    host_id: Optional[str] = None,
                    limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get remote operation audit logs.
        
        Args:
            audit_id: Optional specific audit ID
            host_id: Optional host ID to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of audit records
        """
        if audit_id is not None:
            if 0 <= audit_id < len(self.audit_log):
                return [self.audit_log[audit_id]]
            return []
            
        if host_id:
            # Filter by host ID
            logs = [log for log in self.audit_log if log.get("host_id") == host_id]
        else:
            logs = self.audit_log.copy()
            
        # Sort by timestamp (newest first) and limit
        logs.sort(key=lambda x: x.get("timestamp_start", ""), reverse=True)
        return logs[:limit]
