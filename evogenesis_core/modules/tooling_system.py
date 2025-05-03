"""
Tooling System Module - Manages tool generation, integration, and execution.

This module is responsible for automatically generating tools as needed,
securely integrating and executing them, and enabling tool discovery by agents.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import os
import sys
import json
import yaml
import time
import uuid
import logging
import importlib
import inspect
import subprocess
import traceback
from enum import Enum
import tempfile
import shutil
import hashlib
import threading
import asyncio
from datetime import datetime
import re
import base64
import docker
import requests
from pathlib import Path

# Import Perception-Action Tooling components when available
try:
    from evogenesis_core.modules.perception_action_tooling import (
        RemoteControlModule, RemoteAdapterType, RemoteTargetInfo
    )
    PERCEPTION_ACTION_AVAILABLE = True
except ImportError:
    # Define mock implementations for the required classes
    class RemoteAdapterType(str, Enum):
        """Types of adapters for remote machine control."""
        SSH = "ssh"
        RDP = "rdp"
        VNC = "vnc"
        GRAPH_CLOUDPC = "graph_cloudpc"
        DEV_BOX = "dev_box"
        AVD_REST = "avd_rest"
        ARC_COMMAND = "arc_command"
        AMT_KVM = "amt_kvm"
        VISION_FALLBACK = "vision"
    
    class RemoteTargetInfo:
        """Information about a remote machine target."""
        def __init__(self, host_id, hostname, ip_address=None, os_type=None, 
                    available_adapters=None, metadata=None):
            self.host_id = host_id
            self.hostname = hostname
            self.ip_address = ip_address
            self.os_type = os_type or "Windows"
            self.available_adapters = available_adapters or []
            self.metadata = metadata or {}
    
    class RemoteControlModule:
        """Mock implementation of the Remote Control Module."""
        def __init__(self, kernel=None):
            """Initialize the mock module. Accepts kernel for compatibility."""
            self.kernel = kernel
            
        async def discover_target(self, host_id, hostname, ip_address=None):
            return RemoteTargetInfo(
                host_id=host_id,
                hostname=hostname,
                ip_address=ip_address,
                os_type="Windows",
                available_adapters=[RemoteAdapterType.SSH, RemoteAdapterType.RDP],
                metadata={"status": "mocked", "capabilities": ["basic_control"]}
            )
        
        async def generate_remote_control_tool(self, host_id, hostname, description, 
                                           operation_type="general", parameters=None, 
                                           returns=None, ip_address=None):
            return f"mock-tool-{hash(description) % 10000}"
        
        async def execute_remote_control_tool(self, tool_id, parameters=None):
            return {
                "success": True, 
                "execution_id": f"mock-exec-{hash(tool_id) % 10000}",
                "result": f"Mock execution of {tool_id} completed successfully",
                "target": {"hostname": "mock-target", "host_id": "mock-id"}
            }
        
        async def get_audit_logs(self, start_time=None, end_time=None, 
                               tool_ids=None, host_ids=None, max_results=100):
            # Return mock audit logs for demonstration
            current_time = time.time()
            logs = []
            count = min(max_results, 5)
            for i in range(count):
                logs.append({
                    "audit_id": f"mock-audit-{i}",
                    "tool_name": tool_ids[0] if tool_ids else f"remote-tool-{i}",
                    "hostname": host_ids[0] if host_ids else "mock-target",
                    "timestamp_start": current_time - i * 60,
                    "timestamp_end": current_time - i * 60 + 30,
                    "success": i % 3 != 0
                })
            return logs
    
    # Set to True even though we're using the mock implementation
    PERCEPTION_ACTION_AVAILABLE = True
    logging.info("Using mock implementation for Perception-Action Tooling module")


class ToolScope(str, Enum):
    """Scope/permission level of a tool."""
    SYSTEM = "system"          # Full system access (highest privilege)
    WORKSPACE = "workspace"    # Access to workspace files and limited APIs
    CONTAINER = "container"    # Runs in a Docker container (recommended)
    CLOUD = "cloud"            # Runs in a cloud sandbox (most secure)
    MEMORY = "memory"          # In-memory execution only (no file/network)
    REMOTE = "remote"          # Remote machine control (requires special permissions)


class ToolStatus(str, Enum):
    """Status of a tool in the registry."""
    ACTIVE = "active"          # Tool is active and available
    TESTING = "testing"        # Tool is being tested
    DEPRECATED = "deprecated"  # Tool should not be used for new tasks
    DISABLED = "disabled"      # Tool is temporarily disabled
    FAILED = "failed"          # Tool has issues and cannot be used


class SandboxType(str, Enum):
    """Types of sandboxes for tool execution."""
    NONE = "none"              # No sandboxing (dangerous)
    SUBPROCESS = "subprocess"  # Run in a subprocess (minimal protection)
    DOCKER = "docker"          # Run in a Docker container (recommended)
    WASM = "wasm"              # WebAssembly sandbox (lighter than Docker)
    E2B = "e2b"                # E2B cloud sandbox (most secure)
    MODAL = "modal"            # Modal cloud sandbox (scalable)


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


class ExecutionResult:
    """Result of a tool execution."""
    
    def __init__(self, 
                success: bool, 
                output: Any = None, 
                error: Optional[str] = None,
                logs: Optional[str] = None,
                execution_time: float = 0.0,
                resource_usage: Optional[Dict[str, Any]] = None):
        self.success = success
        self.output = output
        self.error = error
        self.logs = logs
        self.execution_time = execution_time
        self.resource_usage = resource_usage or {}
        self.timestamp = datetime.now()
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "logs": self.logs,
            "execution_time": self.execution_time,
            "resource_usage": self.resource_usage,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """Create from dictionary representation."""
        return cls(
            success=data.get("success", False),
            output=data.get("output"),
            error=data.get("error"),
            logs=data.get("logs"),
            execution_time=data.get("execution_time", 0.0),
            resource_usage=data.get("resource_usage", {})
        )


class Tool:
    """A tool that can be invoked by agents."""
    
    def __init__(self,
                name: str,
                description: str,
                function: Union[Callable, str],
                scope: ToolScope = ToolScope.CONTAINER,
                parameters: Dict[str, Dict[str, Any]] = None,
                returns: Dict[str, Any] = None,
                metadata: Dict[str, Any] = None,
                sandbox_type: SandboxType = SandboxType.DOCKER,
                auto_generated: bool = False,
                file_path: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.function = function
        self.scope = scope
        self.parameters = parameters or {}
        self.returns = returns or {"type": "object"}
        self.metadata = metadata or {}
        self.sandbox_type = sandbox_type
        self.auto_generated = auto_generated
        self.file_path = file_path
        self.status = ToolStatus.TESTING if auto_generated else ToolStatus.ACTIVE
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.execution_count = 0
        self.success_count = 0
        self.error_count = 0
        self.average_execution_time = 0.0
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "parameters": self.parameters,
            "returns": self.returns,
            "metadata": self.metadata,
            "sandbox_type": self.sandbox_type,
            "auto_generated": self.auto_generated,
            "file_path": self.file_path,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "average_execution_time": self.average_execution_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tool':
        """Create from dictionary representation."""
        tool = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            function="",  # Function will be loaded separately
            scope=data.get("scope", ToolScope.CONTAINER),
            parameters=data.get("parameters", {}),
            returns=data.get("returns", {"type": "object"}),
            metadata=data.get("metadata", {}),
            sandbox_type=data.get("sandbox_type", SandboxType.DOCKER),
            auto_generated=data.get("auto_generated", False),
            file_path=data.get("file_path")
        )
        
        # Update additional fields
        tool.id = data.get("id", tool.id)
        tool.status = data.get("status", tool.status)
        tool.created_at = datetime.fromisoformat(data.get("created_at", tool.created_at.isoformat()))
        tool.updated_at = datetime.fromisoformat(data.get("updated_at", tool.updated_at.isoformat()))
        tool.execution_count = data.get("execution_count", 0)
        tool.success_count = data.get("success_count", 0)
        tool.error_count = data.get("error_count", 0)
        tool.average_execution_time = data.get("average_execution_time", 0.0)
        
        return tool


class ToolGenerator:
    """Generates tools from natural language descriptions using LLMs."""
    
    def __init__(self, kernel):
        """
        Initialize the tool generator.
        
        Args:
            kernel: Reference to the EvoGenesis kernel
        """
        self.kernel = kernel
        self.llm_orchestrator = kernel.llm_orchestrator
        self.generation_templates = {
            "python_script": "Generate a Python script that {description}. The script should accept the following parameters: {parameters}. It should return {returns}.",
            "api_client": "Generate a Python function that connects to {api_name} API to {description}. It should accept the following parameters: {parameters} and return {returns}.",
            "web_scraper": "Generate a Python function that scrapes {website} to {description}. It should accept the following parameters: {parameters} and return {returns}."
        }
        self.test_case_template = "Generate a test case for the function that {description}. Include example inputs and expected outputs."
    
    async def generate_tool(self, 
                          description: str, 
                          parameters: Dict[str, Dict[str, Any]], 
                          returns: Dict[str, Any],
                          tool_type: str = "python_script",
                          extra_context: Dict[str, Any] = None) -> Optional[Tool]:
        """
        Generate a tool from a natural language description.
        
        Args:
            description: Natural language description of what the tool should do
            parameters: Dictionary of parameter names and their specifications
            returns: Specification of what the tool should return
            tool_type: Type of tool to generate (python_script, api_client, web_scraper)
            extra_context: Additional context to include in the prompt
            
        Returns:
            Generated Tool object or None if generation failed
        """
        # Prepare the prompt
        template = self.generation_templates.get(tool_type, self.generation_templates["python_script"])
        
        # Format parameters for the prompt
        params_str = ", ".join([f"{name}: {param['type']}" for name, param in parameters.items()])
        
        # Format returns for the prompt
        returns_str = f"a {returns.get('type', 'dict')} containing {', '.join(returns.get('properties', {}).keys())}" \
                     if returns.get('type') == 'object' and 'properties' in returns else \
                     f"a {returns.get('type', 'dict')}"
        
        # Replace placeholders in the template
        prompt = template.format(
            description=description,
            parameters=params_str,
            returns=returns_str,
            **extra_context or {}
        )
        
        # Add code generation guidelines
        prompt += "\n\nThe code should be secure, efficient, and well-documented with docstrings. " \
                 "Include error handling for all possible failure cases. " \
                 "The function should be named appropriately for its purpose."
        
        # Generate the code using an appropriate model
        try:
            response = await self.llm_orchestrator.execute_prompt_async(
                task_type="code_generation",
                prompt_template="direct",
                params={"prompt": prompt},
                model_selection={
                    "model_name": "gpt-4o",  # Use a strong model for code generation
                    "provider": "openai"
                },
                max_tokens=2000
            )
            
            # Extract code from the response
            code = response.get("result", "")
            
            # If the result is a string, try to extract a code block
            if isinstance(code, str):
                # Try to extract code block if present
                code_match = re.search(r'```python\s*(.*?)\s*```', code, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
            
            # Generate a name for the tool based on the description
            tool_name = self._generate_tool_name(description)
            
            # Save the code to a file
            file_path = self._save_tool_code(tool_name, code)
            
            # Create the tool object
            tool = Tool(
                name=tool_name,
                description=description,
                function=code,
                scope=ToolScope.CONTAINER,  # Default to container for security
                parameters=parameters,
                returns=returns,
                metadata={
                    "tool_type": tool_type,
                    "generation_prompt": prompt,
                    "extra_context": extra_context
                },
                sandbox_type=SandboxType.DOCKER,
                auto_generated=True,
                file_path=file_path
            )
            
            # Generate a test case for the tool
            await self._generate_test_case(tool)
            
            return tool
            
        except Exception as e:
            logging.error(f"Error generating tool: {str(e)}")
            return None
    
    def _generate_tool_name(self, description: str) -> str:
        """Generate a appropriate name for a tool based on its description."""
        # Extract key actions from the description
        action_words = ["get", "fetch", "retrieve", "create", "update", "delete", 
                      "convert", "transform", "analyze", "search", "extract", "generate"]
        
        # Find the first action word in the description
        action = next((word for word in action_words if word in description.lower()), "tool")
        
        # Generate a name based on the action and key words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', description)
        words = [w for w in words if w.lower() not in ['the', 'and', 'for', 'from', 'with', 'that']]
        
        if not words:
            # If no suitable words found, use a generic name with a timestamp
            return f"{action}_tool_{int(time.time())}"
        
        # Combine 2-3 key words for the name
        name_parts = [action] + words[:min(2, len(words))]
        return "_".join(name_parts).lower()
    
    def _save_tool_code(self, tool_name: str, code: str) -> str:
        """
        Save tool code to a file.
        
        Args:
            tool_name: Name of the tool
            code: The code to save
            
        Returns:
            Path to the saved file
        """
        # Create tools directory if it doesn't exist
        tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools", "generated")
        os.makedirs(tools_dir, exist_ok=True)
        
        # Create a file for the tool
        file_path = os.path.join(tools_dir, f"{tool_name}.py")
        
        # Add metadata header
        header = f"""#!/usr/bin/env python
# Tool: {tool_name}
# Generated by EvoGenesis ToolGenerator
# Created: {datetime.now().isoformat()}
# This is an auto-generated tool. Use with appropriate security precautions.

"""
        
        # Write the code to the file
        with open(file_path, "w") as f:
            f.write(header + code)
        
        return file_path
    
    async def _generate_test_case(self, tool: Tool) -> None:
        """
        Generate test cases for a tool.
        
        Args:
            tool: The tool to generate tests for
        """
        # Prepare prompt for test case generation
        prompt = self.test_case_template.format(description=tool.description)
        
        # Add information about parameters and return type
        prompt += f"\n\nParameters: {json.dumps(tool.parameters, indent=2)}"
        prompt += f"\n\nReturns: {json.dumps(tool.returns, indent=2)}"
        
        # If we have the function code, include it
        if isinstance(tool.function, str):
            prompt += f"\n\nFunction code:\n```python\n{tool.function}\n```"
        
        # Generate test case
        try:
            response = await self.llm_orchestrator.execute_prompt_async(
                task_type="code_generation",
                prompt_template="direct",
                params={"prompt": prompt},
                max_tokens=1000
            )
            
            # Extract test code
            test_code = response.get("result", "")
            
            # If the result is a string, try to extract a code block
            if isinstance(test_code, str):
                # Try to extract code block if present
                code_match = re.search(r'```python\s*(.*?)\s*```', test_code, re.DOTALL)
                if code_match:
                    test_code = code_match.group(1)
            
            # Save the test code
            if tool.file_path:
                test_file_path = tool.file_path.replace(".py", "_test.py")
                with open(test_file_path, "w") as f:
                    f.write(f"""#!/usr/bin/env python
# Test for Tool: {tool.name}
# Generated by EvoGenesis ToolGenerator
# Created: {datetime.now().isoformat()}

import unittest
import json
import sys
import os

# Add parent directory to path to import the tool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the tool function
from generated.{os.path.basename(tool.file_path).replace(".py", "")} import *

{test_code}

if __name__ == "__main__":
    unittest.main()
""")
                
                # Update tool metadata
                tool.metadata["test_file_path"] = test_file_path
                
        except Exception as e:
            logging.error(f"Error generating test case for tool {tool.name}: {str(e)}")


class SecureExecutionEnvironment:
    """
    Provides secure environments for executing tools with different isolation levels.
    
    Supports:
    - Subprocess execution (minimal isolation)
    - Docker containers (strong isolation)
    - WebAssembly sandboxes (lightweight)
    - Cloud sandboxes (E2B, Modal) for maximum security
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the secure execution environment.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.docker_client = None
        self.wasm_runtime = None
        self.e2b_client = None
        self.modal_client = None
        
        # Resource limits
        self.resource_limits = self.config.get("resource_limits", {
            "memory": "256m",
            "cpu": "0.5",
            "timeout": 30,
            "disk": "100m",
            "network": False
        })
        
        # Initialize Docker if available
        if self.config.get("enable_docker", True):
            try:
                self.docker_client = docker.from_env()
                logging.info("Docker environment initialized")
            except Exception as e:
                logging.warning(f"Could not initialize Docker: {str(e)}")
        
        # Initialize other sandbox environments
        self._init_wasm()
        self._init_cloud_sandboxes()
    
    def _init_wasm(self):
        """Initialize WebAssembly runtime if configured."""
        if self.config.get("enable_wasm", False):
            try:
                # This would typically use a WASM runtime like wasmtime or wasmer
                # For now, just log that it would be initialized
                logging.info("WebAssembly runtime would be initialized here")
                self.wasm_runtime = True  # Placeholder
            except Exception as e:
                logging.warning(f"Could not initialize WebAssembly runtime: {str(e)}")
    
    def _init_cloud_sandboxes(self):
        """Initialize cloud sandbox connections if configured."""
        # E2B initialization if configured
        if self.config.get("enable_e2b", False) and "e2b_api_key" in self.config:
            try:
                # Placeholder for E2B SDK initialization
                logging.info("E2B sandbox would be initialized here")
                self.e2b_client = True  # Placeholder
            except Exception as e:
                logging.warning(f"Could not initialize E2B sandbox: {str(e)}")
        
        # Modal initialization if configured
        if self.config.get("enable_modal", False):
            try:
                # Placeholder for Modal SDK initialization
                logging.info("Modal client would be initialized here")
                self.modal_client = True  # Placeholder
            except Exception as e:
                logging.warning(f"Could not initialize Modal client: {str(e)}")
    
    async def execute(self, 
                    tool: Tool, 
                    args: Dict[str, Any],
                    sandbox_type: Optional[SandboxType] = None,
                    timeout: Optional[float] = None,
                    resource_limits: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute a tool in a secure environment.
        
        Args:
            tool: Tool to execute
            args: Arguments for the tool
            sandbox_type: Override the tool's default sandbox type
            timeout: Maximum execution time in seconds
            resource_limits: Custom resource limits for this execution
            
        Returns:
            ExecutionResult object
        """
        sandbox = sandbox_type or tool.sandbox_type
        
        # Apply resource limits with fallback to defaults
        limits = resource_limits or self.resource_limits
        timeout = timeout or limits.get("timeout", 30)
        
        # Select execution method based on sandbox type
        if sandbox == SandboxType.NONE:
            return await self._execute_direct(tool, args, timeout)
        elif sandbox == SandboxType.SUBPROCESS:
            return await self._execute_subprocess(tool, args, timeout)
        elif sandbox == SandboxType.DOCKER:
            return await self._execute_docker(tool, args, timeout, limits)
        elif sandbox == SandboxType.WASM:
            return await self._execute_wasm(tool, args, timeout, limits)
        elif sandbox == SandboxType.E2B:
            return await self._execute_e2b(tool, args, timeout, limits)
        elif sandbox == SandboxType.MODAL:
            return await self._execute_modal(tool, args, timeout, limits)
        else:
            # Default to safest available option
            if self.docker_client:
                return await self._execute_docker(tool, args, timeout, limits)
            else:
                return await self._execute_subprocess(tool, args, timeout)
    
    async def _execute_direct(self, 
                           tool: Tool, 
                           args: Dict[str, Any],
                           timeout: float) -> ExecutionResult:
        """
        Execute a tool directly in the current process (DANGEROUS).
        Only use for trusted, system-level tools.
        
        Args:
            tool: Tool to execute
            args: Arguments for the tool
            timeout: Timeout in seconds
            
        Returns:
            ExecutionResult object
        """
        logging.warning(f"Executing tool {tool.name} directly without sandbox!")
        
        start_time = time.time()
        
        try:
            # If the function is a string (code), load it
            if isinstance(tool.function, str):
                # Create a module in memory
                module_name = f"dynamic_tool_{tool.id}"
                spec = importlib.util.find_spec("types")
                module = importlib.util.module_from_spec(spec)
                
                # Extract the function name
                func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', tool.function)
                if not func_match:
                    return ExecutionResult(
                        success=False,
                        error="Could not identify function name in tool code",
                        execution_time=time.time() - start_time
                    )
                
                function_name = func_match.group(1)
                
                # Execute the code in the module's context
                exec(tool.function, module.__dict__)
                
                # Get the function
                function = getattr(module, function_name)
            else:
                # Function is already callable
                function = tool.function
            
            # Set up asyncio task with timeout
            async def run_with_timeout():
                if asyncio.iscoroutinefunction(function):
                    return await function(**args)
                else:
                    return await asyncio.to_thread(function, **args)
            
            # Execute with timeout
            result = await asyncio.wait_for(run_with_timeout(), timeout)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                output=result,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout} seconds",
                execution_time=timeout
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                logs=traceback.format_exc(),
                execution_time=time.time() - start_time
            )
    
    async def _execute_subprocess(self, 
                              tool: Tool, 
                              args: Dict[str, Any],
                              timeout: float) -> ExecutionResult:
        """
        Execute a tool in a subprocess.
        
        Args:
            tool: Tool to execute
            args: Arguments for the tool
            timeout: Timeout in seconds
            
        Returns:
            ExecutionResult object
        """
        start_time = time.time()
        
        # Create a temporary file with the code if needed
        temp_dir = None
        if isinstance(tool.function, str):
            # If we don't already have a file path, create one
            if not tool.file_path:
                temp_dir = tempfile.mkdtemp()
                script_path = os.path.join(temp_dir, f"{tool.name}.py")
                
                # Write the function to a file
                with open(script_path, "w") as f:
                    f.write(tool.function)
                    
                    # Add code to call the function with provided args
                    func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', tool.function)
                    if func_match:
                        function_name = func_match.group(1)
                        f.write(f"\n\nif __name__ == '__main__':\n")
                        f.write(f"    import sys\n")
                        f.write(f"    import json\n")
                        f.write(f"    args = json.loads(sys.argv[1])\n")
                        f.write(f"    result = {function_name}(**args)\n")
                        f.write(f"    print(json.dumps(result))\n")
            else:
                script_path = tool.file_path
        else:
            # If the function is callable but not code, we can't run it in a subprocess
            return ExecutionResult(
                success=False,
                error="Cannot execute callable in subprocess",
                execution_time=time.time() - start_time
            )
        
        try:
            # Prepare the subprocess command
            cmd = [sys.executable, script_path, json.dumps(args)]
            
            # Execute the subprocess with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
                
                # Process the result
                if process.returncode == 0:
                    try:
                        output = json.loads(stdout.decode())
                        return ExecutionResult(
                            success=True,
                            output=output,
                            logs=stderr.decode() if stderr else None,
                            execution_time=time.time() - start_time
                        )
                    except json.JSONDecodeError:
                        return ExecutionResult(
                            success=True,
                            output=stdout.decode(),
                            logs=stderr.decode() if stderr else None,
                            execution_time=time.time() - start_time
                        )
                else:
                    return ExecutionResult(
                        success=False,
                        error=f"Process returned non-zero exit code: {process.returncode}",
                        logs=stderr.decode() if stderr else None,
                        execution_time=time.time() - start_time
                    )
                    
            except asyncio.TimeoutError:
                # Kill the process if it times out
                process.kill()
                return ExecutionResult(
                    success=False,
                    error=f"Execution timed out after {timeout} seconds",
                    execution_time=timeout
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                logs=traceback.format_exc(),
                execution_time=time.time() - start_time
            )
        finally:
            # Cleanup temporary directory if created
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def _execute_docker(self, 
                          tool: Tool, 
                          args: Dict[str, Any],
                          timeout: float,
                          limits: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a tool in a Docker container.
        
        Args:
            tool: Tool to execute
            args: Arguments for the tool
            timeout: Timeout in seconds
            limits: Resource limits
            
        Returns:
            ExecutionResult object
        """
        if not self.docker_client:
            logging.warning("Docker client not available, falling back to subprocess")
            return await self._execute_subprocess(tool, args, timeout, limits)
        
        start_time = time.time()
        container = None
        temp_dir = None
        
        try:
            # Create a temporary directory to mount in the container
            temp_dir = tempfile.mkdtemp()
            
            # If function is a string, write it to a file
            if isinstance(tool.function, str):
                script_path = os.path.join(temp_dir, "tool.py")
                
                # Write the function to a file
                with open(script_path, "w") as f:
                    f.write(tool.function)
                    
                    # Extract the function name
                    func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', tool.function)
                    if not func_match:
                        return ExecutionResult(
                            success=False,
                            error="Could not identify function name in tool code",
                            execution_time=time.time() - start_time
                        )
                    
                    function_name = func_match.group(1)
                    
                    # Add code to call the function with provided args
                    f.write(f"\n\nif __name__ == '__main__':\n")
                    f.write(f"    import sys\n")
                    f.write(f"    import json\n")
                    f.write(f"    args = json.loads(sys.argv[1])\n")
                    f.write(f"    result = {function_name}(**args)\n")
                    f.write(f"    print(json.dumps(result))\n")
            else:
                # If the function is callable but not code, we can't run it in Docker
                return ExecutionResult(
                    success=False,
                    error="Cannot execute callable in Docker container",
                    execution_time=time.time() - start_time
                )
            
            # Write arguments to a file
            args_path = os.path.join(temp_dir, "args.json")
            with open(args_path, "w") as f:
                json.dump(args, f)
            
            # Determine which image to use
            image = self.config.get("docker_image", "python:3.9-slim")
            
            # Apply resource limits
            mem_limit = limits.get("memory", "256m")
            cpu_limit = limits.get("cpu", "0.5")
            
            # Network access
            network_mode = "none" if not limits.get("network", False) else "bridge"
            
            # Determine if any mount points are requested via permissions
            mounts = []
            if "mount_points" in tool.metadata:
                for mount in tool.metadata["mount_points"]:
                    # Validate against allowed mount points
                    if mount in self.config.get("allowed_mounts", []):
                        mounts.append(mount)
            
            # Create and run the container
            container = self.docker_client.containers.run(
                image,
                ["python", "/workspace/tool.py", json.dumps(args)],
                volumes={temp_dir: {"bind": "/workspace", "mode": "ro"}},
                mem_limit=mem_limit,
                cpu_quota=int(float(cpu_limit) * 100000),
                network_mode=network_mode,
                detach=True,
                remove=False,
                working_dir="/workspace"
            )
            
            # Wait for the container to finish with timeout
            try:
                container_logs = []
                started = time.time()
                
                while container.status != "exited" and time.time() - started < timeout:
                    # Update container status
                    container.reload()
                    
                    # Collect logs incrementally
                    new_logs = container.logs(stdout=True, stderr=True, stream=False, tail=10)
                    if new_logs:
                        container_logs.append(new_logs)
                    
                    # Short sleep to prevent CPU spinning
                    await asyncio.sleep(0.1)
                
                # If still running after timeout, kill it
                if container.status != "exited":
                    container.kill()
                    return ExecutionResult(
                        success=False,
                        error=f"Execution timed out after {timeout} seconds",
                        logs=b"".join(container_logs).decode("utf-8", errors="replace"),
                        execution_time=timeout
                    )
                
                # Get final logs
                logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
                exit_code = container.attrs["State"]["ExitCode"]
                
                # Get resource usage
                container.reload()
                resource_usage = {
                    "memory_max": container.attrs.get("HostConfig", {}).get("Memory", 0),
                    "cpu_usage": container.attrs.get("HostConfig", {}).get("CpuQuota", 0) / 100000
                }
                
                if exit_code == 0:
                    # Parse output (last line should be JSON result)
                    output_lines = logs.strip().split("\n")
                    try:
                        output = json.loads(output_lines[-1])
                        return ExecutionResult(
                            success=True,
                            output=output,
                            logs="\n".join(output_lines[:-1]) if len(output_lines) > 1 else None,
                            execution_time=time.time() - start_time,
                            resource_usage=resource_usage
                        )
                    except (json.JSONDecodeError, IndexError):
                        # If can't parse as JSON, return the full output
                        return ExecutionResult(
                            success=True,
                            output=logs,
                            execution_time=time.time() - start_time,
                            resource_usage=resource_usage
                        )
                else:
                    return ExecutionResult(
                        success=False,
                        error=f"Container exited with code {exit_code}",
                        logs=logs,
                        execution_time=time.time() - start_time,
                        resource_usage=resource_usage
                    )
                
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    error=f"Error monitoring container: {str(e)}",
                    logs=container.logs().decode("utf-8", errors="replace") if container else None,
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            logging.error(f"Error executing in Docker: {str(e)}")
            return ExecutionResult(
                success=False,
                error=str(e),
                logs=traceback.format_exc(),
                execution_time=time.time() - start_time
            )
        finally:
            # Cleanup
            if container:
                try:
                    container.remove(force=True)
                except:
                    pass
            
            if temp_dir:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass
    
    async def _execute_wasm(self, 
                         tool: Tool, 
                         args: Dict[str, Any],
                         timeout: float,
                         limits: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a tool in a WebAssembly sandbox.
        
        Args:
            tool: Tool to execute
            args: Arguments for the tool
            timeout: Timeout in seconds
            limits: Resource limits
            
        Returns:
            ExecutionResult object
        """
        start_time = time.time()
        
        try:
            # Check if wasmer library is available
            import wasmer
            import wasmer_compiler_cranelift
            import os
            import tempfile
            import subprocess
            import json
            
            logging.info(f"Executing {tool.name} in WebAssembly sandbox")
            
            # Create temp directory for files
            with tempfile.TemporaryDirectory() as temp_dir:
                # First, we need to compile the Python code to WASM
                # We'll use pyodide as it's the most mature Python->WASM solution
                
                # Write the Python code to a file
                py_file = os.path.join(temp_dir, "tool.py")
                with open(py_file, "w") as f:
                    # Extract the function code
                    if isinstance(tool.function, str):
                        f.write(tool.function)
                    else:
                        raise ValueError("Only string-based functions can be executed in WASM")
                # Create a wrapper script that calls the function with arguments
                wrapper_file = os.path.join(temp_dir, "wrapper.py")
                with open(wrapper_file, "w") as f:
                    # This wrapper assumes the WASM runtime executes this Python script
                    # and passes the JSON args as the first command-line argument.
                    # It also assumes the 'tool' module (tool.py) is available.
                    f.write("""
import sys
import json
import traceback
from tool import * # Import the user's code

# Load arguments from command-line (passed as JSON string)
try:
    args = json.loads(sys.argv[1])
except (IndexError, json.JSONDecodeError) as e:
    print(json.dumps({"error": f"Failed to load/parse args: {e}"}))
    sys.exit(1)

# Find the primary function to call (simple heuristic: first callable found)
# A more robust method might require the tool definition to specify the entry point.
main_fn = None
for name, obj in locals().items():
    # Check if it's a function defined in the 'tool' module (or globally if not module-scoped)
    # and avoid builtins or imported names like 'json', 'sys'.
    if callable(obj) and not name.startswith('_') and name not in ['json', 'sys', 'traceback']:
        # Check if the object originates from the 'tool' module if possible
        try:
            if obj.__module__ == 'tool':
                 main_fn = obj
                 break
        except AttributeError:
             # Fallback for functions without __module__ or if 'tool' isn't a real module context
             if name != 'main_fn': # Avoid picking itself
                 main_fn = obj
                 break
        # If module check fails, take the first callable as a guess
        if not main_fn and name != 'main_fn':
             main_fn = obj
             break


if main_fn:
    try:
        result = main_fn(**args)
        # Attempt to serialize result as JSON
        try:
            print(json.dumps({"output": result}))
        except (TypeError, OverflowError) as json_err:
            # Fallback if result is not JSON serializable
            print(json.dumps({"output": repr(result), "warning": f"Result not JSON serializable: {json_err}"}))
    except Exception as e:
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
        sys.exit(1)
else:
    print(json.dumps({"error": "Could not find a suitable function to call in tool.py"}))
    sys.exit(1)
""")

                # Check if we have a pre-built WASM runtime available
                # This runtime MUST contain a Python interpreter (e.g., CPython compiled to WASM)
                wasm_runtime_path = self.config.get("wasm", {}).get("runtime_path")
                if not wasm_runtime_path or not os.path.exists(wasm_runtime_path):
                    logging.warning("WASM runtime path not configured or not found, falling back to Docker")
                    return await self._execute_docker(tool, args, timeout, limits)

                # Run the code using the WASM runtime
                args_json = json.dumps(args)

                # Execute with resource limits (Memory limit applied via Wasmer config if supported)
                # CPU/Timeout limits are harder to enforce directly on WASM execution from Python
                # Using signal.alarm is a host-level timeout, not internal WASM preemption.
                memory_limit_mb = int(limits.get("memory_mb", 128)) # Use MiB for Wasmer if possible

                # Create WASM instance with limits
                # Note: Memory limiting might depend on the specific WASM runtime and WASI implementation.
                store = wasmer.Store() # Default store
                module = wasmer.Module(store, open(wasm_runtime_path, 'rb').read())

                # Create WASI environment
                wasi_env = (
                    wasmer.wasi.StateBuilder("wasm_tool_runner")
                    .argument("wrapper.py") # Arg 0 is script name
                    .argument(args_json)    # Arg 1 is JSON args
                    .map_directory(".", temp_dir) # Map temp dir to WASM's root
                    .capture_stdout() # Capture stdout
                    .capture_stderr() # Capture stderr
                    .finalize()
                )

                # Create import object
                import_object = wasi_env.generate_import_object(store, module)

                # Instantiate the module
                instance = wasmer.Instance(module, import_object)
                start = instance.exports._start # WASI entry point

                # Set execution timeout using signal (best effort for synchronous WASM runs)
                signal_handler = None
                try:
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"WASM execution timed out after {timeout} seconds")

                    signal_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout))

                    # Run the WASM module's start function
                    start()

                    # Cancel the alarm
                    signal.alarm(0)

                    # Get captured stdout/stderr
                    stdout_bytes = wasi_env.stdout()
                    stderr_bytes = wasi_env.stderr()
                    stdout = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ""
                    stderr = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ""

                    # Parse the output from stdout (expecting JSON from wrapper)
                    try:
                        result_data = json.loads(stdout)
                        if "error" in result_data:
                             return ExecutionResult(
                                 success=False,
                                 error=result_data["error"],
                                 logs=result_data.get("traceback", stderr),
                                 execution_time=time.time() - start_time,
                                 resource_usage={"memory_limit_mb": memory_limit_mb}
                             )
                        else:
                             return ExecutionResult(
                                 success=True,
                                 output=result_data.get("output"),
                                 logs=stderr or result_data.get("warning"), # Combine stderr and warnings
                                 execution_time=time.time() - start_time,
                                 resource_usage={"memory_limit_mb": memory_limit_mb}
                             )
                    except json.JSONDecodeError:
                         # Handle cases where stdout wasn't valid JSON
                         return ExecutionResult(
                             success=False, # Treat non-JSON output as failure unless expected
                             error="WASM script produced non-JSON output",
                             output=stdout, # Provide raw output for debugging
                             logs=stderr,
                             execution_time=time.time() - start_time,
                             resource_usage={"memory_limit_mb": memory_limit_mb}
                         )

                except TimeoutError as e:
                    return ExecutionResult(
                        success=False,
                        error=str(e),
                        logs="WASM execution timed out.",
                        execution_time=timeout,
                        resource_usage={"memory_limit_mb": memory_limit_mb}
                    )
                except Exception as e:
                    logging.error(f"Error during WASM execution: {str(e)}")
                    return ExecutionResult(
                        success=False,
                        error=f"WASM execution error: {str(e)}",
                        logs=traceback.format_exc(),
                        execution_time=time.time() - start_time,
                        resource_usage={"memory_limit_mb": memory_limit_mb}
                    )
                finally:
                    # Reset signal handler and cancel alarm
                    if signal_handler:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, signal_handler) # Restore previous handler

        except ImportError:
            logging.warning("WASM libraries (wasmer, wasmer_compiler_cranelift) not available, falling back to Docker")
            return await self._execute_docker(tool, args, timeout, limits)
        except Exception as e:
            logging.error(f"Error setting up WASM environment: {str(e)}")
            # Fallback on general WASM setup error
            return await self._execute_docker(tool, args, timeout, limits)

    async def _execute_e2b(self,
                        tool: Tool,
                        args: Dict[str, Any],
                        timeout: float,
                        limits: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a tool in an E2B cloud sandbox.

        Args:
            tool: Tool to execute
            args: Arguments for the tool
            timeout: Timeout in seconds
            limits: Resource limits (Note: E2B applies its own limits based on template)

        Returns:
            ExecutionResult object
        """
        if not self.e2b_client: # Check if E2B is configured/enabled
            logging.warning("E2B client not available or configured, falling back to Docker")
            return await self._execute_docker(tool, args, timeout, limits)

        start_time = time.time()
        logging.info(f"Executing tool '{tool.name}' in E2B sandbox")
        sandbox = None # Ensure sandbox variable is defined

        try:
            # Dynamic import to avoid hard dependency
            try:
                import e2b
                from e2b import Sandbox # Use Sandbox directly
            except ImportError:
                 logging.error("E2B SDK not installed. Install with: pip install e2b")
                 return ExecutionResult(
                     success=False,
                     error="E2B SDK not installed",
                     logs="Error: E2B SDK not installed. Install with: pip install e2b",
                     execution_time=time.time() - start_time
                 )

            # Configure E2B client with API key
            e2b_api_key = self.config.get("e2b", {}).get("api_key") or os.environ.get("E2B_API_KEY")
            if not e2b_api_key:
                raise ValueError("E2B API key not provided in config or environment variables (E2B_API_KEY)")

            # E2B initialization might be handled globally, but creating Sandbox often needs the key
            # e2b.configure(api_key=e2b_api_key) # Or pass directly to Sandbox

            # Create a new sandbox instance
            template = tool.metadata.get("e2b_template", "base") # Get template from tool metadata
            logging.debug(f"Creating E2B sandbox with template '{template}'")
            sandbox = await Sandbox.create(template=template, api_key=e2b_api_key)

            # Prepare code and dependencies
            if not isinstance(tool.function, str):
                 raise ValueError("E2B execution requires tool function as code string")
            code = tool.function
            requirements = tool.metadata.get("requirements", [])

            # Install dependencies if needed
            if requirements:
                install_cmd = f"pip install --no-cache-dir {' '.join(requirements)}" # Use --no-cache-dir
                logging.debug(f"Installing dependencies in E2B sandbox: {install_cmd}")
                # Use a reasonable timeout for installation
                install_timeout = self.config.get("e2b_install_timeout", 300.0)
                install_proc = await sandbox.process.start_and_wait(install_cmd, timeout=install_timeout)
                if install_proc.exit_code != 0:
                    error_logs = install_proc.stderr or install_proc.stdout # Combine logs on error
                    raise RuntimeError(f"Failed to install dependencies in E2B sandbox (exit code {install_proc.exit_code}): {error_logs}")
                logging.debug("Dependencies installed successfully.")

            # Create a file with the code and a wrapper to handle args/output
            file_name = f"{tool.name.lower().replace(' ', '_')}.py"
            func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
            if not func_match:
                raise ValueError("Could not identify function name in tool code")
            function_name = func_match.group(1)

            # Wrapper reads args from stdin, calls function, prints JSON result to stdout
            wrapper_code = f"""
import sys
import json
import traceback

# --- User Code Start ---
{code}
# --- User Code End ---

if __name__ == '__main__':
    try:
        # Read args from stdin
        input_json = sys.stdin.read()
        args = json.loads(input_json)

        # Call the user's function
        result = {function_name}(**args)

        # Print JSON output to stdout
        try:
            print(json.dumps({{"output": result}}))
        except (TypeError, OverflowError) as json_err:
            return {"output": repr(result), "warning": f"Result not JSON serializable: {json_err}"}
    except Exception as e:
        # Print JSON error to stdout and exit with non-zero code
        print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()}}))
        sys.exit(1)
"""
            await sandbox.filesystem.write(file_name, wrapper_code)
            logging.debug(f"Tool code written to {file_name} in sandbox.")

            # Execute the script, passing args via stdin
            logging.debug(f"Starting tool execution in E2B sandbox with timeout {timeout}s.")
            execution = await sandbox.process.start_and_wait(
                f"python {file_name}",
                stdin=json.dumps(args),
                timeout=timeout # Apply execution timeout
            )
            logging.debug(f"E2B process finished with exit code {execution.exit_code}.")

            # Get output and logs
            output_text = execution.stdout
            logs = execution.stderr if execution.stderr else "No stderr logs."

            # E2B resource usage stats might require specific calls if available in the SDK
            resource_usage = {"platform": "e2b"}

            # Parse output JSON from stdout
            try:
                result_data = json.loads(output_text)
                if "error" in result_data:
                    return ExecutionResult(
                        success=False,
                        error=result_data["error"],
                        logs=result_data.get("traceback", logs),
                        execution_time=time.time() - start_time,
                        resource_usage=resource_usage
                    )
                else:
                    return ExecutionResult(
                        success=True,
                        output=result_data.get("output"),
                        logs=logs or result_data.get("warning"), # Combine stderr and warnings
                        execution_time=time.time() - start_time,
                        resource_usage=resource_usage
                    )
            except json.JSONDecodeError:
                # Handle non-JSON output or execution errors
                if execution.exit_code == 0:
                     # Success exit code but non-JSON output might be valid or an error
                     return ExecutionResult(
                         success=True, # Assume success if exit code is 0, but log warning
                         output=output_text, # Return raw output
                         logs=logs + "\n[Warning] Tool output was not valid JSON.",
                         execution_time=time.time() - start_time,
                         resource_usage=resource_usage
                     )
                else:
                     # Non-zero exit code, likely an error
                     return ExecutionResult(
                         success=False,
                         error=f"E2B process exited with code {execution.exit_code}",
                         output=output_text, # Include output for debugging
                         logs=logs,
                         execution_time=time.time() - start_time,
                         resource_usage=resource_usage
                     )

        except Exception as e:
            logging.error(f"Error executing tool '{tool.name}' in E2B: {str(e)}")
            return ExecutionResult(
                success=False,
                error=str(e),
                logs=traceback.format_exc(),
                execution_time=time.time() - start_time
            )
        finally:
            # Ensure sandbox is closed if it was created
            if sandbox:
                try:
                    await sandbox.close()
                    logging.debug("E2B sandbox closed.")
                except Exception as close_err:
                    logging.warning(f"Error closing E2B sandbox: {close_err}")

    async def _execute_modal(self,
                          tool: Tool,
                          args: Dict[str, Any],
                          timeout: float,
                          limits: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a tool in a Modal cloud sandbox.

        Args:
            tool: Tool to execute
            args: Arguments for the tool
            timeout: Timeout in seconds
            limits: Resource limits (memory_mb)

        Returns:
            ExecutionResult object
        """
        if not self.modal_client: # Check if Modal is configured/enabled
            logging.warning("Modal client not available or configured, falling back to Docker")
            return await self._execute_docker(tool, args, timeout, limits)

        start_time = time.time()
        logging.info(f"Executing tool '{tool.name}' in Modal cloud")

        try:
            # Dynamic import
            try:
                import modal
            except ImportError:
                logging.error("Modal SDK not installed. Install with: pip install modal-client")
                return ExecutionResult(
                    success=False,
                    error="Modal SDK not installed",
                    logs="Error: Modal SDK not installed. Install with: pip install modal-client",
                    execution_time=time.time() - start_time
                )

            # Ensure tool function is code string
            if not isinstance(tool.function, str):
                raise ValueError("Modal execution requires tool function as code string")
            tool_code = tool.function

            # Extract function name (using the same logic as E2B/WASM wrappers)
            func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', tool_code)
            if not func_match:
                raise ValueError("Could not identify function name in tool code")
            function_name = func_match.group(1)

            # Define Modal function dynamically using Stub. A more robust way might be
            # to write the code to a file and use `modal run file.py::stub.function`,
            # but programmatic definition is attempted here.

            stub = modal.Stub(f"evogenesis-tool-{tool.id[:8]}")

            # Prepare image with dependencies
            requirements = tool.metadata.get("requirements", [])
            image = modal.Image.debian_slim().pip_install(requirements)
            # Define the remote function that will execute the tool code
            # Pass the code, function name, and args to it.
            @stub.function(
                image=image,
                timeout=int(timeout),  # Modal timeout is integer seconds
                memory=limits.get("memory_mb", 1024),  # Modal memory in MiB
                _allow_background_volume_commits=True  # May be needed depending on tool actions
            )
            def modal_runner(code_to_run: str, target_func_name: str, tool_args: dict):
                import json
                import traceback
                import requests
                from requests.exceptions import RequestException
                import urllib3

                # Execute the provided tool code in the function's scope
                exec_globals = {}
                try:
                    exec(code_to_run, exec_globals)
                except Exception as exec_err:
                    return {"error": f"Error executing tool code: {exec_err}", "traceback": traceback.format_exc()}

                # Find the target function within the executed code's globals
                target_function = exec_globals.get(target_func_name)

                if not target_function or not callable(target_function):
                    return {"error": f"Function '{target_func_name}' not found or not callable in provided code."}

                # Execute the function with provided arguments
                try:
                    result = target_function(**tool_args)
                    # Serialize result safely
                    try:
                        # Ensure JSON serializable for return
                        return {"output": json.loads(json.dumps(result))}
                    except (TypeError, OverflowError) as json_err:
                        return {"output": repr(result), "warning": f"Result not JSON serializable: {json_err}"}
                except Exception as run_err:
                    return {"error": str(run_err), "traceback": traceback.format_exc()}

            # Run the Modal function remotely using .call() for async context
            logging.debug(f"Calling Modal function for tool '{tool.name}'")
            # Note: Modal's .call() might block the asyncio event loop if not handled carefully
            # depending on the version and context. Consider running in executor if issues arise.
            result_data = await modal_runner.call(tool_code, function_name, args)
            logging.debug("Modal function execution finished.")

            execution_time = time.time() - start_time

            # Process the result dictionary returned by the Modal function
            if "error" in result_data:
                return ExecutionResult(
                    success=False,
                    error=result_data["error"],
                    logs=result_data.get("traceback", "No traceback available."),
                    execution_time=execution_time,
                    resource_usage={"platform": "modal"}  # Modal doesn't easily expose detailed usage
                )
            else:
                return ExecutionResult(
                    success=True,
                    output=result_data.get("output"),
                    logs=result_data.get("warning"),  # Include serialization warnings if any
                    execution_time=execution_time,
                    resource_usage={"platform": "modal"}
                )

        except Exception as e:
            # Catch potential errors during Modal setup or execution call
            logging.error(f"Error executing tool '{tool.name}' in Modal: {str(e)}")
            return ExecutionResult(
                success=False,
                error=str(e),
                logs=traceback.format_exc(),
                execution_time=time.time() - start_time
            )

# --- Remote Control Components ---

# Re-define RemoteTargetInfo if not using the mock from the top
# Ensure this definition matches the one potentially mocked earlier
if 'RemoteTargetInfo' not in globals() or not PERCEPTION_ACTION_AVAILABLE:
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
                "available_adapters": [adapter.value for adapter in self.available_adapters], # Use enum value
                "metadata": self.metadata,
                "last_discovery": self.last_discovery.isoformat()
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'RemoteTargetInfo':
            """Create from dictionary representation."""
            try:
                adapters = [RemoteAdapterType(adapter) for adapter in data.get("available_adapters", [])]
            except ValueError as e:
                 logging.warning(f"Error converting adapter string to enum in RemoteTargetInfo.from_dict: {e}. Skipping invalid adapters.")
                 adapters = [RemoteAdapterType(adapter) for adapter in data.get("available_adapters", []) if adapter in RemoteAdapterType.__members__]


            target = cls(
                host_id=data.get("host_id", ""),
                hostname=data.get("hostname", ""),
                ip_address=data.get("ip_address"),
                os_type=data.get("os_type"),
                available_adapters=adapters,
                metadata=data.get("metadata", {})
            )

            if "last_discovery" in data and data["last_discovery"]:
                try:
                    target.last_discovery = datetime.fromisoformat(data["last_discovery"])
                except (ValueError, TypeError):
                     logging.warning(f"Could not parse last_discovery timestamp: {data['last_discovery']}")
                     target.last_discovery = datetime.now() # Fallback

            return target


class RemoteDiscoveryService:
    """
    Service that discovers capabilities of remote targets asynchronously.

    Responsible for:
    - Probing remote machines for available APIs (Azure, potentially others)
    - Port scanning for remote control protocols (SSH, RDP, VNC)
    - Checking for cloud-specific capabilities (CloudPC, DevBox, AVD, Arc)
    - Optionally detecting OS type via ping TTL.
    - Optionally probing for OOB management (AMT placeholder).
    - Maintaining a cache of discovered target information.
    """

    def __init__(self, kernel):
        """
        Initialize the remote discovery service.

        Args:
            kernel: Reference to the EvoGenesis kernel
        """
        self.kernel = kernel
        self.config = kernel.config.get("remote_discovery", {})
        self.target_registry: Dict[str, RemoteTargetInfo] = {}  # host_id -> RemoteTargetInfo
        self.lock = asyncio.Lock() # Lock for registry access/modification

        # Init network scanning tools and cloud credentials
        self.scanner_initialized = False
        self.have_azure_auth = False
        self.azure_credential = None
        self.have_amt_support = False # Placeholder for AMT library check

        # Attempt to initialize Azure async credentials
        if self.config.get("enable_azure_probes", True): # Allow disabling Azure probes
            try:
                from azure.identity.aio import DefaultAzureCredential # Use async version
                # Consider excluding certain credential types if needed, e.g., for specific environments
                self.azure_credential = DefaultAzureCredential(
                    exclude_shared_token_cache_credential=self.config.get("azure_exclude_shared_token", True),
                    exclude_visual_studio_code_credential=self.config.get("azure_exclude_vscode", False),
                    exclude_environment_credential=self.config.get("azure_exclude_environment", False)
                    # Add other exclusions as needed
                )
                self.have_azure_auth = True
                logging.info("Azure async credentials initialized successfully.")
            except ImportError:
                logging.warning("Azure identity libraries not available (pip install azure-identity). Azure probes disabled.")
            except Exception as e:
                logging.warning(f"Failed to initialize Azure credentials: {e}. Azure probes disabled.")
        else:
            logging.info("Azure probes explicitly disabled in configuration.")


        # Placeholder for initializing AMT libraries if needed
        if self.config.get("enable_oob_scan", False):
            try:
                # import amt_library # Replace with actual library import
                # self.have_amt_support = True
                logging.info("OOB scan enabled, but AMT library check is currently a placeholder.")
                # For now, assume support is false until library is integrated
                self.have_amt_support = False
            except ImportError:
                logging.info("Intel AMT libraries not available. OOB scan for AMT disabled.")
                self.have_amt_support = False

        self.scanner_initialized = True # Mark as initialized

    async def discover_target(self, host_id: str, hostname: str,
                           ip_address: Optional[str] = None,
                           force_refresh: bool = False) -> RemoteTargetInfo:
        """
        Discover or retrieve cached capabilities of a remote target.

        Args:
            host_id: Unique identifier for the host.
            hostname: Hostname or FQDN.
            ip_address: IP address if known.
            force_refresh: If True, ignore cache and perform full discovery.

        Returns:
            RemoteTargetInfo with discovered capabilities.
        """
        cache_key = host_id
        cache_ttl = self.config.get("cache_ttl", 3600) # Default 1 hour cache

        # Check cache first (read-only access doesn't need lock yet)
        if not force_refresh and cache_key in self.target_registry:
            cached_target = self.target_registry[cache_key]
            if (datetime.now() - cached_target.last_discovery).total_seconds() < cache_ttl:
                logging.debug(f"Returning cached discovery info for {hostname} ({host_id})")
                return cached_target

        # Acquire lock for potential modification of the registry
        async with self.lock:
            # Double-check cache after acquiring lock to prevent race conditions
            if not force_refresh and cache_key in self.target_registry:
                 cached_target = self.target_registry[cache_key]
                 if (datetime.now() - cached_target.last_discovery).total_seconds() < cache_ttl:
                     logging.debug(f"Returning cached discovery info for {hostname} ({host_id}) (lock acquired)")
                     return cached_target

            # Proceed with discovery
            logging.info(f"Starting discovery for {hostname} ({host_id}). Force refresh: {force_refresh}")

            # Try to resolve IP if not provided and needed for scans
            resolved_ip = ip_address
            if not resolved_ip and (self.config.get("enable_port_scan", True) or self.config.get("enable_os_detect", False)):
                try:
                    # Use asyncio's getaddrinfo for non-blocking resolution
                    addr_info = await asyncio.get_event_loop().getaddrinfo(hostname, None, family=socket.AF_INET) # Prefer IPv4 for simplicity?
                    if addr_info:
                        resolved_ip = addr_info[0][4][0] # Get first address
                        logging.info(f"Resolved {hostname} to {resolved_ip}")
                except socket.gaierror:
                    logging.warning(f"Could not resolve hostname {hostname} to IP address.")
                except Exception as e:
                     logging.warning(f"Error resolving hostname {hostname}: {e}")

            # Create a new target info object or update existing one if force_refresh
            target = RemoteTargetInfo(
                host_id=host_id,
                hostname=hostname,
                ip_address=resolved_ip # Use resolved IP
            )

            # Run discovery probes concurrently
            discovery_tasks = []
            probe_results = {}

            # Define probes to run based on config
            probes_to_run = {
                "os_detect": self.config.get("enable_os_detect", False),
                "cloud_apis": self.config.get("enable_azure_probes", True) and self.have_azure_auth,
                "protocols": self.config.get("enable_port_scan", True),
                "oob": self.config.get("enable_oob_scan", False)
            }

            if probes_to_run["os_detect"]:
                 discovery_tasks.append(asyncio.create_task(self._detect_os_type(target), name="os_detect"))
            if probes_to_run["cloud_apis"]:
                 discovery_tasks.append(asyncio.create_task(self._check_cloud_apis(target), name="cloud_apis"))
            if probes_to_run["protocols"]:
                 discovery_tasks.append(asyncio.create_task(self._scan_protocols(target), name="protocols"))
            if probes_to_run["oob"]:
                 discovery_tasks.append(asyncio.create_task(self._check_oob_management(target), name="oob"))

            # Execute all enabled probes
            if discovery_tasks:
                 completed, pending = await asyncio.wait(discovery_tasks, return_when=asyncio.ALL_COMPLETED)
                 for task in completed:
                     try:
                         result = task.result()
                         probe_name = task.get_name()
                         probe_results[probe_name] = result
                     except Exception as e:
                         probe_name = task.get_name()
                         logging.error(f"Discovery probe '{probe_name}' failed for {hostname}: {e}")
                         probe_results[probe_name] = None # Indicate failure
            else:
                 logging.info(f"No discovery probes enabled for {hostname}.")


            # Process results and update target info
            target.os_type = probe_results.get("os_detect") # Will be None if failed or disabled

            cloud_adapters = probe_results.get("cloud_apis", []) or []
            protocol_adapters = probe_results.get("protocols", []) or []
            oob_adapters = probe_results.get("oob", []) or []

            available_adapters = list(set(cloud_adapters + protocol_adapters + oob_adapters))

            # Add vision fallback if no other GUI methods are found and vision is enabled
            if self.config.get("enable_vision_fallback", True):
                gui_adapters = {RemoteAdapterType.RDP, RemoteAdapterType.VNC, RemoteAdapterType.AMT_KVM}
                has_gui_adapter = any(adapter in gui_adapters for adapter in available_adapters)
                if not has_gui_adapter and RemoteAdapterType.VISION_FALLBACK not in available_adapters:
                    available_adapters.append(RemoteAdapterType.VISION_FALLBACK)
                    logging.debug(f"Adding VISION_FALLBACK adapter for {hostname}")

            # Update target object
            target.available_adapters = sorted([adapter for adapter in available_adapters], key=lambda x: x.value) # Sort for consistency
            target.last_discovery = datetime.now()
            target.metadata["discovery_timestamp"] = target.last_discovery.isoformat()
            target.metadata["discovery_source"] = "auto"

            # Store in registry
            self.target_registry[cache_key] = target

            logging.info(f"Discovery complete for {hostname}. Adapters: {[a.value for a in target.available_adapters]}")
            return target

    async def _detect_os_type(self, target: RemoteTargetInfo) -> Optional[str]:
        """Detect OS type using TTL in ping response (best effort)."""
        if not target.ip_address:
            logging.debug(f"OS detection skipped for {target.hostname}: No IP address.")
            return None

        ping_timeout = self.config.get("ping_timeout", 1.0)
        logging.debug(f"Attempting OS detection via ping TTL for {target.hostname} ({target.ip_address})")

        try:
            if sys.platform == "win32":
                ping_cmd = ["ping", "-n", "1", f"-w", str(int(ping_timeout * 1000)), target.ip_address]
            else:
                ping_cmd = ["ping", "-c", "1", "-W", str(ping_timeout), target.ip_address]

            proc = await asyncio.create_subprocess_exec(
                *ping_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            # Wait with timeout slightly longer than ping timeout
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=ping_timeout + 0.5)

            if proc.returncode != 0:
                 logging.debug(f"Ping failed for OS detection on {target.hostname}: {stderr.decode(errors='ignore').strip()}")
                 return None

            stdout_str = stdout.decode(errors='ignore')
            ttl_match = re.search(r"[Tt][Tt][Ll]=(\d+)", stdout_str)
            if ttl_match:
                ttl = int(ttl_match.group(1))
                if 100 < ttl <= 128:
                    os_guess = "Windows"
                elif 50 < ttl <= 64:
                    os_guess = "Linux" # Group Unix-like
                else:
                    os_guess = None # Uncommon TTL
                logging.debug(f"OS detected for {target.hostname} as {os_guess} (TTL={ttl})")
                return os_guess
            else:
                logging.debug(f"Could not parse TTL from ping response for {target.hostname}")
                return None

        except asyncio.TimeoutError:
            logging.debug(f"OS detection ping timed out for {target.hostname}")
            return None
        except FileNotFoundError:
             logging.error("OS detection failed: 'ping' command not found.")
             return None
        except Exception as e:
            logging.warning(f"OS detection via ping failed for {target.hostname}: {str(e)}")
            return None

    async def _check_cloud_apis(self, target: RemoteTargetInfo) -> List[RemoteAdapterType]:
        """Check for Azure cloud-specific APIs concurrently."""
        if not self.have_azure_auth:
            logging.debug(f"Skipping Azure cloud API checks for {target.hostname}: Azure auth not available.")
            return []

        logging.debug(f"Checking Azure cloud APIs for {target.hostname}")
        adapters = []
        probes = [
            asyncio.create_task(self._probe_graph_cloudpc(target), name="probe_cloudpc"),
            asyncio.create_task(self._probe_dev_box(target), name="probe_devbox"),
            asyncio.create_task(self._probe_avd(target), name="probe_avd"),
            asyncio.create_task(self._probe_azure_arc(target), name="probe_arc")
        ]

        completed, pending = await asyncio.wait(probes, return_when=asyncio.ALL_COMPLETED)

        probe_map = {
            "probe_cloudpc": RemoteAdapterType.GRAPH_CLOUDPC,
            "probe_devbox": RemoteAdapterType.DEV_BOX,
            "probe_avd": RemoteAdapterType.AVD_REST,
            "probe_arc": RemoteAdapterType.ARC_COMMAND
        }

        for task in completed:
            probe_name = task.get_name()
            adapter_type = probe_map[probe_name]
            try:
                result = task.result()
                if result is True:
                    adapters.append(adapter_type)
                    logging.info(f"Detected Azure capability '{adapter_type.value}' for {target.hostname}")
            except Exception as e:
                logging.warning(f"Azure probe '{probe_name}' failed for {target.hostname}: {e}")

        return adapters

    async def _scan_protocols(self, target: RemoteTargetInfo) -> List[RemoteAdapterType]:
        """Scan for common remote control protocol ports."""
        if not self.scanner_initialized or not target.ip_address:
            logging.debug(f"Protocol scan skipped for {target.hostname}: No IP or scanner not ready.")
            return []

        logging.debug(f"Scanning protocol ports for {target.hostname} ({target.ip_address})")
        adapters = set() # Use set to avoid duplicates
        ports_to_check = {
            22:   RemoteAdapterType.SSH,
            3389: RemoteAdapterType.RDP,
              5900: RemoteAdapterType.VNC, # Default VNC display 0
            5901: RemoteAdapterType.VNC, # VNC display 1
        }
        scan_timeout = self.config.get("port_scan_timeout", 1.0) # Defaultscan_timeout = self.config.get("port_scan_timeout", 1.0) # Default 1 second timeout
        scan_tasks = [
            asyncio.create_task(self._scan_port(target, port, scan_timeout), name=f"scan_{port}")
            for port in ports_to_check.keys()
        ]
        if not scan_tasks:
            return []

        completed, pending = await asyncio.wait(scan_tasks, return_when=asyncio.ALL_COMPLETED)

        for task in completed:
            port = int(task.get_name().split('_')[1])
            adapter_type = ports_to_check[port]
            try:
                is_open = task.result()
                if is_open:
                    adapters.add(adapter_type)
                    logging.info(f"Port {port} open on {target.hostname}, adding adapter {adapter_type.value}")
            except Exception as e:
                 logging.warning(f"Port scan task for {target.hostname}:{port} failed: {e}")

        return list(adapters)

    async def _check_oob_management(self, target: RemoteTargetInfo) -> List[RemoteAdapterType]:
        """Check for out-of-band management capabilities (e.g., Intel AMT)."""
        adapters = []
        if not self.have_amt_support: # Check if AMT support is enabled/available
             logging.debug(f"OOB scan skipped for {target.hostname}: AMT support not available/enabled.")
             return adapters

        logging.debug(f"Checking OOB management (AMT) for {target.hostname}")
        try:
            if await self._probe_amt(target):
                adapters.append(RemoteAdapterType.AMT_KVM)
                logging.info(f"Intel AMT detected on {target.hostname}")
        except Exception as e:
             logging.warning(f"OOB check for {target.hostname} failed: {e}")

        # Add checks for other OOB methods like IPMI if needed

        return adapters

    async def _get_azure_token(self, scope: str) -> Optional[str]:
        """Helper to get Azure token asynchronously."""
        if not self.azure_credential:
            logging.warning(f"Cannot get Azure token for scope {scope}: credential object not available.")
            return None
        try:
            logging.debug(f"Requesting Azure token for scope: {scope}")
            token_credential = await self.azure_credential.get_token(scope)
            logging.debug(f"Successfully obtained Azure token for scope: {scope}")
            return token_credential.token
        except Exception as e:
            # Log specific credential errors if possible (e.g., AuthenticationFailedError)
            logging.warning(f"Failed to get Azure token for scope {scope}: {type(e).__name__} - {e}")
            return None

    async def _make_azure_api_request(self, url: str, scope: str) -> Optional[dict]:
         """Helper to make authenticated Azure API requests asynchronously."""
         token = await self._get_azure_token(scope)
         if not token:
             return None # Error already logged by _get_azure_token

         headers = {
             "Authorization": f"Bearer {token}",
             "Content-Type": "application/json",
             "Accept": "application/json"
         }
         request_timeout = self.config.get("azure_api_timeout", 10.0)

         try:
             # Using requests in a thread pool executor for simplicity, as in original code.
             # For high-concurrency scenarios, consider switching to httpx or aiohttp.
             logging.debug(f"Making Azure API request to: {url}")
             response = await asyncio.to_thread(
                 requests.get, url, headers=headers, timeout=request_timeout
             )
             logging.debug(f"Azure API request to {url} returned status: {response.status_code}")
             response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
             return response.json()
         except requests.exceptions.Timeout:
             logging.warning(f"Azure API request timed out for {url}")
             return None
         except requests.exceptions.HTTPError as e:
             # Log specific HTTP errors (e.g., 401 Unauthorized, 403 Forbidden, 404 Not Found)
             logging.warning(f"Azure API HTTP error for {url}: {e.response.status_code} - {e.response.text}")
             return None
         except requests.exceptions.RequestException as e:
             logging.warning(f"Azure API request failed for {url}: {e}")
             return None
         except Exception as e:
              logging.error(f"Unexpected error during Azure API request to {url}: {e}", exc_info=True)
              return None

    async def _probe_graph_cloudpc(self, target: RemoteTargetInfo) -> bool:
        """Probe Microsoft Graph API for Cloud PC matching hostname."""
        logging.debug(f"Probing Graph API for Cloud PC: {target.hostname}")
        scope = "https://graph.microsoft.com/.default"
        # Filter directly in the query for efficiency if hostname is reliable
        # Note: managedDeviceName might not exactly match hostname in all cases.
        # Using startswith might be safer if domain suffix varies.
        # $filter=startswith(managedDeviceName,'{target.hostname}')
        # For exact match (case-insensitive recommended by Graph): $filter=managedDeviceName eq '{target.hostname}'
        # Let's try exact match first.
        hostname_lower = target.hostname.lower()
        url = f"https://graph.microsoft.com/v1.0/me/cloudPCs?$filter=managedDeviceName eq '{hostname_lower}'&$select=id,managedDeviceName,status"

        data = await self._make_azure_api_request(url, scope)
        # Check if request succeeded and returned at least one value
        if data and isinstance(data.get("value"), list) and data["value"]:
            pc = data["value"][0] # Get the first match
            target.metadata["cloudpc_id"] = pc.get("id")
            target.metadata["cloudpc_status"] = pc.get("status")
            logging.debug(f"Matched {target.hostname} to CloudPC ID {pc.get('id')}")
            return True
        else:
            logging.debug(f"No Cloud PC found matching hostname {target.hostname} via Graph API.")
            return False

    async def _probe_dev_box(self, target: RemoteTargetInfo) -> bool:
        """Probe Azure Dev Box API for a Dev Box matching hostname."""
        logging.debug(f"Probing Azure Dev Box API for: {target.hostname}")
        subscription_id = self.config.get("azure", {}).get("subscription_id")
        if not subscription_id:
            logging.debug("Dev Box probe skipped: Azure subscription ID not configured.")
            return False

        scope = "https://management.azure.com/.default"
        # Listing all dev boxes can be slow. Filtering is preferred.
        # The Dev Box name often matches the hostname.
        hostname_lower = target.hostname.lower()
        # This API might require iterating through projects if not filterable by name directly at subscription level.
        # Let's assume a filter works for this example. Check DevCenter REST API docs for actual filter capabilities.
        # url = (f"https://management.azure.com/subscriptions/{subscription_id}/"
        #        f"providers/Microsoft.DevCenter/devboxes?$filter=name eq '{hostname_lower}'&api-version=2023-04-01")
        # If filtering isn't supported, list all and filter client-side (less efficient).
        url = (f"https://management.azure.com/subscriptions/{subscription_id}/"
               f"providers/Microsoft.DevCenter/devboxes?api-version=2023-04-01")


        data = await self._make_azure_api_request(url, scope)
        if not data or "value" not in data:
            logging.debug(f"Failed to retrieve Dev Boxes or no Dev Boxes found for subscription.")
            return False

        dev_boxes = data["value"]
        for box in dev_boxes:
            # Match name case-insensitively
            if box.get("name", "").lower() == hostname_lower:
                props = box.get("properties", {})
                target.metadata["devbox_id"] = box.get("id")
                target.metadata["devbox_project"] = props.get("projectName")
                target.metadata["devbox_pool"] = props.get("poolName")
                target.metadata["devbox_status"] = props.get("powerState") # Or provisioningState
                target.metadata["devbox_os"] = props.get("osType")
                logging.debug(f"Matched {target.hostname} to Dev Box ID {box.get('id')}")
                return True
        logging.debug(f"No Dev Box found matching hostname {target.hostname}.")
        return False

    async def _probe_avd(self, target: RemoteTargetInfo) -> bool:
        """Probe Azure Virtual Desktop for a Session Host matching hostname."""
        logging.debug(f"Probing AVD API for: {target.hostname}")
        subscription_id = self.config.get("azure", {}).get("subscription_id")
        # Querying requires Resource Group and Host Pool name for efficiency.
        resource_group = self.config.get("azure", {}).get("avd_resource_group")
        host_pool = self.config.get("azure", {}).get("avd_host_pool")

        if not subscription_id or not resource_group or not host_pool:
            logging.debug("AVD probe skipped: Azure subscription, resource group, or host pool not configured.")
            # Alternative: List all host pools, then query each (much slower).
            return False

        scope = "https://management.azure.com/.default"
        # AVD Session Host name format is usually "hostname.domain/resource_id"
        # We need to list hosts in the pool and check if any start with the target hostname.
        url = (f"https://management.azure.com/subscriptions/{subscription_id}/"
               f"resourceGroups/{resource_group}/providers/Microsoft.DesktopVirtualization/"
               f"hostPools/{host_pool}/sessionHosts?api-version=2022-09-09") # Use a stable API version

        data = await self._make_azure_api_request(url, scope)
        if not data or "value" not in data:
            logging.debug(f"Failed to retrieve AVD Session Hosts or none found in pool {host_pool}.")
            return False

        session_hosts = data["value"]
        target_hostname_lower = target.hostname.lower()
        for host in session_hosts:
            session_host_name = host.get("name", "").lower()
            # Check if the session host name starts with the target hostname followed by a dot or slash
            if session_host_name.startswith(target_hostname_lower + '.') or \
               session_host_name.startswith(target_hostname_lower + '/'):
                props = host.get("properties", {})
                target.metadata["avd_session_host_id"] = host.get("id")
                target.metadata["avd_status"] = props.get("status")
                target.metadata["avd_agent_version"] = props.get("agentVersion")
                target.metadata["avd_os"] = props.get("osVersion")
                logging.debug(f"Matched {target.hostname} to AVD Session Host ID {host.get('id')}")
                return True
        logging.debug(f"No AVD Session Host found matching hostname {target.hostname} in pool {host_pool}.")
        return False

    async def _probe_azure_arc(self, target: RemoteTargetInfo) -> bool:
        """Probe Azure Arc for an enabled server matching hostname."""
        logging.debug(f"Probing Azure Arc API for: {target.hostname}")
        subscription_id = self.config.get("azure", {}).get("subscription_id")
        if not subscription_id:
            logging.debug("Azure Arc probe skipped: Azure subscription ID not configured.")
            return False

        scope = "https://management.azure.com/.default"
        # Filtering by name is efficient: $filter=properties/osName ne null and name eq '{target.hostname}'
        # Using 'osName ne null' can help filter out machines still provisioning.
        hostname_lower = target.hostname.lower()
        url = (f"https://management.azure.com/subscriptions/{subscription_id}/"
               f"providers/Microsoft.HybridCompute/machines?"
               f"$filter=name eq '{hostname_lower}'&api-version=2023-06-20-preview")

        data = await self._make_azure_api_request(url, scope)
        if data and isinstance(data.get("value"), list) and data["value"]:
            server = data["value"][0] # Get the first match
            props = server.get("properties", {})
            target.metadata["arc_machine_id"] = server.get("id")
            target.metadata["arc_status"] = props.get("status")
            target.metadata["arc_os_type"] = props.get("osName")
            target.metadata["arc_os_version"] = props.get("osVersion")
            target.metadata["arc_agent_version"] = props.get("agentVersion")
            logging.debug(f"Matched {target.hostname} to Arc Machine ID {server.get('id')}")
            return True
        else:
            logging.debug(f"No Azure Arc machine found matching hostname {target.hostname}.")
            return False

    async def _scan_port(self, target: RemoteTargetInfo, port: int, timeout: float = 1.0) -> bool:
        """
        Asynchronously scan for an open TCP port.

        Args:
            target: Target information.
            port: Port number to scan.
            timeout: Connection timeout in seconds.

        Returns:
            True if port is open, False otherwise.
        """
        if not target.ip_address:
            logging.debug(f"Port scan skipped for {target.hostname}:{port}: No IP address.")
            return False

        writer = None # Ensure writer is defined for finally block
        logging.debug(f"Scanning port {port} on {target.hostname} ({target.ip_address})")
        try:
            # Use asyncio.open_connection with a timeout
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target.ip_address, port),
                timeout=timeout
            )
            # Connection successful
            logging.debug(f"Port scan: {target.ip_address}:{port} is open.")
            return True
        except asyncio.TimeoutError:
            logging.debug(f"Port scan: {target.ip_address}:{port} timed out after {timeout}s.")
            return False
        except ConnectionRefusedError:
            logging.debug(f"Port scan: {target.ip_address}:{port} refused connection.")
            return False
        except OSError as e:
            # Handle specific OS errors if needed (e.g., Network unreachable, Host unreachable)
            logging.debug(f"Port scan OS error for {target.ip_address}:{port}: {e.strerror} (errno {e.errno})")
            return False
        except Exception as e:
            # Catch any other unexpected errors during connection attempt
            logging.warning(f"Unexpected port scan error for {target.ip_address}:{port}: {type(e).__name__} - {e}")
            return False
        finally:
            # Ensure the writer (connection) is closed if it was successfully created
            if writer:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception as close_err:
                    logging.debug(f"Error closing socket during port scan cleanup: {close_err}")

    async def _probe_amt(self, target: RemoteTargetInfo) -> bool:
        """
        Probe for Intel AMT availability using both port scanning and WSMAN protocol validation.
        
        Args:
            target: Target information including hostname and IP address.
            
        Returns:
            True if Intel AMT is confirmed available, False otherwise.
        """
        if not self.have_amt_support or not target.ip_address:
            if self.config.get("enable_oob_scan", False):
                logging.debug(f"AMT probe skipped for {target.hostname}: Support not available/enabled or no IP.")
            return False

        logging.debug(f"Probing for Intel AMT on {target.hostname} ({target.ip_address})")
        
        # Common AMT ports (16992=HTTP, 16993=HTTPS)
        amt_ports = [16992, 16993]
        scan_timeout = self.config.get("amt_scan_timeout", 1.5)
        
        # Step 1: Check if AMT ports are open
        try:
            port_scan_tasks = [
                asyncio.create_task(self._scan_port(target, port, scan_timeout), name=f"amt_scan_{port}")
                for port in amt_ports
            ]
            completed, pending = await asyncio.wait(port_scan_tasks, return_when=asyncio.ALL_COMPLETED)
            
            open_amt_ports = []
            for task in completed:
                port = int(task.get_name().split('_')[-1])
                try:
                    if task.result() is True:
                        open_amt_ports.append(port)
                except Exception as e:
                    logging.debug(f"Error checking AMT port {port}: {e}")
            
            if not open_amt_ports:
                logging.debug(f"AMT probe: No standard AMT ports ({', '.join(map(str, amt_ports))}) open on {target.hostname}")
                return False
                
            # Store detected ports in metadata
            detected_port = min(open_amt_ports)  # Prefer standard port if both open
            target.metadata["amt_port_detected"] = detected_port
            is_tls = detected_port == 16993
            protocol = "https" if is_tls else "http"
            
            # Step 2: Verify AMT service by checking WSMAN endpoint
            try:
                
                # Suppress insecure HTTPS warnings for self-signed certs
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
                # Basic WSMAN identification request
                headers = {
                    "Content-Type": "application/soap+xml;charset=UTF-8",
                    "User-Agent": "AMT-Discovery-Agent/1.0"
                }
                
                url = f"{protocol}://{target.ip_address}:{detected_port}/wsman"
                
                # Execute request in a thread to avoid blocking
                response = await asyncio.to_thread(
                    lambda: requests.get(url, headers=headers, timeout=scan_timeout, verify=False)
                )
                
                # Check for AMT-specific response headers or content
                amt_confirmed = False
                if response.status_code == 200 or response.status_code == 401:  # 401 means auth required, which is good
                    amt_confirmed = True
                    server_header = response.headers.get("Server", "").lower()
                    if "intel" in server_header or "amt" in server_header:
                        target.metadata["amt_server_header"] = response.headers.get("Server")
                    
                if amt_confirmed:
                    logging.info(f"AMT confirmed available on {target.hostname} via {protocol}://{target.ip_address}:{detected_port}")
                    target.metadata["amt_protocol"] = protocol
                    target.metadata["amt_port"] = detected_port
                    return True
                else:
                    logging.debug(f"AMT ports open on {target.hostname} but service not confirmed")
                    return False
                    
            except ImportError:
                logging.warning("Requests library not installed. Cannot fully verify AMT service.")
                # Fall back to port-only detection if we can't verify via WSMAN
                logging.info(f"AMT possibly available on {target.hostname} (port {detected_port} open, but service unverified)")
                target.metadata["amt_status"] = "possible_unverified"
                return True  # Return True with warning in production to allow potential use
                
            except RequestException as e:
                logging.debug(f"AMT service verification failed on {target.hostname}: {e}")
                # If port is open but request fails, it might still be AMT with different config
                target.metadata["amt_status"] = "port_open_request_failed"
                return False
                
        except Exception as e:
            logging.warning(f"Intel AMT probe failed for {target.hostname}: {str(e)}")
            return False


class RemoteAdapterDecisionEngine:
    """
    Decision engine that selects the best adapter for a remote control task.

    Uses a weighted scoring system based on configurable criteria like:
    - Security, Reliability, Latency, Feature Completeness, Cost.
    Scores and weights are configurable via the application configuration.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the decision engine.

        Args:
            config: Configuration dictionary for weights and adapter scores.
        """
        self.config = config or {}
        logging.debug(f"Initializing RemoteAdapterDecisionEngine with config: {self.config}")

        # Default weights (normalized sum to 1.0)
        self.default_weights = {
            "security": 0.35,
            "reliability": 0.20,
            "latency": 0.15, # Lower latency preferred
            "feature_completeness": 0.20,
            "cost": 0.10 # Lower cost preferred
        }
        self.weights = self.config.get("weights", self.default_weights)
        self._normalize_weights(self.weights) # Ensure weights sum to 1

        # Default adapter scores (0.0 to 1.0 scale)
        # Higher is better for security, reliability, features.
        # Lower is better for latency, cost (will be inverted in calculation).
        self.default_adapter_scores = {
            RemoteAdapterType.GRAPH_CLOUDPC: {"security": 0.9, "reliability": 0.8, "latency": 0.6, "feature_completeness": 0.7, "cost": 0.4},
            RemoteAdapterType.DEV_BOX:       {"security": 0.9, "reliability": 0.8, "latency": 0.6, "feature_completeness": 0.7, "cost": 0.4},
            RemoteAdapterType.AVD_REST:      {"security": 0.8, "reliability": 0.7, "latency": 0.7, "feature_completeness": 0.6, "cost": 0.5},
            RemoteAdapterType.ARC_COMMAND:   {"security": 0.8, "reliability": 0.7, "latency": 0.8, "feature_completeness": 0.5, "cost": 0.5},
            RemoteAdapterType.SSH:           {"security": 0.8, "reliability": 0.9, "latency": 0.2, "feature_completeness": 0.8, "cost": 0.1},
            RemoteAdapterType.RDP:           {"security": 0.7, "reliability": 0.7, "latency": 0.4, "feature_completeness": 0.9, "cost": 0.3},
            RemoteAdapterType.VNC:           {"security": 0.5, "reliability": 0.6, "latency": 0.5, "feature_completeness": 0.8, "cost": 0.1},
            RemoteAdapterType.AMT_KVM:       {"security": 0.6, "reliability": 0.5, "latency": 0.7, "feature_completeness": 0.4, "cost": 0.6}, # Hardware cost factored in
            RemoteAdapterType.VISION_FALLBACK: {"security": 0.4, "reliability": 0.3, "latency": 0.9, "feature_completeness": 0.6, "cost": 0.8} # LLM/API cost
        }
        self.adapter_scores = self.config.get("adapter_scores", {})
        # Merge defaults with config (config overrides defaults)
        merged_scores = self.default_adapter_scores.copy()
        merged_scores.update(self.adapter_scores)
        self.adapter_scores = merged_scores


        # Operation type adjustments (multipliers for weights)
        self.default_operation_adjustments = {
            "power": {"security": 1.5, "reliability": 1.2, "latency": 0.5},
            "script": {"feature_completeness": 1.3, "latency": 0.8, "reliability": 1.1},
            "gui": {"feature_completeness": 1.5, "latency": 0.7}, # Lower latency still preferred, but features more important
            "file_transfer": {"reliability": 1.4, "latency": 0.9, "cost": 0.8}
        }
        self.operation_adjustments = self.config.get("operation_adjustments", self.default_operation_adjustments)

        logging.debug(f"Decision Engine Weights: {self.weights}")
        logging.debug(f"Decision Engine Adapter Scores: {self.adapter_scores}")
        logging.debug(f"Decision Engine Operation Adjustments: {self.operation_adjustments}")


    def _normalize_weights(self, weights: Dict[str, float]):
        """Normalizes a dictionary of weights to sum to 1.0."""
        total_weight = sum(weights.values())
        if total_weight > 0 and total_weight != 1.0:
            logging.debug(f"Normalizing weights from total {total_weight:.3f} to 1.0")
            for key in weights:
                weights[key] /= total_weight
        elif total_weight == 0:
             logging.warning("All decision weights are zero. Scoring will be ineffective.")


    def select_adapter(self, available_adapters: List[RemoteAdapterType],
                      operation_type: str = "general",
                      target_info: Optional[RemoteTargetInfo] = None) -> Optional[RemoteAdapterType]:
        """
        Select the best adapter for an operation based on scoring.

        Args:
            available_adapters: List of available adapter types discovered for the target.
            operation_type: Type of operation (e.g., 'power', 'script', 'gui', 'file_transfer').
            target_info: Optional RemoteTargetInfo for context-aware scoring (e.g., OS type).

        Returns:
            The selected RemoteAdapterType, or None if no suitable adapter is found.
        """
        if not available_adapters:
            logging.warning("Cannot select adapter: No available adapters provided.")
            return None

        logging.info(f"Selecting adapter for operation '{operation_type}' from available: {[a.value for a in available_adapters]}")

        # --- Filter adapters based on suitability (optional but recommended) ---
        # Example: Vision fallback might not work for power operations
        # Example: SSH might not work for GUI operations (unless tunneling VNC/RDP)
        # This requires defining adapter capabilities more explicitly.
        # suitable_adapters = self._filter_suitable_adapters(available_adapters, operation_type, target_info)
        suitable_adapters = available_adapters # Use all available for now

        if not suitable_adapters:
             logging.warning(f"No suitable adapters found for operation '{operation_type}' among available adapters.")
             return None


        # --- Adjust weights based on operation type ---
        current_weights = self.weights.copy()
        adjustments = self.operation_adjustments.get(operation_type, {})
        if adjustments:
             logging.debug(f"Applying weight adjustments for operation type '{operation_type}': {adjustments}")
             for key, multiplier in adjustments.items():
                 if key in current_weights:
                     current_weights[key] *= multiplier
             # Re-normalize weights after adjustment
             self._normalize_weights(current_weights)
             logging.debug(f"Adjusted weights for '{operation_type}': {current_weights}")


        # --- Calculate score for each suitable adapter ---
        scores = {}
        for adapter in suitable_adapters:
            if adapter not in self.adapter_scores:
                logging.warning(f"No scores defined for adapter: {adapter.value}. Assigning score 0.")
                scores[adapter] = 0.0
                continue

            adapter_score_config = self.adapter_scores[adapter]
            final_score = 0.0

            # Calculate weighted score based on adjusted weights
            for criterion, weight in current_weights.items():
                # Get the base score for the criterion, default to 0.5 if missing? Or 0? Let's use 0.
                score_value = adapter_score_config.get(criterion, 0.0)

                # Invert score contribution for criteria where lower is better (cost, latency)
                if criterion in ["cost", "latency"]:
                    # Score contribution = (1 - base_score) * weight
                    score_contribution = (1.0 - score_value) * weight
                else: # Higher is better (security, reliability, feature_completeness)
                    # Score contribution = base_score * weight
                    score_contribution = score_value * weight

                final_score += score_contribution

            # --- Contextual Adjustments (Optional) ---
            # Example: Boost SSH score slightly if target OS is Linux
            # if target_info and target_info.os_type == "Linux" and adapter == RemoteAdapterType.SSH:
            #     final_score *= 1.1 # Apply a small boost

            scores[adapter] = final_score
            logging.debug(f"Adapter {adapter.value} score for '{operation_type}': {final_score:.3f}")


        # --- Select the adapter with the highest score ---
        if not scores:
             logging.error(f"No suitable adapters could be scored for operation '{operation_type}'.")
             return None

        # Find the adapter with the maximum score
        selected_adapter = max(scores.items(), key=lambda item: item[1])[0]

        logging.info(f"Selected adapter: {selected_adapter.value} for operation '{operation_type}' with score {scores[selected_adapter]:.3f}")
        return selected_adapter


class RemoteControlToolGenerator:
    """
    Generates tools for remote machine control using various adapters.

    Uses LLMs to generate adapter-specific code based on natural language descriptions.
    Relies on RemoteDiscoveryService and RemoteAdapterDecisionEngine if the main
    Perception-Action module is not available or configured.
    """

    def __init__(self, kernel):
        """
        Initialize the remote control tool generator.

        Args:
            kernel: Reference to the EvoGenesis kernel
        """
        self.kernel = kernel
        self.llm_orchestrator = kernel.llm_orchestrator
        self.remote_control_module = None # Default to internal handling
        self.discovery_service = None
        self.decision_engine = None

        # Check if the PerceptionAction module is available and integrated
        if PERCEPTION_ACTION_AVAILABLE and hasattr(kernel, 'perception_action_module') and kernel.perception_action_module:
             # Ensure it's not just the mock being used internally
             # This check might need refinement based on how the mock vs real module is determined
             is_mock_module = isinstance(kernel.perception_action_module, globals().get('RemoteControlModule')) and \
                              "mock implementation" in logging.getLogger().handlers[0].stream.getvalue() # Heuristic check

             if not is_mock_module:
                 self.remote_control_module = kernel.perception_action_module
                 logging.info("Using integrated Perception-Action module for remote control tool generation.")
             else:
                  logging.info("Perception-Action module seems to be the mock version, using internal remote components.")
        else:
             logging.info("Perception-Action module not available or not integrated, using internal remote components.")

        # Initialize internal components if PerceptionAction module is not used
        if not self.remote_control_module:
             self.discovery_service = RemoteDiscoveryService(kernel)
             self.decision_engine = RemoteAdapterDecisionEngine(
                 kernel.config.get("remote_adapter_decision", {})
             )

        # Adapter-specific generation prompts/templates
        # These should be detailed enough for the LLM to generate useful code.
        # Include hints about required libraries and authentication.
        self.adapter_templates = {
            RemoteAdapterType.GRAPH_CLOUDPC: """
Generate a Python function that uses the msgraph-sdk library to {description} for the Cloud PC identified by host ID '{host_id}' (hostname: {hostname}).
The function should accept parameters: {parameters_json} and return: {returns_json}.
Assume an authenticated GraphServiceClient ('graph_client') is available or passed as an argument. Include necessary imports and error handling.
Example task: Start, Stop, Reboot Cloud PC, Get Cloud PC details.
""",
            RemoteAdapterType.DEV_BOX: """
Generate a Python function using the Azure SDK for Python (azure-mgmt-devcenter) or direct REST calls to {description} for the Dev Box identified by host ID '{host_id}' (hostname: {hostname}).
The function should accept parameters: {parameters_json} and return: {returns_json}.
Assume necessary Azure credentials (e.g., DefaultAzureCredential) are configured and a DevCenterClient is instantiated. Include necessary imports and error handling.
Example task: Start, Stop Dev Box, Get Dev Box connection URL.
""",
            RemoteAdapterType.AVD_REST: """
Generate a Python function using the Azure SDK for Python (azure-mgmt-desktopvirtualization) or direct REST calls to {description} for the AVD session host identified by host ID '{host_id}' (hostname: {hostname}).
The function should accept parameters: {parameters_json} and return: {returns_json}.
Assume necessary Azure credentials are configured and a DesktopVirtualizationMgmtClient is instantiated. Include necessary imports and error handling.
Example task: Log off user session, Send message to session host, Set drain mode.
""",
            RemoteAdapterType.ARC_COMMAND: """
Generate a Python function using the Azure SDK for Python (azure-mgmt-hybridcompute RunCommands) or direct REST calls to execute a script via Azure Arc Run Command.
The goal is to {description} on the Arc-enabled server '{hostname}' (host ID: {host_id}).
The function should accept parameters: {parameters_json} and return: {returns_json}.
The function should construct the appropriate script (e.g., PowerShell for Windows, bash for Linux) based on the description and parameters, then invoke the Run Command API.
Assume necessary Azure credentials are configured and a HybridComputeManagementClient is instantiated. Handle potential script execution errors and timeouts.
""",
            RemoteAdapterType.SSH: """
Generate a Python function using the 'paramiko' library to {description} via SSH on host '{hostname}' (IP: {ip_address}).
The function should accept parameters: {parameters_json} (likely including SSH credentials like username, password/key). It should return: {returns_json}.
Ensure the function establishes an SSH connection, executes the necessary commands or SFTP operations, handles potential errors (connection, authentication, execution), and closes the connection properly. Include necessary imports.
""",
            RemoteAdapterType.RDP: """
Generate a Python script using libraries like 'pyautogui' and potentially 'pywinauto' (if target is Windows) to perform the following GUI automation task: {description}.
The script assumes an RDP connection to '{hostname}' is already established and visible.
The function should accept parameters: {parameters_json} and return: {returns_json} (e.g., success status, extracted text).
Focus on visual automation steps (finding elements by image/coordinates, clicking, typing). Include necessary imports and basic error handling (e.g., element not found).
NOTE: This is fragile and requires careful calibration.
""",
            RemoteAdapterType.VNC: """
Generate a Python script using a library like 'vncdotool' or 'pyautogui' to perform the following GUI automation task via VNC: {description}.
The script should connect to the VNC server on '{hostname}' (IP: {ip_address}).
The function should accept parameters: {parameters_json} (likely including VNC password). It should return: {returns_json}.
Implement the VNC connection, GUI interaction steps (capture screen, find elements, keyboard/mouse actions), error handling, and disconnection. Include necessary imports.
""",
            RemoteAdapterType.AMT_KVM: """
Generate a Python function using a relevant Intel AMT library (e.g., 'python-amt', if suitable and installed) to {description} on host '{hostname}' using AMT KVM redirection or other AMT features.
The function should accept parameters: {parameters_json} (including AMT credentials). It should return: {returns_json}.
Implement the connection to the AMT interface, perform the specified action (e.g., power control, KVM interaction), handle errors, and disconnect. Include necessary imports.
NOTE: Requires a specific AMT library and knowledge of its API.
""",
            RemoteAdapterType.VISION_FALLBACK: """
Generate a Python script using 'pyautogui' for interaction and potentially an external Vision API (like OpenAI Vision or Azure Computer Vision) for analysis.
The task is to {description} on the remote host '{hostname}'. Assume the host's screen is accessible (e.g., via an existing RDP/VNC window or browser-based session).
The function should accept parameters: {parameters_json} and return: {returns_json}.
The script should:
1. Capture the relevant screen area using pyautogui.
2. (Optional but recommended) Send the image to a Vision API with instructions to locate elements or understand the state related to the description.
3. Parse the Vision API response to get coordinates or actions.
4. Use pyautogui to perform mouse clicks, keyboard typing, etc., based on the analysis or predefined logic.
Include necessary imports, API call placeholders, and robust error handling for visual element detection failures.
"""
        }

    async def generate_remote_control_tool(self,
                                           host_id: str,
                                           hostname: str,
                                           description: str,
                                           operation_type: str = "general",
                                           parameters: Optional[Dict[str, Dict[str, Any]]] = None,
                                           returns: Optional[Dict[str, Any]] = None,
                                           ip_address: Optional[str] = None) -> Optional[Tool]:
        """
        Generates or requests generation of a remote control tool.

        Args:
            host_id: Unique ID of the target machine.
            hostname: Hostname of the target machine.
            description: Natural language description of the task.
            operation_type: Category of the operation (e.g., 'power', 'script', 'gui').
            parameters: Expected parameters for the generated tool function.
            returns: Expected return specification for the tool function.
            ip_address: Optional IP address of the target.

        Returns:
            A generated Tool object, or None if generation fails or is not supported.
        """
        parameters = parameters or {}
        returns = returns or {"type": "object", "description": "Result of the remote operation."}

        logging.info(f"Request to generate remote control tool for '{hostname}' ({host_id}): {description}")

        # --- Delegate to PerceptionAction module if available ---
        if self.remote_control_module:
             logging.debug("Delegating remote tool generation to Perception-Action module.")
             try:
                 # Call the module's generation method
                 tool_data = await self.remote_control_module.generate_remote_control_tool(
                     host_id=host_id,
                     hostname=hostname,
                     description=description,
                     operation_type=operation_type,
                     parameters=parameters,
                     returns=returns,
                     ip_address=ip_address
                 )

                 # Process the response from the module
                 if isinstance(tool_data, Tool):
                      logging.info(f"Received Tool object '{tool_data.name}' from Perception-Action module.")
                      tool_data.scope = ToolScope.REMOTE # Ensure scope is correct
                      return tool_data
                 elif isinstance(tool_data, dict):
                      logging.info("Received tool data dict from Perception-Action module. Creating Tool object.")
                      # Ensure essential fields are present
                      if not all(k in tool_data for k in ["name", "description"]):
                           raise ValueError("Received tool data dict is missing required fields (name, description).")
                      # Create Tool object, handling function representation
                      tool = Tool.from_dict(tool_data)
                      if "function_code" in tool_data:
                           tool.function = tool_data["function_code"]
                      elif "remote_tool_id" in tool_data.get("metadata", {}):
                           # If only an ID is provided, function might be a placeholder call
                           remote_id = tool_data["metadata"]["remote_tool_id"]
                           tool.function = f"# Placeholder: Execute remote tool ID {remote_id}"
                      tool.scope = ToolScope.REMOTE # Ensure scope
                      tool.auto_generated = True
                      return tool
                 elif isinstance(tool_data, str):
                      # If only an ID string is returned, create a basic Tool representation
                      logging.warning(f"PerceptionAction module returned only tool ID '{tool_data}'. Creating basic Tool object.")
                      tool_name = f"remote_tool_{tool_data}"
                      return Tool(
                           name=tool_name,
                           description=f"Remote control tool (ID: {tool_data}) for {hostname}: {description}",
                           function=f"# Placeholder: Execute remote tool ID {tool_data}",
                           scope=ToolScope.REMOTE,
                           parameters=parameters,
                           returns=returns,
                           metadata={"remote_tool_id": tool_data, "target_host_id": host_id, "target_hostname": hostname},
                           auto_generated=True,
                           status=ToolStatus.ACTIVE # Assume active if module provided ID
                      )
                 else:
                      logging.error(f"Unexpected return type from PerceptionAction module: {type(tool_data)}. Cannot create tool.")
                      return None

             except NotImplementedError:
                  logging.warning("Perception-Action module does not implement 'generate_remote_control_tool'. Falling back to internal generation.")
                  # Fall through to internal generation logic
             except Exception as e:
                 logging.error(f"Error calling PerceptionAction module for tool generation: {e}", exc_info=True)
                 return None

        # --- Internal Generation Logic (Fallback) ---
        if not self.discovery_service or not self.decision_engine:
             logging.error("Internal remote discovery/decision engine not initialized. Cannot generate remote tool.")
             return None

        logging.debug("Using internal remote tool generation logic.")

        # 1. Discover target capabilities
        try:
            target_info = await self.discovery_service.discover_target(host_id, hostname, ip_address)
        except Exception as e:
            logging.error(f"Failed to discover target {hostname} for tool generation: {e}", exc_info=True)
            return None

        if not target_info.available_adapters:
            logging.error(f"No remote control adapters discovered for {hostname}. Cannot generate tool.")
            return None

        # 2. Select the best adapter using the decision engine
        selected_adapter = self.decision_engine.select_adapter(
            target_info.available_adapters,
            operation_type,
            target_info
        )

        if not selected_adapter:
            logging.error(f"Could not select a suitable adapter for {hostname} (available: {[a.value for a in target_info.available_adapters]}) and operation '{operation_type}'.")
            return None

        logging.info(f"Selected adapter '{selected_adapter.value}' for generating tool for {hostname}.")

        # 3. Prepare generation prompt using the template for the selected adapter
        template = self.adapter_templates.get(selected_adapter)
        if not template:
            logging.error(f"Internal error: No generation template found for adapter {selected_adapter.value}.")
            return None

        try:
            prompt = template.format(
                description=description,
                hostname=hostname,
                host_id=host_id,
                ip_address=target_info.ip_address or "N/A",
                parameters_json=json.dumps(parameters, indent=2),
                returns_json=json.dumps(returns, indent=2)
            ).strip()
        except KeyError as e:
             logging.error(f"Error formatting prompt template for {selected_adapter.value}: Missing key {e}")
             return None

        logging.debug(f"Generated LLM prompt for remote tool:\n---\n{prompt}\n---")

        # 4. Generate code using LLM Orchestrator
        try:
            # Use a model suitable for code generation
            model_config = self.kernel.config.get("llm", {}).get("code_generation_model", {"model_name": "gpt-4o"})

            response = await self.llm_orchestrator.execute_prompt_async(
                task_type="code_generation",
                prompt_template="direct", # Pass the fully formatted prompt directly
                params={"prompt": prompt},
                model_selection=model_config,
                max_tokens=self.config.get("remote_tool_max_tokens", 2048)
            )

            generated_content = response.get("result", "")
            if not generated_content:
                 raise ValueError("LLM did not return any content.")

            # Extract Python code block (handle potential markdown fences)
            code = generated_content
            code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', generated_content, re.DOTALL | re.IGNORECASE)
            if code_match:
                code = code_match.group(1).strip()
            else:
                 # Assume the response is just the code if no markdown block found
                 code = generated_content.strip()


            # Generate a name for the tool based on the description
            tool_name = self._generate_tool_name(description)

            # Save the code to a file
            file_path = self._save_tool_code(tool_name, code)

            # Create the tool object
            tool = Tool(
                name=tool_name,
                description=description,
                function=code,
                scope=ToolScope.CONTAINER,  # Default to container for security
                parameters=parameters,
                returns=returns,
                metadata={
                    "tool_type": tool_type,
                    "generation_prompt": prompt,
                    "extra_context": extra_context
                },
                sandbox_type=SandboxType.DOCKER,
                auto_generated=True,
                file_path=file_path
            )

            # Generate a test case for the tool
            await self._generate_test_case(tool)

            return tool
            
        except Exception as e:
            logging.error(f"Error generating tool: {str(e)}")
            return None

    # Optional: Helper to save generated remote tool code
    def _save_remote_tool_code(self, tool_name: str, code: str, adapter_name: str) -> str:
        """Saves generated remote tool code to a structured directory."""
        try:
            # Ensure the base path is relative to the current file's directory
            current_file_path = Path(__file__).resolve()
            # Navigate up one level from 'modules' to 'evogenesis_core', then to 'tools'
            base_tools_dir = current_file_path.parent.parent / "tools" / "generated_remote"
            adapter_dir = base_tools_dir / adapter_name
            adapter_dir.mkdir(parents=True, exist_ok=True)

            file_path = adapter_dir / f"{tool_name}.py"

            header = f"""#!/usr/bin/env python
# Tool Name: {tool_name}
# Adapter: {adapter_name}
# Generated by: EvoGenesis RemoteControlToolGenerator
# Created: {datetime.now().isoformat()}
# Description: Auto-generated remote tool code. Requires appropriate execution context.
"""
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(header + "\n\n" + code)

            logging.info(f"Saved generated remote tool code to: {file_path}")
            return str(file_path)
        except Exception as e:
            logging.error(f"Failed to save generated remote tool code for {tool_name}: {e}")
            return None


class ToolingSystem:
    """
    Central coordinator for tool management, generation, and execution.
    
    Responsibilities:
    - Registering, discovering, and managing tools
    - Executing tools safely in appropriate sandboxes
    - Generating new tools on-demand via LLMs
    - Handling tool lifecycle (creation, testing, deployment, updates)
    """
    
    def __init__(self, kernel=None):
        """
        Initialize the Tooling System.
        
        Args:
            kernel: Reference to the EvoGenesis kernel
        """
        self.kernel = kernel
        self.tool_registry = {}  # Dictionary of tool_id -> Tool
        self.execution_environment = None
        self.tool_generator = None
        self.remote_tool_generator = None
        self.max_concurrent_executions = 5
        self.execution_semaphore = asyncio.Semaphore(self.max_concurrent_executions)
        
        # Load configuration if kernel is provided
        if kernel:
            config = kernel.config.get("tooling_system", {})
            self.max_concurrent_executions = config.get("max_concurrent_executions", 5)
            self.execution_semaphore = asyncio.Semaphore(self.max_concurrent_executions)
        
        logging.info("Tooling System initialized")
    
    def start(self):
        """Alias for initialize to be called by the kernel."""
        self.initialize()

    def stop(self):
        """Stop the tooling system and clean up resources."""
        logging.info("Stopping Tooling System...")
        # Add cleanup logic here if needed in the future
        # For example, closing sandbox connections or stopping background tasks
        if hasattr(self, 'docker_client') and self.docker_client: # Check if docker_client exists
            try:
                # No explicit close method for the low-level API client by default
                # If using context managers elsewhere, cleanup happens there.
                # If specific containers/networks need cleanup, add logic here.
                pass
            except Exception as e:
                logging.warning(f"Error during Docker client cleanup (if any): {e}")
        # Add cleanup for WASM, E2B, Modal if they have explicit close/shutdown methods
        logging.info("Tooling System stopped.")

    def initialize(self):
        """Initialize the tooling system."""
        if not self.kernel:
            logging.warning("Cannot initialize Tooling System: No kernel reference provided")
            return
            
        # Initialize execution environment
        self.execution_environment = SecureExecutionEnvironment(
            self.kernel.config.get("execution_environment", {})
        )
        
        # Initialize tool generator
        self.tool_generator = ToolGenerator(self.kernel)
        
        # Initialize remote tool generator if perception-action is enabled
        self.remote_tool_generator = RemoteControlToolGenerator(self.kernel)
        
        # Load tools from disk if configured
        if self.kernel.config.get("tooling_system", {}).get("auto_load_tools", True):
            self._load_tools_from_disk()
            
        logging.info("Tooling System components initialized")
    
    def register_tool(self, tool: Tool) -> bool:
        """
        Register a tool in the registry.
        
        Args:
            tool: The tool to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        if tool.id in self.tool_registry:
            logging.warning(f"Tool with ID {tool.id} already exists in registry")
            return False
            
        self.tool_registry[tool.id] = tool
        logging.info(f"Tool '{tool.name}' (ID: {tool.id}) registered successfully")
        return True
    
    def unregister_tool(self, tool_id: str) -> bool:
        """
        Remove a tool from the registry.
        
        Args:
            tool_id: ID of the tool to unregister
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        if tool_id not in self.tool_registry:
            logging.warning(f"Tool with ID {tool_id} not found in registry")
            return False
            
        tool = self.tool_registry.pop(tool_id)
        logging.info(f"Tool '{tool.name}' (ID: {tool_id}) unregistered successfully")
        return True
    
    def get_tool(self, tool_id: str):
        """
        Get a tool by ID.
        
        Args:
            tool_id: ID of the tool to retrieve
            
        Returns:
            The Tool object or None if not found
        """
        return self.tool_registry.get(tool_id)
    
    def get_tools(self, active_only: bool = False):
        """
        Get all registered tools.
        
        Args:
            active_only: If True, only return tools with status ACTIVE
            
        Returns:
            Dictionary of tool_id -> Tool
        """
        if not active_only:
            return self.tool_registry
            
        return {
            tool_id: tool for tool_id, tool in self.tool_registry.items() 
            if tool.status == ToolStatus.ACTIVE
             }
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Tooling System."""
        active_tools = sum(1 for tool in self.tool_registry.values() if tool.status == ToolStatus.ACTIVE)
        testing_tools = sum(1 for tool in self.tool_registry.values() if tool.status == ToolStatus.TESTING)
        failed_tools = sum(1 for tool in self.tool_registry.values() if tool.status == ToolStatus.FAILED)
   
        exec_env_status = {}
        if self.execution_environment:
            exec_env_status["docker_available"] = bool(self.execution_environment.docker_client)
            exec_env_status["wasm_available"] = bool(self.execution_environment.wasm_runtime)
            exec_env_status["e2b_available"] = bool(self.execution_environment.e2b_client)
            exec_env_status["modal_available"] = bool(self.execution_environment.modal_client)
        
        # Get current concurrent executions (approximate)
        # Semaphore value is max_concurrent - currently_acquired
        current_executions = self.max_concurrent_executions - self.execution_semaphore._value if hasattr(self.execution_semaphore, '_value') else 'unknown'

        return {
            "status": "active" if self.execution_environment else "initializing",
            "total_tools": len(self.tool_registry),
            "active_tools": active_tools,
            "testing_tools": testing_tools,
            "failed_tools": failed_tools,
            "execution_environment": exec_env_status,
            "tool_generator_status": "active" if self.tool_generator else "inactive",
            "remote_tool_generator_status": "active" if self.remote_tool_generator else "inactive",
            "max_concurrent_executions": self.max_concurrent_executions,
            "current_executions": current_executions
        }

    async def execute_tool_safely(self,
                               tool_id: str, 
                               args,
                               timeout=None,
                               resource_limits=None):
        """
        Execute a tool safely with resource limits and timeouts.
        
        Args:
            tool_id: ID of the tool to execute
            args: Arguments to pass to the tool
            timeout: Maximum execution time in seconds
            resource_limits: Custom resource limits for this execution
            
        Returns:
            ExecutionResult containing output or error
        """
        tool = self.get_tool(tool_id)
        if not tool:
            return ExecutionResult(
                success=False,
                error=f"Tool with ID {tool_id} not found"
            )
            
        if tool.status != ToolStatus.ACTIVE:
            return ExecutionResult(
                success=False,
                error=f"Tool '{tool.name}' is not active (status: {tool.status})"
            )
            
        # Use semaphore to limit concurrent executions
        async with self.execution_semaphore:
            try:
                # Execute using the execution environment
                result = await self.execution_environment.execute(
                    tool=tool,
                    args=args,
                    timeout=timeout,
                    resource_limits=resource_limits
                )
                
                # Update tool execution statistics
                tool.execution_count += 1
                if result.success:
                    tool.success_count += 1
                else:
                    tool.error_count += 1
                
                # Update average execution time
                tool.average_execution_time = (
                    (tool.average_execution_time * (tool.execution_count - 1) + result.execution_time) 
                    / tool.execution_count
                )
                
                return result
                
            except Exception as e:
                logging.error(f"Error executing tool '{tool.name}': {str(e)}", exc_info=True)
                return ExecutionResult(
                    success=False,
                    error=f"Execution error: {str(e)}",
                    logs=traceback.format_exc()
                )
    
    async def generate_tool(self, 
                         description: str, 
                         parameters: Dict[str, Dict[str, Any]], 
                         returns: Dict[str, Any],
                         auto_register: bool = True) -> Optional[Tool]:
        """
        Generate a new tool from a natural language description.
        
        Args:
            description: Natural language description of what the tool should do
            parameters: Dictionary of parameter names and their specifications
            returns: Specification of what the tool should return
            auto_register: Whether to automatically register the generated tool
            
        Returns:
            Generated Tool object or None if generation failed
        """
        if not self.tool_generator:
            logging.error("Tool generator not initialized")
            return None
            
        try:
            tool = await self.tool_generator.generate_tool(
                description=description,
                parameters=parameters,
                returns=returns
            )
            
            if tool and auto_register:
                self.register_tool(tool)
                
            return tool
            
        except Exception as e:
            logging.error(f"Error generating tool: {str(e)}", exc_info=True)
            return None
    
    async def generate_remote_control_tool(self,
                                        host_id: str,
                                        hostname: str,
                                        description: str,
                                        operation_type: str = "general",
                                        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
                                        auto_register: bool = True) -> Optional[Tool]:
        """
        Generate a remote control tool for a specific target machine.
        
        Args:
            host_id: Unique ID of the target machine
            hostname: Hostname of the target machine
            description: Natural language description of the task
            operation_type: Category of the operation (e.g., 'power', 'script', 'gui')
            parameters: Expected parameters for the generated tool function
            auto_register: Whether to automatically register the generated tool
            
        Returns:
            Generated Tool object or None if generation failed
        """
        if not self.remote_tool_generator:
            logging.error("Remote tool generator not initialized")
            return None
            
        try:
            tool = await self.remote_tool_generator.generate_remote_control_tool(
                host_id=host_id,
                hostname=hostname,
                description=description,
                operation_type=operation_type,
                parameters=parameters
            )
            
            if tool and auto_register:
                self.register_tool(tool)
                
            return tool
            
        except Exception as e:
            logging.error(f"Error generating remote control tool: {str(e)}", exc_info=True)
            return None
    
    def _load_tools_from_disk(self):
        """Load tools from the tools directory."""
        try:
            tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
            if not os.path.exists(tools_dir):
                logging.warning(f"Tools directory not found: {tools_dir}")
                return
                
            # Count loaded tools for logging
            total_loaded = 0
            
            # Load from tools/builtin directory
            builtin_dir = os.path.join(tools_dir, "builtin")
            if os.path.exists(builtin_dir):
                loaded = self._load_tools_from_directory(builtin_dir)
                total_loaded += loaded
                logging.info(f"Loaded {loaded} builtin tools")
                
            # Load from tools/generated directory
            generated_dir = os.path.join(tools_dir, "generated")
            if os.path.exists(generated_dir):
                loaded = self._load_tools_from_directory(generated_dir)
                total_loaded += loaded
                logging.info(f"Loaded {loaded} generated tools")
                
            # Load from tools/generated_remote directory
            remote_dir = os.path.join(tools_dir, "generated_remote")
            if os.path.exists(remote_dir):
                loaded = self._load_tools_from_directory(remote_dir, is_remote=True)
                total_loaded += loaded
                logging.info(f"Loaded {loaded} remote tools")
                
            logging.info(f"Total tools loaded: {total_loaded}")
            
        except Exception as e:
            logging.error(f"Error loading tools from disk: {str(e)}", exc_info=True)
    
    def _load_tools_from_directory(self, directory: str, is_remote: bool = False) -> int:
        """
        Load tools from a directory.
        
        Args:
            directory: Directory to load tools from
            is_remote: Whether these are remote tools
            
        Returns:
            Number of tools loaded
        """
        loaded_count = 0
        
        # Walk the directory recursively
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py") and not file.endswith("_test.py"):
                    try:
                        file_path = os.path.join(root, file)
                        
                        # Read the file to look for a Tool header
                        with open(file_path, "r") as f:
                            content = f.read()
                            
                        # Extract the function name
                        func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
                        if not func_match:
                            logging.warning(f"No function found in {file_path}, skipping")
                            continue
                            
                        function_name = func_match.group(1)
                        
                        # Extract description from docstring
                        doc_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                        description = doc_match.group(1).strip() if doc_match else f"Tool in {file}"
                        
                        # Create a basic tool
                        tool_name = os.path.splitext(file)[0]
                        tool = Tool(
                            name=tool_name,
                            description=description,
                            function=content,
                            file_path=file_path,
                            scope=ToolScope.REMOTE if is_remote else ToolScope.CONTAINER,
                            sandbox_type=SandboxType.NONE if is_remote else SandboxType.DOCKER,
                            auto_generated=True
                        )
                        
                        self.register_tool(tool)
                        loaded_count += 1
                        
                    except Exception as e:
                        logging.error(f"Error loading tool from {file}: {str(e)}")
        
        return loaded_count
