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


class ToolScope(str, Enum):
    """Scope/permission level of a tool."""
    SYSTEM = "system"          # Full system access (highest privilege)
    WORKSPACE = "workspace"    # Access to workspace files and limited APIs
    CONTAINER = "container"    # Runs in a Docker container (recommended)
    CLOUD = "cloud"            # Runs in a cloud sandbox (most secure)
    MEMORY = "memory"          # In-memory execution only (no file/network)


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
            return await self._execute_subprocess(tool, args, timeout)
        
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
        # Placeholder implementation - would need actual WASM runtime
        logging.warning("WASM execution not fully implemented, falling back to Docker")
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
            limits: Resource limits
            
        Returns:
            ExecutionResult object
        """
        # Placeholder for E2B implementation - would need E2B SDK
        if not self.e2b_client:
            logging.warning("E2B client not available, falling back to Docker")
            return await self._execute_docker(tool, args, timeout, limits)
        
        # Mock implementation
        start_time = time.time()
        logging.info(f"Would execute {tool.name} in E2B sandbox")
        
        await asyncio.sleep(0.5)  # Simulate some processing time
        
        return ExecutionResult(
            success=True,
            output={"result": "This is a placeholder for E2B execution"},
            logs="E2B execution logs would appear here",
            execution_time=time.time() - start_time,
            resource_usage={"memory": "100m", "cpu": "0.1"}
        )
    
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
            limits: Resource limits
            
        Returns:
            ExecutionResult object
        """
        # Placeholder for Modal implementation - would need Modal SDK
        if not self.modal_client:
            logging.warning("Modal client not available, falling back to Docker")
            return await self._execute_docker(tool, args, timeout, limits)
        
        # Mock implementation
        start_time = time.time()
        logging.info(f"Would execute {tool.name} in Modal sandbox")
        
        await asyncio.sleep(0.5)  # Simulate some processing time
        
        return ExecutionResult(
            success=True,
            output={"result": "This is a placeholder for Modal execution"},
            logs="Modal execution logs would appear here",
            execution_time=time.time() - start_time,
            resource_usage={"memory": "100m", "cpu": "0.1"}
        )


class ToolingSystem:
    """
    Manages tool generation, integration, and secure execution.
    
    Responsible for:
    - Automatically generating tools as needed
    - Securely executing tools with appropriate sandboxing
    - Enabling tool discovery and integration
    - Self-correction and error handling
    """
    
    def __init__(self, kernel):
        """
        Initialize the Tooling System.
        
        Args:
            kernel: Reference to the EvoGenesis kernel
        """
        self.kernel = kernel
        self.config = kernel.config.get("tooling_system", {})
        self.tool_registry = {}  # id -> Tool
        self.tool_generator = ToolGenerator(kernel)
        self.secure_execution = SecureExecutionEnvironment(self.config.get("execution", {}))
        
        # Load built-in tools
        self.load_built_in_tools()
    
    def start(self):
        """Start the Tooling System."""
        logging.info("Starting Tooling System")
    
    def stop(self):
        """Stop the Tooling System."""
        logging.info("Stopping Tooling System")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Tooling System.
        
        Returns:
            Dictionary with status information
        """
        return {
            "tools_registered": len(self.tool_registry),
            "docker_available": bool(self.secure_execution.docker_client),
            "wasm_available": bool(self.secure_execution.wasm_runtime),
            "e2b_available": bool(self.secure_execution.e2b_client),
            "modal_available": bool(self.secure_execution.modal_client)
        }
    
    def load_built_in_tools(self):
        """Load built-in tools."""
        tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools", "builtin")
        
        if not os.path.exists(tools_dir):
            return
        
        for file_name in os.listdir(tools_dir):
            if file_name.endswith(".py") and not file_name.startswith("_"):
                try:
                    module_name = file_name[:-3]
                    module_path = f"evogenesis_core.tools.builtin.{module_name}"
                    module = importlib.import_module(module_path)
                    
                    # Look for tool definitions
                    for name, obj in inspect.getmembers(module):
                        if name.endswith("_tool") and callable(obj):
                            # Get tool metadata if available
                            metadata = getattr(obj, "metadata", {})
                            description = metadata.get("description", obj.__doc__ or f"{name} tool")
                            
                            # Create the tool
                            tool = Tool(
                                name=name,
                                description=description,
                                function=obj,
                                scope=metadata.get("scope", ToolScope.WORKSPACE),
                                parameters=metadata.get("parameters", {}),
                                returns=metadata.get("returns", {"type": "object"}),
                                metadata=metadata,
                                sandbox_type=metadata.get("sandbox_type", SandboxType.SUBPROCESS),
                                auto_generated=False,
                                file_path=os.path.join(tools_dir, file_name)
                            )
                            
                            # Register the tool
                            self.register_tool(tool)
                            
                except Exception as e:
                    logging.error(f"Error loading built-in tool from {file_name}: {str(e)}")
    
    def register_tool(self, tool: Tool) -> str:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool to register
            
        Returns:
            Tool ID
        """
        self.tool_registry[tool.id] = tool
        logging.info(f"Registered tool: {tool.name} [{tool.id}]")
        return tool.id
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """
        Get a tool by ID.
        
        Args:
            tool_id: ID of the tool
            
        Returns:
            Tool if found, None otherwise
        """
        return self.tool_registry.get(tool_id)
    
    def get_tools_by_name(self, name: str) -> List[Tool]:
        """
        Get tools by name.
        
        Args:
            name: Name to search for
            
        Returns:
            List of matching tools
        """
        return [tool for tool in self.tool_registry.values() if tool.name == name]
    
    def search_tools(self, query: str) -> List[Tool]:
        """
        Search for tools by keyword.
        
        Args:
            query: Search query
            
        Returns:
            List of matching tools
        """
        query = query.lower()
        results = []
        
        for tool in self.tool_registry.values():
            if (query in tool.name.lower() or 
                query in tool.description.lower() or
                any(query in param.lower() for param in tool.parameters.keys())):
                results.append(tool)
        
        return results
    
    async def generate_tool_from_description(self, 
                                           description: str,
                                           parameters: Dict[str, Dict[str, Any]] = None,
                                           returns: Dict[str, Any] = None,
                                           tool_type: str = "python_script",
                                           extra_context: Dict[str, Any] = None) -> Optional[str]:
        """
        Generate a tool from a natural language description.
        
        Args:
            description: Description of what the tool should do
            parameters: Dictionary describing the parameters
            returns: Dictionary describing the return type
            tool_type: Type of tool to generate
            extra_context: Additional context for generation
            
        Returns:
            ID of the generated tool, or None if generation failed
        """
        # Set default parameter structure if not provided
        if parameters is None:
            parameters = {}
        
        # Set default return structure if not provided
        if returns is None:
            returns = {"type": "object"}
        
        # Generate the tool
        tool = await self.tool_generator.generate_tool(
            description=description,
            parameters=parameters,
            returns=returns,
            tool_type=tool_type,
            extra_context=extra_context
        )
        
        if tool:
            # Register the tool
            return self.register_tool(tool)
        
        return None
    
    async def execute_tool_safely(self, 
                               tool_id: str, 
                               args: Dict[str, Any],
                               sandbox_override: Optional[SandboxType] = None,
                               timeout: Optional[float] = None,
                               resource_limits: Optional[Dict[str, Any]] = None,
                               permission_override: bool = False) -> ExecutionResult:
        """
        Execute a tool with appropriate security measures.
        
        Args:
            tool_id: ID of the tool to execute
            args: Arguments for the tool
            sandbox_override: Override the default sandbox type
            timeout: Custom timeout for this execution
            resource_limits: Custom resource limits for this execution
            permission_override: Whether to override permission restrictions
            
        Returns:
            ExecutionResult object
        """
        # Get the tool
        tool = self.get_tool(tool_id)
        if not tool:
            return ExecutionResult(
                success=False,
                error=f"Tool not found: {tool_id}"
            )
        
        # Check tool status
        if tool.status != ToolStatus.ACTIVE and tool.status != ToolStatus.TESTING:
            return ExecutionResult(
                success=False,
                error=f"Tool is {tool.status}, cannot execute"
            )
        
        # Check if HITL permission is required for sensitive scopes
        if (tool.scope in [ToolScope.SYSTEM, ToolScope.WORKSPACE] and 
            not permission_override):
            # Get HITL interface
            hitl = self.kernel.hitl_interface
            
            # Request permission
            request_id = hitl.request_permission(
                agent_id="tooling_system",
                action_description=f"Execute tool '{tool.name}' with {tool.scope} scope",
                details={
                    "tool_id": tool_id,
                    "tool_name": tool.name,
                    "scope": tool.scope,
                    "arguments": args
                },
                timeout=300  # 5 minute timeout
            )
            
            # Wait for approval (with timeout)
            max_wait = 300  # 5 minutes
            start_wait = time.time()
            
            while time.time() - start_wait < max_wait:
                approved, notes = hitl.check_permission(request_id)
                
                if approved is True:
                    # Permission granted, continue
                    break
                elif approved is False:
                    # Permission denied
                    return ExecutionResult(
                        success=False,
                        error=f"Permission denied: {notes}"
                    )
                
                # Wait before checking again
                await asyncio.sleep(1)
            
            if time.time() - start_wait >= max_wait:
                # Timeout waiting for permission
                return ExecutionResult(
                    success=False,
                    error="Timeout waiting for permission approval"
                )
        
        # Validate arguments against parameter schema
        # This validation would typically use a schema validation library
        # For simplicity, just check if all required parameters are present
        for param_name, param_spec in tool.parameters.items():
            if param_spec.get("required", False) and param_name not in args:
                return ExecutionResult(
                    success=False,
                    error=f"Missing required parameter: {param_name}"
                )
        
        # Execute the tool
        start_time = time.time()
        try:
            result = await self.secure_execution.execute(
                tool=tool,
                args=args,
                sandbox_type=sandbox_override,
                timeout=timeout,
                resource_limits=resource_limits
            )
            
            # Update tool statistics
            tool.execution_count += 1
            if result.success:
                tool.success_count += 1
            else:
                tool.error_count += 1
            
            # Update average execution time
            exec_time = time.time() - start_time
            tool.average_execution_time = ((tool.average_execution_time * (tool.execution_count - 1)) + 
                                          exec_time) / tool.execution_count
            
            # Update tool status based on success rate
            if tool.execution_count >= 5:
                success_rate = tool.success_count / tool.execution_count
                if success_rate < 0.5:
                    tool.status = ToolStatus.FAILED
                elif tool.status == ToolStatus.TESTING and success_rate >= 0.8:
                    tool.status = ToolStatus.ACTIVE
            
            return result
            
        except Exception as e:
            exec_time = time.time() - start_time
            
            # Update tool statistics
            tool.execution_count += 1
            tool.error_count += 1
            tool.average_execution_time = ((tool.average_execution_time * (tool.execution_count - 1)) + 
                                          exec_time) / tool.execution_count
            
            return ExecutionResult(
                success=False,
                error=f"Error executing tool: {str(e)}",
                logs=traceback.format_exc(),
                execution_time=exec_time
            )
    
    async def execute_with_self_correction(self,
                                        tool_id: str,
                                        args: Dict[str, Any],
                                        max_attempts: int = 3,
                                        **kwargs) -> ExecutionResult:
        """
        Execute a tool with automatic error correction attempts.
        
        Args:
            tool_id: ID of the tool to execute
            args: Arguments for the tool
            max_attempts: Maximum number of correction attempts
            **kwargs: Additional arguments for execute_tool_safely
            
        Returns:
            ExecutionResult object with correction history
        """
        # Initial execution
        result = await self.execute_tool_safely(tool_id, args, **kwargs)
        
        # If successful, return immediately
        if result.success:
            result.metadata = {"correction_attempts": 0}
            return result
        
        # Get the tool for correction
        tool = self.get_tool(tool_id)
        if not tool:
            result.metadata = {"correction_attempts": 0}
            return result
        
        # Prepare for correction attempts
        attempts = 0
        correction_history = [{
            "attempt": 0,
            "args": args,
            "error": result.error,
            "logs": result.logs
        }]
        
        # Try self-correction
        while not result.success and attempts < max_attempts:
            attempts += 1
            
            # Generate correction using LLM
            corrected_args, code_fix = await self._generate_correction(
                tool=tool,
                args=args,
                error=result.error,
                logs=result.logs
            )
            
            # If we have a code fix, apply it to the tool
            if code_fix and isinstance(tool.function, str):
                # Create a new version of the tool with fixed code
                new_tool = Tool(
                    name=f"{tool.name}_fixed_{attempts}",
                    description=tool.description,
                    function=code_fix,
                    scope=tool.scope,
                    parameters=tool.parameters,
                    returns=tool.returns,
                    metadata={**tool.metadata, "original_tool_id": tool.id},
                    sandbox_type=tool.sandbox_type,
                    auto_generated=True
                )
                
                # Register the fixed tool
                new_tool_id = self.register_tool(new_tool)
                
                # Try executing with the fixed tool
                result = await self.execute_tool_safely(new_tool_id, corrected_args or args, **kwargs)
            else:
                # Try executing with corrected arguments
                result = await self.execute_tool_safely(tool_id, corrected_args or args, **kwargs)
            
            # Update correction history
            correction_history.append({
                "attempt": attempts,
                "args": corrected_args or args,
                "code_fix": bool(code_fix),
                "error": result.error,
                "logs": result.logs,
                "success": result.success
            })
        
        # Add correction history to result
        result.metadata = {
            "correction_attempts": attempts,
            "correction_history": correction_history
        }
        
        return result
    
    async def _generate_correction(self, 
                                tool: Tool, 
                                args: Dict[str, Any],
                                error: str,
                                logs: Optional[str]) -> tuple:
        """
        Generate a correction for a failed tool execution.
        
        Args:
            tool: The tool that failed
            args: Arguments that were passed
            error: Error message
            logs: Execution logs
            
        Returns:
            Tuple of (corrected_args, code_fix)
        """
        # Prepare the prompt
        prompt = f"""
You are tasked with fixing an error in a tool execution. Here are the details:

Tool Description: {tool.description}
Error Message: {error}

Arguments passed to the tool:
```json
{json.dumps(args, indent=2)}
```
"""

        # Add code if available
        if isinstance(tool.function, str):
            prompt += f"\nTool Code:\n```python\n{tool.function}\n```\n"
        
        # Add logs if available
        if logs:
            prompt += f"\nExecution Logs:\n```\n{logs}\n```\n"
        
        prompt += """
Please analyze the error and suggest ONE of the following fixes:

1. Corrected arguments - If the issue is with the input arguments, provide fixed arguments in JSON format.
2. Code fix - If the issue is with the tool code, provide the full corrected code.

Respond with EITHER corrected arguments OR a code fix, not both. Format your response as follows:

For argument fix:
```json
{
  "fix_type": "arguments",
  "corrected_args": { ... corrected arguments ... }
}
```

For code fix:
```json
{
  "fix_type": "code",
  "corrected_code": "... full corrected code ..."
}
```

Include a brief explanation of what was wrong and how your fix addresses it.
"""

        # Generate the correction
        try:
            response = await self.kernel.llm_orchestrator.execute_prompt_async(
                task_type="code_correction",
                prompt_template="direct",
                params={"prompt": prompt},
                model_selection={
                    "model_name": "gpt-4o",  # Use a strong model for fixes
                    "provider": "openai"
                },
                max_tokens=2000
            )
            
            # Extract the correction from the response
            correction = response.get("result", "")
            
            # Parse the JSON response
            if isinstance(correction, str):
                # Try to extract JSON block if present
                json_match = re.search(r'```json\s*(.*?)\s*```', correction, re.DOTALL)
                if json_match:
                    correction = json_match.group(1)
                
                try:
                    fix = json.loads(correction)
                    fix_type = fix.get("fix_type")
                    
                    if fix_type == "arguments":
                        return fix.get("corrected_args"), None
                    elif fix_type == "code":
                        return None, fix.get("corrected_code")
                    else:
                        logging.warning(f"Unknown fix type: {fix_type}")
                        return None, None
                        
                except json.JSONDecodeError:
                    logging.warning("Could not parse correction as JSON")
                    return None, None
            
            return None, None
            
        except Exception as e:
            logging.error(f"Error generating correction: {str(e)}")
            return None, None
    
    def get_tool_summary(self, tool_id: str) -> Dict[str, Any]:
        """
        Get a summary of a tool's metadata and usage statistics.
        
        Args:
            tool_id: ID of the tool
            
        Returns:
            Dictionary with tool summary
        """
        tool = self.get_tool(tool_id)
        if not tool:
            return {"error": f"Tool not found: {tool_id}"}
        
        return {
            "id": tool.id,
            "name": tool.name,
            "description": tool.description,
            "status": tool.status,
            "scope": tool.scope,
            "sandbox_type": tool.sandbox_type,
            "auto_generated": tool.auto_generated,
            "parameters": tool.parameters,
            "returns": tool.returns,
            "execution_count": tool.execution_count,
            "success_rate": tool.success_count / tool.execution_count if tool.execution_count > 0 else 0,
            "average_execution_time": tool.average_execution_time
        }
    
    def update_tool_status(self, tool_id: str, status: ToolStatus) -> bool:
        """
        Update a tool's status.
        
        Args:
            tool_id: ID of the tool
            status: New status
            
        Returns:
            True if successful, False otherwise
        """
        tool = self.get_tool(tool_id)
        if not tool:
            return False
        
        tool.status = status
        tool.updated_at = datetime.now()
        return True
