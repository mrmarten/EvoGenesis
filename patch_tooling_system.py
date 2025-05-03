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
    
    def initialize(self):
        """Initialize components that require the kernel to be fully set up."""
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
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """
        Get a tool by ID.
        
        Args:
            tool_id: ID of the tool to retrieve
            
        Returns:
            The Tool object or None if not found
        """
        return self.tool_registry.get(tool_id)
    
    def get_tools(self, active_only: bool = False) -> Dict[str, Tool]:
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
    
    async def execute_tool_safely(self, 
                               tool_id: str, 
                               args: Dict[str, Any],
                               timeout: Optional[float] = None,
                               resource_limits: Optional[Dict[str, Any]] = None) -> ExecutionResult:
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
                # For remote tools, use a different execution path
                if tool.scope == ToolScope.REMOTE and self.remote_tool_generator:
                    # If we have a remote tool ID in metadata, use it for execution
                    remote_tool_id = tool.metadata.get("remote_tool_id")
                    if remote_tool_id and hasattr(self.kernel, 'perception_action_module'):
                        result = await self.kernel.perception_action_module.execute_remote_control_tool(
                            remote_tool_id, args
                        )
                        # Convert result to ExecutionResult
                        return ExecutionResult(
                            success=result.get("success", False),
                            output=result.get("result"),
                            logs=result.get("logs"),
                            execution_time=result.get("execution_time", 0.0)
                        )
                    else:
                        return ExecutionResult(
                            success=False,
                            error="Remote tool execution not supported (missing remote_tool_id or perception_action_module)"
                        )
                
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
