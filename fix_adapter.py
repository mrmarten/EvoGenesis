# Simple script to fix the semantic_kernel_adapter.py file
# This addresses the "object dict can't be used in 'await' expression" error

with open("evogenesis_core/adapters/semantic_kernel_adapter.py", "r") as f:
    lines = f.readlines()

# Find the async method
async_method_start = -1
async_method_end = -1

for i, line in enumerate(lines):
    if "async def _get_framework_capabilities_async" in line:
        async_method_start = i
        break

if async_method_start != -1:
    # Find the end of the method
    indent_level = len(lines[async_method_start]) - len(lines[async_method_start].lstrip())
    for i in range(async_method_start + 1, len(lines)):
        if i < len(lines) - 1 and lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= indent_level:
            async_method_end = i
            break
    if async_method_end == -1:
        async_method_end = len(lines)
    
    # Replace the async method with a proper implementation
    fixed_method = [
        "    async def _get_framework_capabilities_async(self) -> Dict[str, Any]:\n",
        '        """\n',
        "        Get the capabilities of Semantic Kernel (async implementation).\n",
        "        This method ensures proper async behavior by including an await.\n",
        "        \n",
        "        Returns:\n",
        "            Dictionary describing Semantic Kernel's capabilities\n",
        '        """\n',
        "        # Include a minimal await operation to ensure this is a proper coroutine\n",
        "        await asyncio.sleep(0)\n",
        "        \n",
        "        # Return the capabilities\n",
        "        return {\n",
        '            "name": "Semantic Kernel",\n',
        '            "version": sk.__version__ if hasattr(sk, "__version__") else "unknown",\n',
        '            "features": {\n',
        '                "planning": True,\n',
        '                "function_calling": True,\n',
        '                "plugins": True,\n',
        '                "memory": True,\n',
        '                "connectors": {\n',
        '                    "openai": True,\n',
        '                    "azure_openai": True\n',
        "                }\n",
        "            },\n",
        '            "capabilities": {\n',
        '                "agents": True,\n',
        '                "multi_agent": True,\n',
        '                "reasoning": True\n',
        "            },\n",
        '            "supported_models": ["gpt-4", "gpt-3.5-turbo", "claude-3"]\n',
        "        }\n",
    ]
    
    # Replace the method in the file
    new_lines = lines[:async_method_start] + fixed_method + lines[async_method_end:]
    
    # Write the changes back to the file
    with open("evogenesis_core/adapters/semantic_kernel_adapter.py", "w") as f:
        f.writelines(new_lines)
    
    print("Successfully fixed the _get_framework_capabilities_async method")
else:
    print("Could not find the async method in the file. No changes were made.")
