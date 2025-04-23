#!/usr/bin/env python
"""
This script fixes the async-related issues in the Semantic Kernel adapter.
"""

import re

# Path to the semantic_kernel_adapter.py file
file_path = "evogenesis_core/adapters/semantic_kernel_adapter.py"

# Read the current content of the file
with open(file_path, "r") as f:
    content = f.read()

# Fix the async method implementation
# Find the async method declaration and its implementation
async_method_pattern = r"async def _get_framework_capabilities_async\(self\).*?(?=\n\s*def|\n\s*$|$)"
async_method_match = re.search(async_method_pattern, content, re.DOTALL)

if async_method_match:
    async_method = async_method_match.group(0)
    
    # Create a fixed version of the method with proper awaitable behavior
    fixed_async_method = """async def _get_framework_capabilities_async(self) -> Dict[str, Any]:
        '''
        Get the capabilities of Semantic Kernel (async implementation).
        This method ensures proper async behavior by including an await.
        
        Returns:
            Dictionary describing Semantic Kernel's capabilities
        '''
        # Include a minimal await operation to ensure this is a proper coroutine
        await asyncio.sleep(0)
        
        # Return the capabilities
        return {
            "name": "Semantic Kernel",
            "version": sk.__version__ if hasattr(sk, "__version__") else "unknown",
            "features": {
                "planning": True,
                "function_calling": True,
                "plugins": True,
                "memory": True,
                "connectors": {
                    "openai": True,
                    "azure_openai": True
                }
            },
            "capabilities": {
                "agents": True,
                "multi_agent": True,
                "reasoning": True
            },
            "supported_models": ["gpt-4", "gpt-3.5-turbo", "claude-3"]
        }"""
    
    # Replace the async method with the fixed version
    content = content.replace(async_method, fixed_async_method)

    # Write the fixed content back to the file
    with open(file_path, "w") as f:
        f.write(content)
    
    print(f"Successfully fixed the semantic_kernel_adapter.py file")
else:
    print("Could not find the async method in the file. No changes were made.")
