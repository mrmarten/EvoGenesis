#!/usr/bin/env python
"""
This script will fix the specific issue with the _get_framework_capabilities_async method
in the semantic_kernel_adapter.py file by rewriting the method entirely.
"""

import re

file_path = "evogenesis_core/adapters/semantic_kernel_adapter.py"

with open(file_path, "r") as f:
    content = f.read()

# First, find the class definition to get proper indentation level
class_match = re.search(r"class SemanticKernelAdapter\(AgentExecutionAdapter\):", content)
if not class_match:
    print("Could not find SemanticKernelAdapter class definition. Aborting.")
    exit(1)

# Now find the async method and its current implementation
async_method_pattern = r'(\s+)async def _get_framework_capabilities_async\([^)]*\)[^:]*:.*?(?=\s+def|\s*$)'
async_method_match = re.search(async_method_pattern, content, re.DOTALL)

if async_method_match:
    # Get the indentation level
    indentation = async_method_match.group(1)
    
    # Create a fixed version of the method
    fixed_method = f'''{indentation}async def _get_framework_capabilities_async(self) -> Dict[str, Any]:
{indentation}    """
{indentation}    Get the capabilities of Semantic Kernel (async implementation).
{indentation}    This method is kept for API compatibility but not used in the sync version.
{indentation}    
{indentation}    Returns:
{indentation}        Dictionary describing Semantic Kernel's capabilities
{indentation}    """
{indentation}    # Use await to make this a proper coroutine
{indentation}    await asyncio.sleep(0)
{indentation}    
{indentation}    # Return the capabilities dictionary
{indentation}    return {{
{indentation}        "name": "Semantic Kernel",
{indentation}        "version": sk.__version__ if hasattr(sk, "__version__") else "unknown",
{indentation}        "features": {{
{indentation}            "planning": True,
{indentation}            "function_calling": True,
{indentation}            "memory": True,
{indentation}            "native_plugins": True,
{indentation}            "semantic_plugins": True,
{indentation}            "team_coordination": False  # Not built into SK
{indentation}        }},
{indentation}        "supported_models": ["gpt-4", "gpt-3.5-turbo", "claude-3"]
{indentation}    }}'''
    
    # Replace the async method with the fixed version
    content = re.sub(async_method_pattern, fixed_method, content, flags=re.DOTALL)

    # Fix any remaining issues with the file
    # Remove duplicate or overlapping blocks if present
    content = re.sub(r'}}\s+}},', '}},', content)
    
    # Write the fixed content back to the file
    with open(file_path, "w") as f:
        f.write(content)
    
    print("Successfully fixed the _get_framework_capabilities_async method")
else:
    print("Could not find the async method in the file. No changes were made.")
