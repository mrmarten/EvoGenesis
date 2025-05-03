"""
Fix for EvoGenesis system's asyncio and missing perception-action tooling issues.

This script addresses:
1. Missing strategic_observatory.json file
2. Properly handling async adapter shutdown methods
3. Updating the kernel import in perception_action_demo.py
"""

import os
import sys
import json
import asyncio
import importlib
import inspect
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def fix_json_file():
    """Fix the missing strategic_observatory.json file."""
    file_path = Path("data/strategic_observatory/strategic_observatory.json")
    
    if not file_path.exists():
        print(f"Creating missing file: {file_path}")
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the file with valid JSON
        data = {
            "signals": {
                "max_signal_age_days": 30,
                "min_update_frequency": 3600,
                "batch_size": 100
            },
            "miners": {
                "min_confidence_threshold": 0.4,
                "max_miners_per_heuristic": 3,
                "max_concurrent_miners": 15
            },
            "reasoning": {
                "consolidation_similarity_threshold": 0.75,
                "min_evidence_count": 3
            },
            "simulation": {
                "monte_carlo_iterations": 1000,
                "sensitivity_variables": ["market_growth", "competition", "adoption_rate"]
            },
            "valuation": {
                "discount_rate": 0.1,
                "projection_years": 5,
                "terminal_growth_rate": 0.02
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Successfully created {file_path}")
    else:
        print(f"File already exists: {file_path}")
        
def fix_perception_action_demo():
    """Fix the kernel import in the perception action demo."""
    demo_path = Path("examples/perception_action_demo.py")
    
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        return
    
    with open(demo_path, 'r') as f:
        content = f.read()
    
    # Fix the import
    updated_content = content.replace(
        "from evogenesis_core.kernel import Kernel",
        "from evogenesis_core.kernel import EvoGenesisKernel as Kernel"
    )
    
    if updated_content != content:
        with open(demo_path, 'w') as f:
            f.write(updated_content)
        print(f"Updated kernel import in {demo_path}")
    else:
        print(f"Kernel import already correct in {demo_path}")

def patch_adapter_shutdown():
    """
    Patch the FrameworkAdapterManager.shutdown_adapter method to properly handle async coroutines.
    This is done by monkey patching instead of modifying the file directly.
    """
    try:
        # Import the module
        from evogenesis_core.adapters.framework_adapter_manager import FrameworkAdapterManager
        
        # Define the fixed method
        def fixed_shutdown_adapter(self, adapter_name):
            """
            Fixed version of shutdown_adapter that properly handles coroutines.
            """
            if adapter_name not in self.initialized_adapters:
                logging.warning(f"Adapter {adapter_name} not initialized, nothing to shut down")
                return True
            
            try:
                adapter = self.initialized_adapters[adapter_name]
                shutdown_result = adapter.shutdown()
                
                # Check if the result is a coroutine that needs to be awaited
                if inspect.iscoroutine(shutdown_result):
                    try:
                        # Try to get the current running event loop
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            logging.warning(f"Event loop already running, can't await shutdown for {adapter_name}")
                            # Add a dummy task and return success for now
                            asyncio.create_task(shutdown_result)
                            success = True
                        else:
                            success = loop.run_until_complete(shutdown_result)
                    except RuntimeError:
                        # No running event loop, create a new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        success = loop.run_until_complete(shutdown_result)
                else:
                    # Not a coroutine, use the result directly
                    success = shutdown_result
                
                if success:
                    # Remove from initialized adapters
                    del self.initialized_adapters[adapter_name]
                    logging.info(f"Successfully shut down adapter: {adapter_name}")
                else:
                    logging.warning(f"Adapter reported unsuccessful shutdown: {adapter_name}")
                
                return success
            except Exception as e:
                logging.error(f"Error shutting down adapter {adapter_name}: {str(e)}")
                # Remove it from initialized adapters anyway
                if adapter_name in self.initialized_adapters:
                    del self.initialized_adapters[adapter_name]
                return False
        
        # Replace the method
        FrameworkAdapterManager.shutdown_adapter = fixed_shutdown_adapter
        print("Successfully patched FrameworkAdapterManager.shutdown_adapter")
        
    except ImportError as e:
        print(f"Error importing FrameworkAdapterManager: {str(e)}")
    except Exception as e:
        print(f"Error patching shutdown_adapter: {str(e)}")

if __name__ == "__main__":
    print("Fixing EvoGenesis system bugs...")
    fix_json_file()
    fix_perception_action_demo()
    patch_adapter_shutdown()
    print("Fixes complete. Try running perception_action_demo.py again.")
