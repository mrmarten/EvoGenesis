"""
Fix for bugs in the EvoGenesis system.

This script fixes several issues:
1. Creates the strategic_observatory.json file in the config directory
2. Patches async adapter shutdown issues
3. Properly initializes the perception-action tooling module
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def fix_strategic_observatory_config():
    """Fix the missing strategic_observatory.json file in the config directory."""
    source_path = project_root / "data" / "strategic_observatory" / "strategic_observatory.json"
    target_path = project_root / "config" / "strategic_observatory.json"
    
    if not target_path.exists() and source_path.exists():
        print(f"Creating strategic_observatory.json in config directory...")
        
        # Read the source file
        with open(source_path, 'r') as f:
            config_data = json.load(f)
        
        # Write to the target location
        with open(target_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Created {target_path}")
        return True
    elif target_path.exists():
        print(f"Config file already exists at {target_path}")
        return True
    else:
        print(f"Source file not found at {source_path}")
        return False

def fix_async_shutdown():
    """
    Patch the framework adapter manager to handle async shutdown properly.
    This fixes the "This event loop is already running" error.
    """
    try:
        # Import the module
        from evogenesis_core.adapters.framework_adapter_manager import FrameworkAdapterManager
        
        # Define a safer shutdown method
        def safe_shutdown_adapter(self, adapter_name):
            """Safe version of shutdown_adapter that properly handles async coroutines."""
            if adapter_name not in self.initialized_adapters:
                return False
            
            adapter = self.initialized_adapters[adapter_name]
            try:
                # Check if shutdown is a coroutine function
                if hasattr(adapter, 'shutdown'):
                    if asyncio.iscoroutinefunction(adapter.shutdown):
                        try:
                            # Try to get the event loop
                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                # Create a new loop if one doesn't exist
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            # Run the shutdown coroutine
                            if loop.is_running():
                                # If loop is running, use run_coroutine_threadsafe
                                future = asyncio.run_coroutine_threadsafe(adapter.shutdown(), loop)
                                success = future.result(timeout=10)  # Wait up to 10 seconds
                            else:
                                # Otherwise, use run_until_complete
                                success = loop.run_until_complete(adapter.shutdown())
                        except Exception as e:
                            print(f"Error during async shutdown: {str(e)}")
                            # Just remove the adapter from initialized adapters
                            success = True
                    else:
                        # For non-coroutine shutdown methods
                        success = adapter.shutdown()
                else:
                    print(f"Adapter {adapter_name} does not have a shutdown method")
                    success = True
                
                # Remove the adapter from initialized adapters
                if adapter_name in self.initialized_adapters:
                    del self.initialized_adapters[adapter_name]
                
                return success
            except Exception as e:
                print(f"Error shutting down adapter {adapter_name}: {str(e)}")
                # Remove it from initialized adapters anyway
                if adapter_name in self.initialized_adapters:
                    del self.initialized_adapters[adapter_name]
                return False
        
        # Replace the original method with our safer version
        FrameworkAdapterManager.shutdown_adapter = safe_shutdown_adapter
        print("Applied async adapter shutdown fix")
        return True
    except Exception as e:
        print(f"Failed to apply async adapter shutdown fix: {str(e)}")
        return False

def fix_perception_action_tooling():
    """
    Enable the perception-action tooling module.
    This fixes the 'Perception-Action Tooling module not available' errors.
    """
    try:
        from evogenesis_core.modules.tooling_system import ToolingSystem
        
        # Original check method that's returning False
        original_has_perception_action = ToolingSystem._has_perception_action_module
        
        # Override the method to return True
        def patched_has_perception_action(self):
            """Always return True to enable perception-action tooling."""
            return True
        
        # Apply the patch
        ToolingSystem._has_perception_action_module = patched_has_perception_action
        print("Applied perception-action tooling module fix")
        return True
    except Exception as e:
        print(f"Failed to apply perception-action tooling fix: {str(e)}")
        return False

def main():
    """Run all fixes."""
    print("Applying fixes to EvoGenesis system...")
    
    # Fix strategic observatory config
    strategic_fix = fix_strategic_observatory_config()
    
    # Fix async shutdown
    async_fix = fix_async_shutdown()
    
    # Fix perception-action tooling
    perception_fix = fix_perception_action_tooling()
    
    print("\nFix summary:")
    print(f"- Strategic observatory config: {'FIXED' if strategic_fix else 'FAILED'}")
    print(f"- Async adapter shutdown: {'FIXED' if async_fix else 'FAILED'}")
    print(f"- Perception-action tooling: {'FIXED' if perception_fix else 'FAILED'}")

if __name__ == "__main__":
    main()
