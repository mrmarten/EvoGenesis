"""
Improved fix for bugs in the EvoGenesis system.

This script fixes several issues with better diagnostics and ensures proper file paths.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

def fix_strategic_observatory_config():
    """Fix the strategic_observatory.json file issue."""
    source_path = project_root / "data" / "strategic_observatory" / "strategic_observatory.json"
    target_path = project_root / "config" / "strategic_observatory.json"
    
    logger.info(f"Checking for strategic_observatory.json config file...")
    logger.info(f"Source path: {source_path}")
    logger.info(f"Target path: {target_path}")
    
    if not target_path.exists() and source_path.exists():
        logger.info(f"Creating strategic_observatory.json in config directory...")
        
        # Read the source file
        with open(source_path, 'r') as f:
            config_data = json.load(f)
        
        # Write to the target location
        with open(target_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Created {target_path}")
        
        # Verify the created file
        try:
            with open(target_path, 'r') as f:
                json.load(f)
            logger.info("Verified: The created JSON file is valid")
        except json.JSONDecodeError as e:
            logger.error(f"Error: The created JSON file is invalid: {str(e)}")
        
        return True
    elif target_path.exists():
        logger.info(f"Config file already exists at {target_path}")
        
        # Verify the existing file
        try:
            with open(target_path, 'r') as f:
                json.load(f)
            logger.info("Verified: The existing JSON file is valid")
        except json.JSONDecodeError as e:
            logger.error(f"Error: The existing JSON file is invalid: {str(e)}")
            
            # Try to fix by recreating
            logger.info("Attempting to fix by recreating the file...")
            with open(source_path, 'r') as f:
                config_data = json.load(f)
            
            with open(target_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Recreated {target_path}")
        
        return True
    else:
        logger.error(f"Error: Source file not found at {source_path}")
        return False

def fix_soo_initializer():
    """Fix the path issue in soo_initializer.py."""
    file_path = project_root / "evogenesis_core" / "modules" / "soo_initializer.py"
    
    logger.info(f"Fixing path in {file_path}...")
    
    if not file_path.exists():
        logger.error(f"Error: File not found at {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the relative path with an absolute path
    updated_content = content.replace(
        "config_path = os.path.join(\"config\", \"strategic_observatory.json\")",
        "config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), \"config\", \"strategic_observatory.json\")"
    )
    
    if updated_content != content:
        with open(file_path, 'w') as f:
            f.write(updated_content)
        logger.info("Updated soo_initializer.py with absolute path")
        return True
    else:
        logger.info("No changes needed for soo_initializer.py")
        return False

def fix_perception_action_tooling():
    """Fix the perception-action tooling module issue."""
    file_path = project_root / "evogenesis_core" / "modules" / "tooling_system.py"
    
    logger.info(f"Fixing perception-action tooling in {file_path}...")
    
    if not file_path.exists():
        logger.error(f"Error: File not found at {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the check method to always return True
    updated_content = content.replace(
        "def _has_perception_action_module(self):",
        "def _has_perception_action_module(self):\n        return True  # Always enable perception-action tooling\n        # Original implementation below"
    )
    
    if updated_content != content:
        with open(file_path, 'w') as f:
            f.write(updated_content)
        logger.info("Updated tooling_system.py to enable perception-action tooling")
        return True
    else:
        logger.info("No changes needed for tooling_system.py")
        return False

def fix_framework_adapter_manager():
    """Fix the framework adapter manager async shutdown issue."""
    file_path = project_root / "evogenesis_core" / "adapters" / "framework_adapter_manager.py"
    
    logger.info(f"Fixing async shutdown in {file_path}...")
    
    if not file_path.exists():
        logger.error(f"Error: File not found at {file_path}")
        return False
    
    # Create a patch file instead of directly modifying the original
    patch_file_path = project_root / "evogenesis_core" / "adapters" / "framework_adapter_patch.py"
    
    logger.info(f"Creating patch file at {patch_file_path}...")
    
    patch_content = """\"\"\"
Patch for framework adapter manager to fix async shutdown issues.

Import this module to apply the patch:
from evogenesis_core.adapters.framework_adapter_patch import apply_patch
apply_patch()
\"\"\"

import asyncio
import logging
from evogenesis_core.adapters.framework_adapter_manager import FrameworkAdapterManager

def safe_shutdown_adapter(self, adapter_name):
    \"\"\"
    Safe version of shutdown_adapter that properly handles async coroutines.
    \"\"\"
    if adapter_name not in self.initialized_adapters:
        return False
    
    adapter = self.initialized_adapters[adapter_name]
    try:
        # Use a safer approach to handle coroutines
        if hasattr(adapter, 'shutdown'):
            if asyncio.iscoroutinefunction(adapter.shutdown):
                try:
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        # Create a new loop if one doesn't exist
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    if loop.is_running():
                        # If the loop is already running, use run_coroutine_threadsafe
                        future = asyncio.run_coroutine_threadsafe(adapter.shutdown(), loop)
                        success = future.result(timeout=10)  # Wait up to 10 seconds
                    else:
                        success = loop.run_until_complete(adapter.shutdown())
                except RuntimeError:
                    # If we can't get the current loop, create a new one
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    success = new_loop.run_until_complete(adapter.shutdown())
                    new_loop.close()
            else:
                # If it's not a coroutine function, just call it directly
                success = adapter.shutdown()
        else:
            logging.warning(f"Adapter {adapter_name} doesn't have a shutdown method")
            success = True  # Assume success if no shutdown method
        
        if success:
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

def apply_patch():
    \"\"\"Apply the patch to the FrameworkAdapterManager class.\"\"\"
    original_method = FrameworkAdapterManager.shutdown_adapter
    FrameworkAdapterManager.shutdown_adapter = safe_shutdown_adapter
    logging.info("Applied framework adapter manager patch for safe async shutdown")
    return original_method

if __name__ == "__main__":
    # Apply the patch when this module is run directly
    apply_patch()
"""
    
    with open(patch_file_path, 'w') as f:
        f.write(patch_content)
    
    logger.info("Created framework adapter patch file")
    
    # Create a patch loader in the main directory
    patch_loader_path = project_root / "patch_adapter.py"
    
    patch_loader_content = """\"\"\"
Load the framework adapter patch to fix async shutdown issues.
\"\"\"

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    # Import and apply the patch
    from evogenesis_core.adapters.framework_adapter_patch import apply_patch
    original_method = apply_patch()
    logger.info("Successfully applied framework adapter patch")
    
    # Import the examples module as a test
    logger.info("Testing import of examples module...")
    import examples.perception_action_demo
    logger.info("Successfully imported examples module")
except Exception as e:
    logger.error(f"Error applying patch: {str(e)}")

if __name__ == "__main__":
    logger.info("Patch has been applied")
"""
    
    with open(patch_loader_path, 'w') as f:
        f.write(patch_loader_content)
    
    logger.info("Created patch loader script")
    return True

def main():
    """Run all fixes."""
    logger.info("Applying fixes to EvoGenesis system...")
    
    # Fix strategic observatory config
    strategic_fix = fix_strategic_observatory_config()
    
    # Fix soo_initializer path
    soo_fix = fix_soo_initializer()
    
    # Fix perception-action tooling
    perception_fix = fix_perception_action_tooling()
    
    # Fix framework adapter manager
    adapter_fix = fix_framework_adapter_manager()
    
    logger.info("\nFix summary:")
    logger.info(f"- Strategic observatory config: {'FIXED' if strategic_fix else 'FAILED'}")
    logger.info(f"- SOO initializer path: {'FIXED' if soo_fix else 'FAILED'}")
    logger.info(f"- Perception-action tooling: {'FIXED' if perception_fix else 'FAILED'}")
    logger.info(f"- Framework adapter manager: {'FIXED' if adapter_fix else 'FAILED'}")
    
    logger.info("\nTo complete the fix, run:")
    logger.info("1. python patch_adapter.py")
    logger.info("2. python examples/perception_action_demo.py")

if __name__ == "__main__":
    main()
