"""
Fix and run the perception_action_demo.py script.

This script applies all the necessary fixes to make the perception_action_demo.py
script work correctly, then runs it to demonstrate the fixes.
"""

import os
import sys
import asyncio
import subprocess
import importlib
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def fix_perception_action_module():
    """Fix the RemoteControlModule class."""
    print("Applying fixes to RemoteControlModule class...")
    
    # Check if the fix_perception_action.py script exists
    if os.path.exists("fix_perception_action.py"):
        # Import and run the fix
        try:
            fix_module = importlib.import_module("fix_perception_action")
            fix_module.apply_fixes()
            print("✅ Successfully applied fixes to RemoteControlModule")
            return True
        except Exception as e:
            print(f"❌ Error fixing RemoteControlModule: {str(e)}")
            return False
    else:
        print("❌ fix_perception_action.py not found")
        return False


def fix_demo_and_tooling():
    """Fix the demo script and tooling system."""
    print("Applying fixes to demo script and tooling system...")
    
    # Check if the fix_demo_script.py exists
    if os.path.exists("fix_demo_script.py"):
        # Import and run the fix
        try:
            fix_module = importlib.import_module("fix_demo_script")
            fix_module.fix_demo_script()
            fix_module.fix_tooling_system()
            print("✅ Successfully applied fixes to demo script and tooling system")
            return True
        except Exception as e:
            print(f"❌ Error fixing demo script and tooling system: {str(e)}")
            return False
    else:
        print("❌ fix_demo_script.py not found")
        return False


def fix_strategic_observatory():
    """Fix the strategic_observatory.json file."""
    print("Fixing strategic_observatory.json...")
    
    # Path to the strategic_observatory.json file
    json_path = os.path.join("config", "strategic_observatory.json")
    
    # Check if the file exists
    if not os.path.exists(json_path):
        # Create a basic valid JSON file
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            f.write('{\n  "enabled": true,\n  "config": {\n    "check_interval": 3600\n  }\n}')
        print(f"✅ Created {json_path} with default content")
    else:
        # Try to fix the existing file if it's empty or invalid
        try:
            with open(json_path, 'r') as f:
                content = f.read().strip()
            
            # If empty or invalid, replace with a valid JSON structure
            if not content:
                with open(json_path, 'w') as f:
                    f.write('{\n  "enabled": true,\n  "config": {\n    "check_interval": 3600\n  }\n}')
                print(f"✅ Fixed empty {json_path}")
            else:
                print(f"ℹ️ {json_path} seems to have content already")
        except Exception as e:
            # If there's an error, replace with a valid JSON structure
            with open(json_path, 'w') as f:
                f.write('{\n  "enabled": true,\n  "config": {\n    "check_interval": 3600\n  }\n}')
            print(f"✅ Fixed {json_path} after error: {str(e)}")
    
    return True


def patch_kernel():
    """Patch the kernel to register the remote_control module."""
    print("Patching kernel to register remote_control module...")
    
    # Path to the kernel.py file
    kernel_path = os.path.join("evogenesis_core", "kernel.py")
    
    # Check if the file exists
    if not os.path.exists(kernel_path):
        print(f"❌ Kernel file not found at {kernel_path}")
        return False
    
    # Read the current file contents
    with open(kernel_path, 'r') as f:
        content = f.read()
    
    # Check if the remote_control module is already registered
    if "from evogenesis_core.modules.perception_action_tooling import RemoteControlModule" not in content:
        # Add the import
        import_index = content.find("import logging")
        if import_index > 0:
            updated_content = content[:import_index + len("import logging")] + "\n" + \
                             "import importlib" + \
                             content[import_index + len("import logging"):]
            content = updated_content
        
        # Find the _initialize_modules method
        init_modules_index = content.find("    def _initialize_modules(self):")
        if init_modules_index > 0:
            # Add code to initialize the RemoteControlModule
            end_of_method = content.find("\n    def", init_modules_index)
            if end_of_method > 0:
                # Prepare the code to add
                code_to_add = """
        # Initialize RemoteControlModule if available
        try:
            from evogenesis_core.modules.perception_action_tooling import RemoteControlModule
            remote_control_module = RemoteControlModule(kernel=self)
            self.register_module("remote_control", remote_control_module)
            logging.info("Registered RemoteControlModule")
        except ImportError:
            logging.warning("RemoteControlModule not available")
        
"""
                # Insert the code
                updated_content = content[:end_of_method] + code_to_add + content[end_of_method:]
                content = updated_content
        
        # Write the updated content back to the file
        with open(kernel_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Patched {kernel_path} to register RemoteControlModule")
    else:
        print(f"ℹ️ RemoteControlModule already registered in kernel")
    
    return True


def run_demo():
    """Run the perception_action_demo.py script."""
    print("\n=== Running the perception_action_demo.py script ===\n")
    time.sleep(1)  # Small delay to make the output clearer
    
    # Path to the demo script
    demo_path = os.path.join("examples", "perception_action_demo.py")
    
    # Check if the file exists
    if not os.path.exists(demo_path):
        print(f"❌ Demo script not found at {demo_path}")
        return False
    
    # Run the script using subprocess
    try:
        result = subprocess.run([sys.executable, demo_path], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        
        # Print the output
        print(result.stdout)
        
        # Check for errors
        if result.returncode != 0:
            print("❌ Demo script returned an error:")
            print(result.stderr)
            return False
        
        print("✅ Demo script ran successfully")
        return True
    except Exception as e:
        print(f"❌ Error running demo script: {str(e)}")
        return False


if __name__ == "__main__":
    print("=== Fixing and running perception_action_demo.py ===\n")
    
    # Fix the strategic_observatory.json file
    if not fix_strategic_observatory():
        print("❌ Failed to fix strategic_observatory.json")
        sys.exit(1)
    
    # Patch the kernel
    if not patch_kernel():
        print("❌ Failed to patch kernel")
        sys.exit(1)
    
    # Fix the RemoteControlModule class
    if not fix_perception_action_module():
        print("❌ Failed to fix RemoteControlModule")
        sys.exit(1)
    
    # Fix the demo script and tooling system
    if not fix_demo_and_tooling():
        print("❌ Failed to fix demo script and tooling system")
        sys.exit(1)
    
    # Run the demo
    if not run_demo():
        print("❌ Failed to run demo successfully")
        sys.exit(1)
    
    print("\n=== All fixes applied and demo ran successfully ===")
