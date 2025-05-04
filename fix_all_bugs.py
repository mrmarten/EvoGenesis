"""
Automatic bug fixer for EvoGenesis project.

This script identifies and fixes the following bugs:
1. Missing tools directory
2. Indentation issues in WebSocket handlers
3. Missing time module import in server.py
4. Indentation issues in metrics update task
5. Missing psutil module
"""

import os
import sys
import shutil
import logging
import subprocess
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Define the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def ensure_package_installed(package_name):
    """Ensure a Python package is installed."""
    if importlib.util.find_spec(package_name) is None:
        logging.info(f"Installing {package_name} package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logging.info(f"{package_name} installed successfully")
    else:
        logging.info(f"{package_name} is already installed")

def fix_tools_directory():
    """Fix missing tools directory structure."""
    logging.info("Fixing tools directory structure...")
    
    tools_dir = os.path.join(PROJECT_ROOT, "evogenesis_core", "tools")
    os.makedirs(tools_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ["builtin", "generated", "generated_remote"]:
        os.makedirs(os.path.join(tools_dir, subdir), exist_ok=True)
    
    # Create __init__.py files
    init_files = [
        os.path.join(tools_dir, "__init__.py"),
        os.path.join(tools_dir, "builtin", "__init__.py"),
        os.path.join(tools_dir, "generated", "__init__.py"),
        os.path.join(tools_dir, "generated_remote", "__init__.py")
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('"""\nTools package for EvoGenesis framework.\n"""')
    
    logging.info("Tools directory structure fixed")

def fix_ws_handlers_indentation():
    """Fix indentation issues in WebSocket handlers __init__.py."""
    logging.info("Fixing WebSocket handlers indentation...")
    
    ws_handlers_init = os.path.join(
        PROJECT_ROOT, "evogenesis_core", "interfaces", "web_ui", 
        "ws_handlers", "__init__.py"
    )
    
    if os.path.exists(ws_handlers_init):
        with open(ws_handlers_init, 'r') as f:
            content = f.read()
        
        # Fix indentation issues - properly indent self.kernel = kernel
        if "        self.kernel = kernel" not in content:
            content = content.replace(
                '"""        self.kernel = kernel', 
                '"""\n        self.kernel = kernel'
            )
            
            with open(ws_handlers_init, 'w') as f:
                f.write(content)
            
            logging.info("WebSocket handlers indentation fixed")
        else:
            logging.info("WebSocket handlers indentation already correct")
    else:
        logging.warning(f"WebSocket handlers file not found: {ws_handlers_init}")

def fix_server_time_import():
    """Fix missing time import in server.py."""
    logging.info("Fixing server.py time import...")
    
    server_py = os.path.join(
        PROJECT_ROOT, "evogenesis_core", "interfaces", "web_ui", "server.py"
    )
    
    if os.path.exists(server_py):
        with open(server_py, 'r') as f:
            content = f.read()
        
        # Add time import if missing
        if "import time" not in content:
            content = content.replace(
                'import uuid\nimport traceback', 
                'import uuid\nimport time\nimport traceback'
            )
            
            with open(server_py, 'w') as f:
                f.write(content)
            
            logging.info("Server.py time import fixed")
        else:
            logging.info("Server.py time import already present")
    else:
        logging.warning(f"Server.py file not found: {server_py}")

def fix_metrics_update_indentation():
    """Fix indentation in metrics_update_task function."""
    logging.info("Fixing metrics_update_task indentation...")
    
    server_py = os.path.join(
        PROJECT_ROOT, "evogenesis_core", "interfaces", "web_ui", "server.py"
    )
    
    if os.path.exists(server_py):
        with open(server_py, 'r') as f:
            content = f.read()
        
        # Fix indentation in metrics_update_task
        if '"""Background task to update system metrics and broadcast them."""    while' in content:
            content = content.replace(
                '"""Background task to update system metrics and broadcast them."""    while', 
                '"""Background task to update system metrics and broadcast them."""\n    while'
            )
            
            # Also fix indentation of "# 1. Get agent metrics"
            content = content.replace(
                '                metrics = {}\n                  # 1. Get agent metrics', 
                '                metrics = {}\n                # 1. Get agent metrics'
            )
            
            with open(server_py, 'w') as f:
                f.write(content)
            
            logging.info("Metrics update task indentation fixed")
        else:
            logging.info("Metrics update task indentation already correct")
    else:
        logging.warning(f"Server.py file not found: {server_py}")

def main():
    """Main function to fix all bugs."""
    logging.info("Starting EvoGenesis bug fixer...")
    
    # Install required packages
    ensure_package_installed("psutil")
    
    # Fix missing tools directory
    fix_tools_directory()
    
    # Fix WebSocket handlers indentation
    fix_ws_handlers_indentation()
    
    # Fix server.py time import
    fix_server_time_import()
    
    # Fix metrics update indentation
    fix_metrics_update_indentation()
    
    logging.info("All bugs have been fixed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
