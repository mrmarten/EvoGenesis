"""
Verification script for bug fixes.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def check_tools_directory():
    """Check if tools directory exists and is properly structured."""
    tools_dir = os.path.join(PROJECT_ROOT, "evogenesis_core", "tools")
    
    if not os.path.exists(tools_dir):
        logging.error(f"Tools directory not found: {tools_dir}")
        return False
    
    # Check subdirectories
    for subdir in ["builtin", "generated", "generated_remote"]:
        subdir_path = os.path.join(tools_dir, subdir)
        if not os.path.exists(subdir_path):
            logging.error(f"{subdir} directory not found")
            return False
            
    # Check for __init__.py files
    for path in [
        os.path.join(tools_dir, "__init__.py"),
        os.path.join(tools_dir, "builtin", "__init__.py"),
        os.path.join(tools_dir, "generated", "__init__.py"),
        os.path.join(tools_dir, "generated_remote", "__init__.py")
    ]:
        if not os.path.exists(path):
            logging.error(f"__init__.py file not found: {path}")
            return False
    
    logging.info("✓ Tools directory structure is correct")
    return True

def check_ws_handlers_indentation():
    """Check if WebSocket handlers indentation is fixed."""
    ws_handlers_init = os.path.join(
        PROJECT_ROOT, "evogenesis_core", "interfaces", "web_ui", 
        "ws_handlers", "__init__.py"
    )
    
    if not os.path.exists(ws_handlers_init):
        logging.error(f"WebSocket handlers file not found: {ws_handlers_init}")
        return False
        
    with open(ws_handlers_init, 'r') as f:
        content = f.read()
        
    if "        self.kernel = kernel" not in content:
        logging.error("WebSocket handlers indentation is still incorrect")
        return False
        
    logging.info("✓ WebSocket handlers indentation is correct")
    return True

def check_server_time_import():
    """Check if time module is properly imported in server.py."""
    server_py = os.path.join(
        PROJECT_ROOT, "evogenesis_core", "interfaces", "web_ui", "server.py"
    )
    
    if not os.path.exists(server_py):
        logging.error(f"Server.py file not found: {server_py}")
        return False
        
    with open(server_py, 'r') as f:
        content = f.read()
        
    if "import time" not in content:
        logging.error("Time module is not imported in server.py")
        return False
        
    logging.info("✓ Time module is properly imported in server.py")
    return True

def check_metrics_update_indentation():
    """Check if metrics_update_task indentation is fixed."""
    server_py = os.path.join(
        PROJECT_ROOT, "evogenesis_core", "interfaces", "web_ui", "server.py"
    )
    
    if not os.path.exists(server_py):
        logging.error(f"Server.py file not found: {server_py}")
        return False
        
    with open(server_py, 'r') as f:
        content = f.read()
        
    if '"""Background task to update system metrics and broadcast them."""    while' in content:
        logging.error("Metrics update task indentation is still incorrect")
        return False
        
    logging.info("✓ Metrics update task indentation is correct")
    return True

def check_psutil_installed():
    """Check if psutil is installed."""
    try:
        import psutil
        logging.info("✓ psutil is installed")
        return True
    except ImportError:
        logging.error("psutil is not installed")
        return False

def main():
    """Main verification function."""
    logging.info("Verifying bug fixes...")
    
    # Check all fixes
    tools_ok = check_tools_directory()
    ws_handlers_ok = check_ws_handlers_indentation()
    time_import_ok = check_server_time_import()
    metrics_task_ok = check_metrics_update_indentation()
    psutil_ok = check_psutil_installed()
    
    # Print summary
    logging.info("\n--- Verification Summary ---")
    logging.info(f"Tools directory structure: {'✓' if tools_ok else '✗'}")
    logging.info(f"WebSocket handlers indentation: {'✓' if ws_handlers_ok else '✗'}")
    logging.info(f"Server.py time import: {'✓' if time_import_ok else '✗'}")
    logging.info(f"Metrics update task indentation: {'✓' if metrics_task_ok else '✗'}")
    logging.info(f"psutil installation: {'✓' if psutil_ok else '✗'}")
    
    # Overall result
    all_ok = all([tools_ok, ws_handlers_ok, time_import_ok, metrics_task_ok, psutil_ok])
    if all_ok:
        logging.info("\n✅ All bugs have been fixed successfully!")
    else:
        logging.error("\n❌ Some bugs still need fixing")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
