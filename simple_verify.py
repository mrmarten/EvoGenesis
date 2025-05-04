"""
Simplified verification script to check for fixes.
"""

import os
import sys
import logging

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Check tools directory
tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evogenesis_core", "tools")
if os.path.exists(tools_dir):
    logging.info(f"Tools directory exists: {tools_dir}")
    
    # Check subdirectories
    for subdir in ["builtin", "generated", "generated_remote"]:
        subdir_path = os.path.join(tools_dir, subdir)
        if os.path.exists(subdir_path):
            logging.info(f"  - {subdir} directory exists")
        else:
            logging.error(f"  - {subdir} directory is missing")
else:
    logging.error(f"Tools directory not found: {tools_dir}")

# Check for indentation in __init__.py
ws_handlers_init = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "evogenesis_core", "interfaces", "web_ui", "ws_handlers", "__init__.py")
if os.path.exists(ws_handlers_init):
    logging.info(f"WebSocket handlers __init__.py exists")
    
    # Check for indentation issues
    with open(ws_handlers_init, 'r') as f:
        content = f.read()
        if "        self.kernel = kernel" in content and "self._initialize_handlers()" in content:
            logging.info("  - Indentation appears to be fixed")
        else:
            logging.error("  - Indentation may still have issues")
else:
    logging.error(f"WebSocket handlers __init__.py not found")

# Check for time import in server.py
server_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                       "evogenesis_core", "interfaces", "web_ui", "server.py")
if os.path.exists(server_py):
    logging.info(f"Server.py exists")
    
    # Check for time import
    with open(server_py, 'r') as f:
        content = f.read()
        if "import time" in content:
            logging.info("  - 'import time' is present")
        else:
            logging.error("  - 'import time' is missing")
else:
    logging.error(f"Server.py not found")

logging.info("Verification complete!")
