"""
Simple test to verify our fixes.
"""

import os
import sys

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Test 1: Check if tools directory exists
tools_dir = os.path.join(project_root, "evogenesis_core", "tools")
if os.path.exists(tools_dir):
    print("✓ Tools directory exists")
else:
    print("✗ Tools directory is missing")

# Test 2: Import time from server.py
try:
    from evogenesis_core.interfaces.web_ui.server import time
    print("✓ Time module imported successfully from server.py")
except ImportError:
    print("✗ Time module import failed from server.py")

# Test 3: Import WebSocketHandlersManager
try:
    from evogenesis_core.interfaces.web_ui.ws_handlers import WebSocketHandlersManager
    print("✓ WebSocketHandlersManager imported successfully")
except ImportError:
    print("✗ WebSocketHandlersManager import failed")

# Test 4: Import psutil
try:
    import psutil
    print("✓ psutil package is installed")
except ImportError:
    print("✗ psutil package is not installed")

print("\nAll tests completed!")
