"""
Test script to verify all bug fixes.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add necessary paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the kernel
from evogenesis_core.kernel import EvoGenesisKernel

def test_tools_directory():
    """Check if tools directory exists and is properly structured."""
    tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evogenesis_core", "tools")
    
    if not os.path.exists(tools_dir):
        logging.error(f"Tools directory not found: {tools_dir}")
        return False
        
    builtin_dir = os.path.join(tools_dir, "builtin")
    if not os.path.exists(builtin_dir):
        logging.error(f"Builtin tools directory not found: {builtin_dir}")
        return False
        
    generated_dir = os.path.join(tools_dir, "generated")
    if not os.path.exists(generated_dir):
        logging.error(f"Generated tools directory not found: {generated_dir}")
        return False
        
    remote_dir = os.path.join(tools_dir, "generated_remote")
    if not os.path.exists(remote_dir):
        logging.error(f"Remote tools directory not found: {remote_dir}")
        return False
        
    logging.info("Tools directory structure is correct")
    return True

def test_web_ui_handlers():
    """Test web UI handlers initialization."""
    try:
        from evogenesis_core.interfaces.web_ui.ws_handlers import WebSocketHandlersManager
        
        class MockKernel:
            pass
            
        class MockWSManager:
            pass
            
        manager = WebSocketHandlersManager(MockKernel(), MockWSManager())
        logging.info("WebSocket Handlers Manager initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize WebSocket Handlers Manager: {str(e)}")
        return False

def test_time_import():
    """Test if time module is properly imported in web_ui server."""
    try:
        from evogenesis_core.interfaces.web_ui.server import time
        logging.info("Time module is properly imported in web_ui server")
        return True
    except ImportError:
        logging.error("Time module is not imported in web_ui server")
        return False

def main():
    """Main test function."""
    logging.info("Starting verification tests")
    
    # Test tools directory
    tools_ok = test_tools_directory()
    
    # Test web UI handlers
    handlers_ok = test_web_ui_handlers()
    
    # Test time import
    time_ok = test_time_import()
    
    # Print summary
    logging.info("\n--- Test Summary ---")
    logging.info(f"Tools directory: {'✓' if tools_ok else '✗'}")
    logging.info(f"Web UI handlers: {'✓' if handlers_ok else '✗'}")
    logging.info(f"Time import: {'✓' if time_ok else '✗'}")
    
    if all([tools_ok, handlers_ok, time_ok]):
        logging.info("All bugs have been fixed successfully!")
        return 0
    else:
        logging.error("Some bugs still need fixing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
