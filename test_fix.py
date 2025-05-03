"""
Test script to verify the fix_perception_action module is working correctly.
"""

import sys
import os

# Add the directory containing the fix_perception_action module to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from fix_perception_action import patch_tooling_system
    print("✅ Successfully imported patch_tooling_system function")
    
    # Try running the function
    result = patch_tooling_system()
    print(f"✅ Function returned: {result}")
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error running function: {e}")
