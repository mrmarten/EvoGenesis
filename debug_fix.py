"""
Debug script for EvoGenesis - prints output directly to console
"""

import os
import sys
import json
from pathlib import Path

# Project paths
project_root = Path(__file__).parent.absolute()

def fix_strategic_observatory_json():
    """Fix the strategic_observatory.json file issue."""
    print("\n--- FIXING STRATEGIC OBSERVATORY JSON ---")
    
    # Source and target paths
    source_path = project_root / "data" / "strategic_observatory" / "strategic_observatory.json"
    target_path = project_root / "config" / "strategic_observatory.json"
    
    print(f"Source path: {source_path}")
    print(f"Target path: {target_path}")
    
    # Check if source exists
    if not source_path.exists():
        print(f"ERROR: Source file does not exist: {source_path}")
        return False
    
    # Read from source
    try:
        with open(source_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully read source file with {len(data)} sections")
    except Exception as e:
        print(f"ERROR reading source file: {e}")
        return False
    
    # Create config directory if needed
    if not target_path.parent.exists():
        os.makedirs(target_path.parent)
        print(f"Created directory: {target_path.parent}")
    
    # Write to target
    try:
        with open(target_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully wrote to target file: {target_path}")
    except Exception as e:
        print(f"ERROR writing target file: {e}")
        return False
    
    # Verify the file was properly created
    try:
        with open(target_path, 'r') as f:
            verify_data = json.load(f)
        print(f"Verification successful - file contains valid JSON with {len(verify_data)} sections")
        return True
    except Exception as e:
        print(f"ERROR verifying target file: {e}")
        return False

if __name__ == "__main__":
    print("\n=== EvoGenesis Debug Fix Script ===")
    result = fix_strategic_observatory_json()
    print(f"\nFix result: {'SUCCESS' if result else 'FAILURE'}")
