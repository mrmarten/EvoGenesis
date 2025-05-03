"""
Simple fix for strategic_observatory.json in the config directory.
"""
import os
import json
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.absolute()
config_dir = project_root / "config"
data_dir = project_root / "data" / "strategic_observatory"

# Create config directory if it doesn't exist
if not config_dir.exists():
    os.makedirs(config_dir)
    print(f"Created config directory: {config_dir}")

# Read the source JSON file
source_file = data_dir / "strategic_observatory.json"
if not source_file.exists():
    print(f"ERROR: Source file not found: {source_file}")
    sys.exit(1)

print(f"Reading source file: {source_file}")
with open(source_file, 'r') as f:
    data = json.load(f)
    print("Successfully loaded JSON data")

# Write to the target location
target_file = config_dir / "strategic_observatory.json"
print(f"Writing to target file: {target_file}")
with open(target_file, 'w') as f:
    json.dump(data, f, indent=2)
    print("Successfully wrote JSON data")

print("JSON file fix complete!")
