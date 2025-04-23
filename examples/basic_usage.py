#!/usr/bin/env python
"""
Basic EvoGenesis Example

This example demonstrates how to initialize the EvoGenesis framework
and perform some basic operations with it.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evogenesis_core.kernel import Kernel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function demonstrating basic EvoGenesis usage."""
    # Initialize the kernel with default configuration
    kernel = Kernel(config_path="config/default.json")
    
    # Print basic system information
    print(f"EvoGenesis initialized with {len(kernel.get_active_modules())} active modules")
    print(f"System status: {kernel.get_status()}")
    
    # Example: Access the tooling system
    tooling_system = kernel.tooling_system
    print(f"Available tools: {len(tooling_system.get_tools())}")
    
    # Example: Access the LLM orchestrator
    llm_orchestrator = kernel.llm_orchestrator
    available_models = llm_orchestrator.get_available_models()
    print(f"Available LLM models: {len(available_models)}")
    
    # Example: Run a simple A/B test
    if hasattr(kernel, 'self_evolution_engine'):
        evolution_engine = kernel.self_evolution_engine
        test_id = evolution_engine.run_ab_test(
            feature="example_feature",
            version_a="current",
            version_b="experimental",
            duration_seconds=300,  # 5 minutes
            metrics=["latency", "error_rate"]
        )
        print(f"Started A/B test with ID: {test_id}")
    
    # Shut down the system
    kernel.shutdown()
    print("EvoGenesis system has been shut down")

if __name__ == "__main__":
    main()
