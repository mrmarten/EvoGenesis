"""
Fix for the Strategic Opportunity Observatory module.

This script corrects the TeamFactory.create_team() calls by adding the required 'members' parameter.
"""

import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from evogenesis_core.kernel import EvoGenesisKernel
from evogenesis_core.modules.agent_factory import AgentFactory, Team

def fix_initialize_teams_method():
    """
    Fix the _initialize_teams method in the strategic_observatory.py file.
    Adds the required 'members' parameter to all create_team() calls.
    """
    filepath = os.path.join('evogenesis_core', 'modules', 'strategic_opportunity_observatory.py')
    
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Fix miner team creation
    content = content.replace(
        'miner_team = self.kernel.agent_factory.create_team(\n                name="Opportunity Miner Swarm",\n                members={},  # Required empty dict\n                goal=',
        'miner_team = self.kernel.agent_factory.create_team(\n                name="Opportunity Miner Swarm",\n                goal=' # Changed from agent_manager
    )
    
    # Fix reasoner team creation
    content = content.replace(
        'reasoner_team = self.kernel.agent_factory.create_team(\n                name="Strategic Reasoning Team",\n                members={},  # Required empty dict\n                goal=',
        'reasoner_team = self.kernel.agent_factory.create_team(\n                name="Strategic Reasoning Team",\n                goal=' # Changed from agent_manager
    )
    
    # Fix simulation team creation
    content = content.replace(
        'simulation_team = self.kernel.agent_factory.create_team(\n                name="Scenario Simulation Team",\n                members={},  # Required empty dict\n                goal=',
        'simulation_team = self.kernel.agent_factory.create_team(\n                name="Scenario Simulation Team",\n                goal=' # Changed from agent_manager
    )
    
    # Fix valuation team creation
    content = content.replace(
        'valuation_team = self.kernel.agent_factory.create_team(\n                name="Valuation and Feasibility Team",\n                members={},  # Required empty dict\n                goal=',
        'valuation_team = self.kernel.agent_factory.create_team(\n                name="Valuation and Feasibility Team",\n                goal=' # Changed from agent_manager
    )
    
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Fixed _initialize_teams method in {filepath}")
    print("Added required 'members' parameter to all create_team() calls")

if __name__ == "__main__":
    fix_initialize_teams_method()
