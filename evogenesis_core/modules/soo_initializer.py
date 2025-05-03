"""
Initialize the Strategic Opportunity Observatory with default sources and heuristics.

This script loads configuration from strategic_observatory.json and sets up
the initial signal sources and miner heuristics for the SOO module.
"""

import json
import os
import logging
from typing import Dict, Any, List

def initialize_observatory(observatory):
    """
    Initialize the Strategic Opportunity Observatory with default configuration.
    
    Args:
        observatory: The StrategicOpportunityObservatory instance
    """    # Load default configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "strategic_observatory.json")
    
    if not os.path.exists(config_path):
        logging.warning(f"Strategic Observatory config not found at {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing strategic_observatory.json: {str(e)}")
        return False
    
    # Update observatory configuration
    for section, settings in config.items():
        if section in observatory.config and isinstance(settings, dict):
            observatory.config[section].update(settings)
    
    # Add default signal sources
    source_map = {}  # name -> id mapping
    for source_config in config.get("default_signal_sources", []):
        name = source_config.get("name")
        if not name:
            continue
            
        source_type = source_config.get("source_type", "api")
        source_config_data = source_config.get("config", {})
        update_frequency = source_config.get("update_frequency", 3600)
        
        # Skip if already exists
        existing_sources = [s for s in observatory.signal_sources.values() if s.name == name]
        if existing_sources:
            source_map[name] = existing_sources[0].id
            continue
        
        # Add new source
        source_id = observatory.add_signal_source(
            name=name,
            source_type=source_type,
            config=source_config_data,
            update_frequency=update_frequency
        )
        source_map[name] = source_id
    
    # Add default heuristics
    for heuristic_config in config.get("default_heuristics", []):
        name = heuristic_config.get("name")
        if not name:
            continue
            
        # Skip if already exists
        existing_heuristics = [h for h in observatory.miner_heuristics.values() if h.name == name]
        if existing_heuristics:
            continue
        
        description = heuristic_config.get("description", "")
        prompt_template = heuristic_config.get("prompt_template", "")
        
        # Map signal source names to IDs
        signal_source_names = heuristic_config.get("signal_sources", [])
        signal_source_ids = []
        for source_name in signal_source_names:
            if source_name in source_map:
                signal_source_ids.append(source_map[source_name])
        
        # Add new heuristic
        observatory.add_miner_heuristic(
            name=name,
            description=description,
            prompt_template=prompt_template,
            signal_sources=signal_source_ids
        )
    
    logging.info(f"Initialized Strategic Opportunity Observatory with {len(source_map)} signal sources and {len(config.get('default_heuristics', []))} heuristics")
    return True
