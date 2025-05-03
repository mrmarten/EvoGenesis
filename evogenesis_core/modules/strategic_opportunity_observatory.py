"""
Strategic Opportunity Observatory (SOO) Module - Identifies unknown business opportunities, risks, and efficiencies.

This module serves as a meta-layer above the task-oriented EvoGenesis agents, continuously exploring
potential growth areas, efficiency improvements, and risk mitigations - even in spaces not yet imagined.
It transforms unknown-unknowns into ranked, decision-ready growth theses by leveraging specialized
agent teams and combining external signals with internal data.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import uuid
import time
import json
import logging
import asyncio
import threading
from enum import Enum
from datetime import datetime, timedelta
import yaml
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class OpportunityStatus(str, Enum):
    """Status of identified opportunities."""
    CANDIDATE = "candidate"           # Initial discovery, not yet validated
    UNDER_EVALUATION = "evaluating"   # Currently being evaluated
    VALIDATED = "validated"           # Has passed initial validation
    SIMULATED = "simulated"           # Has been through scenario simulation
    VALUED = "valued"                 # Has financial valuation attached
    APPROVED = "approved"             # Approved by governance team
    REJECTED = "rejected"             # Rejected by governance team
    ARCHIVED = "archived"             # No longer actively considered


class OpportunityType(str, Enum):
    """Types of strategic opportunities."""
    NEW_MARKET = "new_market"         # Expansion into new market segments
    NEW_PRODUCT = "new_product"       # Development of new product offerings
    EFFICIENCY = "efficiency"         # Internal efficiency improvements
    RISK_MITIGATION = "risk"          # Risk reduction opportunities
    PARTNERSHIP = "partnership"       # Strategic partnership possibilities
    ACQUISITION = "acquisition"       # Potential acquisition targets
    REGULATION = "regulation"         # Regulatory opportunity/compliance
    TECHNOLOGY = "technology"         # New technology adoption
    TALENT = "talent"                 # Talent/workforce opportunities
    OTHER = "other"                   # Other opportunity types


class ConfidenceLevel(str, Enum):
    """Confidence levels for opportunities."""
    SPECULATIVE = "speculative"       # <30% confidence
    PLAUSIBLE = "plausible"           # 30-50% confidence
    PROMISING = "promising"           # 50-70% confidence 
    PROBABLE = "probable"             # 70-90% confidence
    CERTAIN = "certain"               # >90% confidence


class Opportunity(BaseModel):
    """Represents a strategic opportunity identified by the observatory."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    opportunity_type: OpportunityType
    status: OpportunityStatus = OpportunityStatus.CANDIDATE
    confidence: ConfidenceLevel = ConfidenceLevel.SPECULATIVE
    discovered_at: float = Field(default_factory=time.time)
    discovered_by: str  # ID of the agent that discovered it
    
    # Business metrics
    estimated_tam: Optional[float] = None  # Total addressable market in USD
    estimated_growth_rate: Optional[float] = None  # Annual growth rate as decimal
    npv: Optional[float] = None  # Net present value
    required_investment: Optional[float] = None  # Required capital in USD
    time_to_market: Optional[int] = None  # Time to market in months
    time_to_breakeven: Optional[int] = None  # Time to break-even in months
    
    # Classification and scoring
    tags: List[str] = []
    impact_score: Optional[float] = None  # Estimated impact (0-1)
    feasibility_score: Optional[float] = None  # Estimated feasibility (0-1)
    risk_score: Optional[float] = None  # Estimated risk (0-1)
    combined_score: Optional[float] = None  # Combined score (0-1)
    
    # Supporting data
    evidence: List[Dict[str, Any]] = []  # Supporting evidence/citations
    assumptions: List[str] = []  # Key assumptions made
    required_capabilities: List[str] = []  # Capabilities needed to pursue
    related_opportunities: List[str] = []  # IDs of related opportunities
    
    # Governance and feedback
    governance_votes: Dict[str, bool] = {}  # user_id -> vote (approve/reject)
    governance_comments: List[Dict[str, Any]] = []  # Feedback comments
    
    # Simulation results
    simulation_results: Dict[str, Any] = {}  # Results from scenario simulations
    
    def calculate_combined_score(self):
        """Calculate a combined score based on impact, feasibility, and risk."""
        if self.impact_score is not None and self.feasibility_score is not None and self.risk_score is not None:
            # Higher impact and feasibility are good, higher risk is bad
            self.combined_score = (
                (self.impact_score * 0.5) + 
                (self.feasibility_score * 0.3) - 
                (self.risk_score * 0.2)
            )
            return self.combined_score
        return None
    
    def add_evidence(self, source: str, content: str, url: Optional[str] = None):
        """Add a piece of evidence supporting this opportunity."""
        self.evidence.append({
            "source": source,
            "content": content,
            "url": url,
            "added_at": time.time()
        })
    
    def add_governance_comment(self, user_id: str, comment: str, rating: Optional[int] = None):
        """Add a governance comment or rating to this opportunity."""
        self.governance_comments.append({
            "user_id": user_id,
            "comment": comment,
            "rating": rating,
            "timestamp": time.time()
        })
    
    def record_governance_vote(self, user_id: str, approve: bool):
        """Record a governance vote (approve/reject) for this opportunity."""
        self.governance_votes[user_id] = approve
        
        # Update status if threshold is reached (e.g., 3 or more votes in same direction)
        approve_count = sum(1 for vote in self.governance_votes.values() if vote)
        reject_count = len(self.governance_votes) - approve_count
        
        if approve_count >= 3 and approve_count > reject_count:
            self.status = OpportunityStatus.APPROVED
        elif reject_count >= 3 and reject_count > approve_count:
            self.status = OpportunityStatus.REJECTED


class SignalSource(BaseModel):
    """Configuration for a data signal source."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    source_type: str  # api, file, database, web, etc.
    config: Dict[str, Any]  # Source-specific configuration
    update_frequency: int  # Update frequency in seconds
    last_update: Optional[float] = None
    enabled: bool = True
    
    # Statistics
    total_signals: int = 0
    total_updates: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


class MinerHeuristic(BaseModel):
    """A heuristic used by opportunity miner agents."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    prompt_template: str
    signal_sources: List[str]  # IDs of required signal sources
    success_count: int = 0
    failure_count: int = 0
    last_success: Optional[float] = None
    created_at: float = Field(default_factory=time.time)
    modified_at: float = Field(default_factory=time.time)
    
    # Evolution tracking
    parent_id: Optional[str] = None  # ID of parent heuristic if evolved
    generation: int = 1
    mutation_factor: float = 0.0  # How much this deviates from parent
    
    def calculate_success_rate(self) -> float:
        """Calculate the success rate of this heuristic."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


class Signal(BaseModel):
    """Represents a data signal ingested from an external source."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str  # ID of the signal source
    content: Any  # The actual signal content
    signal_type: Optional[str] = None  # Type of signal
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional metadata
    processed: bool = False  # Whether this signal has been processed
    created_at: float = Field(default_factory=time.time)  # When the signal was created


class StrategicOpportunityObservatory:
    """
    Strategic Opportunity Observatory (SOO) for the EvoGenesis framework.
    
    Responsible for:
    - Ingesting and processing external signals
    - Coordinating opportunity miner agents
    - Evaluating, simulating, and valuing opportunities
    - Managing the governance process
    - Evolving mining heuristics based on feedback
    """
    
    def __init__(self, kernel):
        """
        Initialize the Strategic Opportunity Observatory.
        
        Args:
            kernel: The EvoGenesis kernel instance
        """
        self.kernel = kernel
        self.opportunities = {}  # id -> Opportunity
        self.signal_sources = {}  # id -> SignalSource
        self.miner_heuristics = {}  # id -> MinerHeuristic
        
        # Team tracking
        self.miner_team_id = None  # ID of the miner agent team
        self.reasoning_team_id = None  # ID of the strategic reasoning team
        self.simulation_team_id = None  # ID of the scenario simulation team
        self.valuation_team_id = None  # ID of the valuation team
        
        # Performance metrics
        self.metrics = {
            "total_opportunities_discovered": 0,
            "approved_opportunities": 0,
            "rejected_opportunities": 0,
            "avg_evaluation_time": 0.0,
            "opportunity_discovery_rate": 0.0,  # per day
            "last_discovery": None,
            "opportunity_success_rate": 0.0,  # % of approved vs total evaluated
        }
        
        # Shared context
        self.global_context = {
            "market_assumptions": {},
            "regulatory_context": {},
            "competitive_landscape": {},
            "technology_trends": {},
            "macroeconomic_indicators": {}
        }
        
        # Configuration
        self.config = {
            "signals": {
                "max_signal_age_days": 30,
                "min_update_frequency": 3600,  # seconds
                "batch_size": 100
            },
            "miners": {
                "min_confidence_threshold": 0.4,
                "max_miners_per_heuristic": 3,
                "max_concurrent_miners": 50
            },
            "reasoning": {
                "consolidation_similarity_threshold": 0.75,
                "min_evidence_count": 3
            },
            "simulation": {
                "monte_carlo_iterations": 1000,
                "sensitivity_variables": ["market_growth", "competition", "adoption_rate"]
            },
            "valuation": {
                "discount_rate": 0.1,
                "projection_years": 5,
                "terminal_growth_rate": 0.02
            },
            "governance": {
                "approval_threshold": 0.6,  # 60% approval needed
                "min_votes_required": 3
            },
            "evolution": {
                "evolution_cycle_days": 7,
                "mutation_rate": 0.1,
                "selection_pressure": 0.7,
                "population_size": 50
            }
        }
        
        # Persistence
        self.data_path = os.path.join("data", "strategic_observatory")
        os.makedirs(self.data_path, exist_ok=True)
        
        # Background tasks
        self.running = False
        self.signal_ingest_thread = None
        self.miner_coordination_thread = None
        self.reasoning_thread = None
        
        # Load existing configuration and data
        self._load_data()

    def start(self):
        """Start the Strategic Opportunity Observatory."""
        self.running = True
        
        # Initialize from configuration
        from evogenesis_core.modules.soo_initializer import initialize_observatory
        initialize_observatory(self)
        
        # Initialize component teams if they don't exist
        self._initialize_teams()
        
        # Start background signal ingestion
        self.signal_ingest_thread = threading.Thread(
            target=self._signal_ingest_loop,
            daemon=True
        )
        self.signal_ingest_thread.start()
        
        # Start miner coordination
        self.miner_coordination_thread = threading.Thread(
            target=self._miner_coordination_loop,
            daemon=True
        )
        self.miner_coordination_thread.start()
        
        # Start reasoning and evaluation
        self.reasoning_thread = threading.Thread(
            target=self._reasoning_loop,
            daemon=True
        )
        self.reasoning_thread.start()
        
        logging.info("Strategic Opportunity Observatory started")
    
    def stop(self):
        """Stop the Strategic Opportunity Observatory."""
        self.running = False
        
        # Save current state
        self._save_data()
        
        logging.info("Strategic Opportunity Observatory stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Strategic Opportunity Observatory.
        
        Returns:
            Dictionary with status information
        """
        # Count opportunities by status
        status_counts = {}
        for status in OpportunityStatus:
            status_counts[status.value] = sum(
                1 for opp in self.opportunities.values() if opp.status == status
            )
        
        # Get miner statistics
        active_miners = 0
        if self.miner_team_id and hasattr(self.kernel, "agent_manager"):
            team = self.kernel.agent_manager.teams.get(self.miner_team_id)
            if team:
                active_miners = len(team.agents)
        
        return {
            "active": self.running,
            "opportunities": {
                "total": len(self.opportunities),
                "by_status": status_counts,
                "by_type": self._count_by_type()
            },
            "signals": {
                "sources": len(self.signal_sources),
                "last_update": max([s.last_update or 0 for s in self.signal_sources.values()], default=0)
            },
            "miners": {
                "heuristics": len(self.miner_heuristics),
                "active_miners": active_miners
            },
            "metrics": self.metrics
        }
    
    def add_signal_source(self, name: str, source_type: str, config: Dict[str, Any], 
                          update_frequency: int = 3600) -> str:
        """
        Add a new signal source to ingest data from.
        
        Args:
            name: Name of the signal source
            source_type: Type of source (api, file, database, web, etc.)
            config: Source-specific configuration
            update_frequency: Update frequency in seconds
            
        Returns:
            ID of the created signal source
        """
        source = SignalSource(
            name=name,
            source_type=source_type,
            config=config,
            update_frequency=update_frequency
        )
        
        self.signal_sources[source.id] = source
        self._save_signal_sources()
        
        logging.info(f"Added signal source: {name} (type: {source_type})")
        return source.id
    
    def add_miner_heuristic(self, name: str, description: str, prompt_template: str,
                           signal_sources: List[str]) -> str:
        """
        Add a new opportunity miner heuristic.
        
        Args:
            name: Name of the heuristic
            description: Description of what the heuristic looks for
            prompt_template: Prompt template for the miner agent
            signal_sources: List of signal source IDs this heuristic requires
            
        Returns:
            ID of the created heuristic
        """
        heuristic = MinerHeuristic(
            name=name,
            description=description,
            prompt_template=prompt_template,
            signal_sources=signal_sources
        )
        
        self.miner_heuristics[heuristic.id] = heuristic
        self._save_miner_heuristics()
        
        # Create miner agents for this heuristic
        self._create_miner_agents_for_heuristic(heuristic.id)
        
        logging.info(f"Added miner heuristic: {name}")
        return heuristic.id
    
    def get_opportunities(self, status: Optional[OpportunityStatus] = None, 
                         opportunity_type: Optional[OpportunityType] = None,
                         min_confidence: Optional[ConfidenceLevel] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get opportunities, optionally filtered by status, type and confidence.
        
        Args:
            status: Optional status to filter by
            opportunity_type: Optional type to filter by
            min_confidence: Optional minimum confidence level
            limit: Maximum number of opportunities to return
            
        Returns:
            List of opportunity dictionaries
        """
        filtered = self.opportunities.values()
        
        if status:
            filtered = [o for o in filtered if o.status == status]
            
        if opportunity_type:
            filtered = [o for o in filtered if o.opportunity_type == opportunity_type]
            
        if min_confidence:
            confidence_values = list(ConfidenceLevel)
            min_index = confidence_values.index(min_confidence)
            filtered = [o for o in filtered if confidence_values.index(o.confidence) >= min_index]
        
        # Sort by combined score (if available) then by discovery date
        filtered = sorted(
            filtered, 
            key=lambda o: (o.combined_score or 0, o.discovered_at),
            reverse=True
        )
        
        # Return limited number as dictionaries
        return [o.dict() for o in filtered[:limit]]
    
    def update_opportunity(self, opportunity_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an opportunity with new data.
        
        Args:
            opportunity_id: ID of the opportunity to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        if opportunity_id not in self.opportunities:
            return False
        
        opportunity = self.opportunities[opportunity_id]
        
        # Special handling for certain fields
        if "evidence" in updates:
            # Add new evidence items instead of replacing
            for evidence in updates["evidence"]:
                opportunity.add_evidence(
                    source=evidence.get("source", "unknown"),
                    content=evidence.get("content", ""),
                    url=evidence.get("url")
                )
            del updates["evidence"]
        
        if "governance_comments" in updates:
            # Add new comments instead of replacing
            for comment in updates["governance_comments"]:
                opportunity.add_governance_comment(
                    user_id=comment.get("user_id", "unknown"),
                    comment=comment.get("comment", ""),
                    rating=comment.get("rating")
                )
            del updates["governance_comments"]
            
        if "governance_votes" in updates:
            # Process votes
            for user_id, vote in updates["governance_votes"].items():
                opportunity.record_governance_vote(user_id, vote)
            del updates["governance_votes"]
            
        # Update remaining fields
        for key, value in updates.items():
            if hasattr(opportunity, key):
                setattr(opportunity, key, value)
        
        # Recalculate combined score if impact, feasibility or risk was updated
        score_components = ["impact_score", "feasibility_score", "risk_score"]
        if any(field in updates for field in score_components):
            opportunity.calculate_combined_score()
        
        self._save_opportunities()
        return True
    
    def record_opportunity_feedback(self, opportunity_id: str, approved: bool, 
                                 user_id: str, feedback: str) -> bool:
        """
        Record feedback on an opportunity, affecting its status and heuristic evolution.
        
        Args:
            opportunity_id: ID of the opportunity
            approved: Whether the opportunity was approved
            user_id: ID of the user providing feedback
            feedback: Feedback comments
            
        Returns:
            True if successful, False otherwise
        """
        if opportunity_id not in self.opportunities:
            return False
        
        opportunity = self.opportunities[opportunity_id]
        
        # Record the governance vote and comment
        opportunity.record_governance_vote(user_id, approved)
        opportunity.add_governance_comment(user_id, feedback)
        
        # Update metrics
        if opportunity.status == OpportunityStatus.APPROVED:
            self.metrics["approved_opportunities"] += 1
        elif opportunity.status == OpportunityStatus.REJECTED:
            self.metrics["rejected_opportunities"] += 1
        
        if approved != (opportunity.status == OpportunityStatus.APPROVED):
            # There's a mismatch between the vote and the current status
            # This happens when we don't have enough votes yet
            return True
        
        # Update heuristic success/failure counts
        if opportunity.discovered_by:
            for heuristic in self.miner_heuristics.values():
                miner_agents = self._get_miners_for_heuristic(heuristic.id)
                if opportunity.discovered_by in miner_agents:
                    if approved:
                        heuristic.success_count += 1
                        heuristic.last_success = time.time()
                    else:
                        heuristic.failure_count += 1
                    self._save_miner_heuristics()
                    break
        
        self._save_opportunities()
        self._update_metrics()
        return True
    
    def submit_opportunity_candidate(self, title: str, description: str, opportunity_type: str,
                                  discovered_by: str, evidence: List[Dict[str, Any]],
                                  tags: List[str] = None) -> str:
        """
        Submit a candidate opportunity discovered by a miner agent.
        
        Args:
            title: Title of the opportunity
            description: Description of the opportunity
            opportunity_type: Type of opportunity (must be a valid OpportunityType)
            discovered_by: ID of the agent that discovered it
            evidence: List of evidence items supporting the opportunity
            tags: Optional tags to categorize the opportunity
            
        Returns:
            ID of the created opportunity
        """
        # Validate opportunity type
        try:
            opp_type = OpportunityType(opportunity_type)
        except ValueError:
            opp_type = OpportunityType.OTHER
            
        # Create opportunity
        opportunity = Opportunity(
            title=title,
            description=description,
            opportunity_type=opp_type,
            discovered_by=discovered_by,
            tags=tags or []
        )
        
        # Add evidence
        for item in evidence:
            opportunity.add_evidence(
                source=item.get("source", "unknown"),
                content=item.get("content", ""),
                url=item.get("url")
            )
        
        # Store opportunity
        self.opportunities[opportunity.id] = opportunity
        self.metrics["total_opportunities_discovered"] += 1
        self.metrics["last_discovery"] = time.time()
        
        # Update discovery rate
        discovery_period = 86400 * 7  # 7 days in seconds
        recent_discoveries = sum(
            1 for opp in self.opportunities.values() 
            if time.time() - opp.discovered_at < discovery_period
        )
        self.metrics["opportunity_discovery_rate"] = recent_discoveries / 7  # per day
        
        self._save_opportunities()
        self._update_metrics()
        
        # Trigger evaluation process
        self._queue_for_evaluation(opportunity.id)
        
        logging.info(f"New opportunity candidate submitted: {title} (ID: {opportunity.id})")
        return opportunity.id
    
    def evolve_heuristics(self, force: bool = False) -> int:
        """
        Evolve miner heuristics based on feedback and success rates.
        
        Args:
            force: Whether to force evolution regardless of timing
            
        Returns:
            Number of new heuristics created
        """
        # Check if it's time for evolution
        last_evolution = self.metrics.get("last_evolution_time", 0)
        evolution_interval = self.config["evolution"]["evolution_cycle_days"] * 86400
        
        if not force and time.time() - last_evolution < evolution_interval:
            return 0
        
        # Select successful heuristics to reproduce
        heuristics = list(self.miner_heuristics.values())
        if not heuristics:
            return 0
            
        # Sort by success rate
        heuristics.sort(key=lambda h: h.calculate_success_rate(), reverse=True)
        
        # Selection pressure determines how many top heuristics we use
        selection_count = max(2, int(len(heuristics) * self.config["evolution"]["selection_pressure"]))
        parent_heuristics = heuristics[:selection_count]
        
        # Create new generation with mutations
        new_heuristics = []
        desired_population = self.config["evolution"]["population_size"]
        
        # Keep best performers, replace others
        to_create = desired_population - len(parent_heuristics)
        
        for i in range(to_create):
            # Select a parent, with probability weighted by success rate
            weights = [h.calculate_success_rate() + 0.1 for h in parent_heuristics]  # Add 0.1 to avoid zero weights
            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1] * len(parent_heuristics)  # Equal weights if all zero
                
            norm_weights = [w / sum(weights) for w in weights]
            parent = np.random.choice(parent_heuristics, p=norm_weights)
            
            # Create mutation
            mutation_factor = self.config["evolution"]["mutation_rate"] * np.random.random()
            
            # Create new heuristic based on parent
            new_heuristic = MinerHeuristic(
                name=f"{parent.name} Gen{parent.generation+1}",
                description=parent.description,
                prompt_template=self._mutate_prompt(parent.prompt_template, mutation_factor),
                signal_sources=parent.signal_sources.copy(),
                parent_id=parent.id,
                generation=parent.generation + 1,
                mutation_factor=mutation_factor
            )
            
            self.miner_heuristics[new_heuristic.id] = new_heuristic
            new_heuristics.append(new_heuristic)
        
        # Update metrics
        self.metrics["last_evolution_time"] = time.time()
        self._save_miner_heuristics()
        self._update_metrics()
        
        # Create agents for new heuristics
        for heuristic in new_heuristics:
            self._create_miner_agents_for_heuristic(heuristic.id)
        
        logging.info(f"Evolved {len(new_heuristics)} new heuristics")
        return len(new_heuristics)
    
    def run_scenario_simulation(self, opportunity_id: str) -> Dict[str, Any]:
        """
        Run a scenario simulation for an opportunity to test robustness.
        
        Args:
            opportunity_id: ID of the opportunity to simulate
            
        Returns:
            Dictionary with simulation results
        """
        if opportunity_id not in self.opportunities:
            return {"error": "Opportunity not found"}
        
        opportunity = self.opportunities[opportunity_id]
        
        # Queue for simulation by the simulation team
        if hasattr(self.kernel, "task_planner"):
            self.kernel.task_planner.create_task(
                title=f"Simulate scenarios for opportunity: {opportunity.title}",
                description=f"Run Monte Carlo simulations with {self.config['simulation']['monte_carlo_iterations']} iterations to test the robustness of opportunity {opportunity_id}",
                priority="medium",
                team_id=self.simulation_team_id,
                metadata={
                    "action": "simulate_opportunity",
                    "opportunity_id": opportunity_id,
                    "iterations": self.config["simulation"]["monte_carlo_iterations"],
                    "variables": self.config["simulation"]["sensitivity_variables"]
                }
            )
        
        return {
            "status": "simulation_queued",
            "opportunity_id": opportunity_id,
            "estimated_completion_time": time.time() + 3600  # Estimate 1 hour
        }
    
    def perform_valuation(self, opportunity_id: str) -> Dict[str, Any]:
        """
        Perform financial valuation for an opportunity.
        
        Args:
            opportunity_id: ID of the opportunity to value
            
        Returns:
            Dictionary with valuation results
        """
        if opportunity_id not in self.opportunities:
            return {"error": "Opportunity not found"}
        
        opportunity = self.opportunities[opportunity_id]
        
        # Queue for valuation by the valuation team
        if hasattr(self.kernel, "task_planner"):
            self.kernel.task_planner.create_task(
                title=f"Perform valuation for opportunity: {opportunity.title}",
                description=f"Calculate NPV, required investment, and other financial metrics for opportunity {opportunity_id}",
                priority="medium",
                team_id=self.valuation_team_id,
                metadata={
                    "action": "value_opportunity",
                    "opportunity_id": opportunity_id,
                    "discount_rate": self.config["valuation"]["discount_rate"],
                    "projection_years": self.config["valuation"]["projection_years"],
                    "terminal_growth_rate": self.config["valuation"]["terminal_growth_rate"]
                }
            )
        
        return {
            "status": "valuation_queued",
            "opportunity_id": opportunity_id,
            "estimated_completion_time": time.time() + 1800  # Estimate 30 minutes
        }
    
    def _initialize_teams(self):
        """Initialize the component teams for the Strategic Opportunity Observatory."""
        if not hasattr(self.kernel, "agent_manager"):
            logging.warning("Agent Manager not available, teams will not be created")
            return

        # Create Opportunity Miner Team if it doesn't exist
        if not self.miner_team_id:
            miner_team = self.kernel.agent_manager.create_team(
                name="Opportunity Miner Swarm",
                members={},  # Required empty dict, will add members later
                goal="Discover potential strategic opportunities from diverse signal sources",
                agent_roles={}  # Agents will be added based on heuristics
            )
            self.miner_team_id = miner_team.team_id

        # Create Strategic Reasoner Team if it doesn't exist
        if not self.reasoning_team_id:
            reasoner_team = self.kernel.agent_manager.create_team(
                name="Strategic Reasoning Team",
                members={},  # Required empty dict, will add members later
                goal="Evaluate, refine, and synthesize opportunity candidates into actionable recommendations",
                agent_roles={
                    "lead_reasoner": {
                        "type": "planner",
                        "capabilities": ["strategic_analysis", "critical_thinking", "pattern_recognition"]
                    },
                    "narrative_builder": {
                        "type": "researcher",
                        "capabilities": ["causal_inference", "storytelling", "market_analysis"]
                    },
                    "evidence_analyst": {
                        "type": "critic",
                        "capabilities": ["data_analysis", "fact_checking", "evidence_evaluation"]
                    }
                }
            )
            self.reasoning_team_id = reasoner_team.team_id

        # Create Scenario Simulation Team if it doesn't exist
        if not self.simulation_team_id:
            simulation_team = self.kernel.agent_manager.create_team(
                name="Scenario Simulation Team",
                members={},  # Required empty dict, will add members later
                goal="Simulate various scenarios to stress-test opportunity hypotheses",
                agent_roles={
                    "simulation_lead": {
                        "type": "planner",
                        "capabilities": ["scenario_planning", "monte_carlo_simulation", "stress_testing"]
                    },
                    "market_modeler": {
                        "type": "researcher",
                        "capabilities": ["market_modeling", "competitive_analysis", "trend_forecasting"]
                    },
                    "risk_analyst": {
                        "type": "critic",
                        "capabilities": ["risk_analysis", "sensitivity_analysis", "uncertainty_quantification"]
                    }
                }
            )
            self.simulation_team_id = simulation_team.team_id

        # Create Valuation Team if it doesn't exist
        if not self.valuation_team_id:
            valuation_team = self.kernel.agent_manager.create_team(
                name="Valuation and Feasibility Team",
                members={},  # Required empty dict, will add members later
                goal="Evaluate financial value and implementation feasibility of opportunities",
                agent_roles={
                    "financial_analyst": {
                        "type": "researcher",
                        "capabilities": ["financial_modeling", "npv_calculation", "market_sizing"]
                    },
                    "capability_analyst": {
                        "type": "critic",
                        "capabilities": ["capability_gap_analysis", "resource_assessment", "implementation_planning"]
                    },
                    "value_optimizer": {
                        "type": "planner",
                        "capabilities": ["portfolio_optimization", "resource_allocation", "value_maximization"]
                    }
                }
            )
            self.valuation_team_id = valuation_team.team_id
    
    def _load_data(self):
        """Load persistent data from disk."""
        self._load_config()
        self._load_opportunities()
        self._load_signal_sources()
        self._load_miner_heuristics()
        self._load_metrics()
    
    def _save_data(self):
        """Save all data to disk."""
        self._save_config()
        self._save_opportunities()
        self._save_signal_sources()
        self._save_miner_heuristics()
        self._save_metrics()
    
    def _load_config(self):
        """Load configuration from disk."""
        config_path = os.path.join(self.data_path, "config.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        self.config.update(loaded_config)
            except Exception as e:
                logging.error(f"Error loading SOO configuration: {str(e)}")
    
    def _save_config(self):
        """Save configuration to disk."""
        config_path = os.path.join(self.data_path, "config.yaml")
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)
        except Exception as e:
            logging.error(f"Error saving SOO configuration: {str(e)}")
    
    def _load_opportunities(self):
        """Load opportunities from disk."""
        opps_path = os.path.join(self.data_path, "opportunities.json")
        if os.path.exists(opps_path):
            try:
                with open(opps_path, 'r') as f:
                    opps_data = json.load(f)
                    for opp_dict in opps_data:
                        try:
                            opp = Opportunity(**opp_dict)
                            self.opportunities[opp.id] = opp
                        except Exception as e:
                            logging.error(f"Error parsing opportunity: {str(e)}")
            except Exception as e:
                logging.error(f"Error loading opportunities: {str(e)}")
    
    def _save_opportunities(self):
        """Save opportunities to disk."""
        opps_path = os.path.join(self.data_path, "opportunities.json")
        try:
            with open(opps_path, 'w') as f:
                json.dump([opp.dict() for opp in self.opportunities.values()], f)
        except Exception as e:
            logging.error(f"Error saving opportunities: {str(e)}")
    
    def _load_signal_sources(self):
        """Load signal sources from disk."""
        sources_path = os.path.join(self.data_path, "signal_sources.json")
        if os.path.exists(sources_path):
            try:
                with open(sources_path, 'r') as f:
                    sources_data = json.load(f)
                    for source_dict in sources_data:
                        try:
                            source = SignalSource(**source_dict)
                            self.signal_sources[source.id] = source
                        except Exception as e:
                            logging.error(f"Error parsing signal source: {str(e)}")
            except Exception as e:
                logging.error(f"Error loading signal sources: {str(e)}")
    
    def _save_signal_sources(self):
        """Save signal sources to disk."""
        sources_path = os.path.join(self.data_path, "signal_sources.json")
        try:
            with open(sources_path, 'w') as f:
                json.dump([source.dict() for source in self.signal_sources.values()], f)
        except Exception as e:
            logging.error(f"Error saving signal sources: {str(e)}")
    
    def _load_miner_heuristics(self):
        """Load miner heuristics from disk."""
        heuristics_path = os.path.join(self.data_path, "miner_heuristics.json")
        if os.path.exists(heuristics_path):
            try:
                with open(heuristics_path, 'r') as f:
                    heuristics_data = json.load(f)
                    for heuristic_dict in heuristics_data:
                        try:
                            heuristic = MinerHeuristic(**heuristic_dict)
                            self.miner_heuristics[heuristic.id] = heuristic
                        except Exception as e:
                            logging.error(f"Error parsing miner heuristic: {str(e)}")
            except Exception as e:
                logging.error(f"Error loading miner heuristics: {str(e)}")
    
    def _save_miner_heuristics(self):
        """Save miner heuristics to disk."""
        heuristics_path = os.path.join(self.data_path, "miner_heuristics.json")
        try:
            with open(heuristics_path, 'w') as f:
                json.dump([h.dict() for h in self.miner_heuristics.values()], f)
        except Exception as e:
            logging.error(f"Error saving miner heuristics: {str(e)}")
    
    def _load_metrics(self):
        """Load metrics from disk."""
        metrics_path = os.path.join(self.data_path, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    loaded_metrics = json.load(f)
                    if loaded_metrics:
                        self.metrics.update(loaded_metrics)
            except Exception as e:
                logging.error(f"Error loading SOO metrics: {str(e)}")
    
    def _save_metrics(self):
        """Save metrics to disk."""
        metrics_path = os.path.join(self.data_path, "metrics.json")
        try:
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f)
        except Exception as e:
            logging.error(f"Error saving SOO metrics: {str(e)}")
    
    def _update_metrics(self):
        """Update and save the metrics."""
        total_evaluated = self.metrics["approved_opportunities"] + self.metrics["rejected_opportunities"]
        self.metrics["opportunity_success_rate"] = (
            self.metrics["approved_opportunities"] / total_evaluated
            if total_evaluated > 0 else 0.0
        )
        
        self._save_metrics()
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count opportunities by type."""
        type_counts = {}
        for type_enum in OpportunityType:
            type_counts[type_enum.value] = sum(
                1 for opp in self.opportunities.values() 
                if opp.opportunity_type == type_enum
            )
        return type_counts
    
    def _signal_ingest_loop(self):
        """Background loop for ingesting signals from sources."""
        while self.running:
            try:
                # Process each signal source
                for source_id, source in list(self.signal_sources.items()):
                    if not source.enabled:
                        continue
                        
                    current_time = time.time()
                    if (source.last_update is None or 
                        current_time - source.last_update >= source.update_frequency):
                        
                        try:
                            self._ingest_from_source(source_id)
                            source.last_update = current_time
                            source.total_updates += 1
                        except Exception as e:
                            source.error_count += 1
                            source.last_error = str(e)
                            logging.error(f"Error ingesting from source {source.name}: {str(e)}")
                
                # Save updated sources
                self._save_signal_sources()
                
                # Sleep to prevent CPU overuse
                time.sleep(10)
                
            except Exception as e:
                logging.error(f"Error in signal ingest loop: {str(e)}")
                time.sleep(30)  # Longer sleep on error
    
    def _ingest_from_source(self, source_id: str):
        """
        Ingest data from a specific signal source.
        
        Args:
            source_id: ID of the source to ingest from
        """
        if source_id not in self.signal_sources:
            return
            
        source = self.signal_sources[source_id]
        
        if source.source_type == "api":
            self._ingest_from_api(source)
        elif source.source_type == "file":
            self._ingest_from_file(source)
        elif source.source_type == "database":
            self._ingest_from_database(source)
        elif source.source_type == "web":
            self._ingest_from_web(source)
        else:
            logging.warning(f"Unknown source type: {source.source_type}")
    
    def _ingest_from_api(self, source: SignalSource):
        """Ingest signals from an API source."""
        try:
            import requests
            from datetime import datetime
            
            # Get API configuration
            api_config = source.config.get("api", {})
            url = api_config.get("url")
            method = api_config.get("method", "GET").upper()
            headers = api_config.get("headers", {})
            params = api_config.get("params", {})
            body = api_config.get("body")
            auth_type = api_config.get("auth_type")
            
            # Validate required config
            if not url:
                self.logger.error(f"Missing URL for API source: {source.id}")
                return
            
            # Configure authentication if needed
            auth = None
            if auth_type == "basic":
                auth = (api_config.get("username", ""), api_config.get("password", ""))
            elif auth_type == "bearer":
                headers["Authorization"] = f"Bearer {api_config.get('token', '')}"
            elif auth_type == "api_key":
                # Handle API key in header or query param
                key_name = api_config.get("api_key_name", "api_key")
                key_value = api_config.get("api_key", "")
                key_in = api_config.get("api_key_in", "header")
                
                if key_in == "header":
                    headers[key_name] = key_value
                else:
                    params[key_name] = key_value
            
            # Make the API request
            response = None
            if method == "GET":
                response = requests.get(url, headers=headers, params=params, auth=auth, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=headers, params=params, json=body, auth=auth, timeout=30)
            
            # Process the response
            if response and response.status_code == 200:
                # Parse the response based on content type
                content_type = response.headers.get('Content-Type', '')
                
                if 'application/json' in content_type:
                    data = response.json()
                    
                    # Extract signals using path from configuration
                    path = api_config.get("data_path", "")
                    if path:
                        import jmespath
                        data = jmespath.search(path, data)
                    
                    # Process each item as a signal
                    signals_count = 0
                    for item in (data if isinstance(data, list) else [data]):
                        # Create a signal with proper attributes
                        signal_id = item.get("id", f"api-{source.id}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{signals_count}")
                        signal = Signal(
                            id=signal_id,
                            source_id=source.id,
                            content=item,
                            signal_type=source.signal_type,
                            metadata={
                                "ingested_at": datetime.now().isoformat(),
                                "api_source": url
                            }
                        )
                        
                        # Add the signal
                        self._add_signal(signal)
                        signals_count += 1
                    
                    self.logger.info(f"Ingested {signals_count} signals from API source: {source.id}")
                else:
                    # Handle non-JSON responses
                    signal = Signal(
                        id=f"api-{source.id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        source_id=source.id,
                        content=response.text,
                        signal_type=source.signal_type,
                        metadata={
                            "ingested_at": datetime.now().isoformat(),
                            "api_source": url,
                            "content_type": content_type
                        }
                    )
                    self._add_signal(signal)
                    self.logger.info(f"Ingested 1 text signal from API source: {source.id}")
            else:
                error_msg = f"API request failed: {response.status_code if response else 'No response'}"
                self.logger.error(f"Error ingesting from API source {source.id}: {error_msg}")
        
        except Exception as e:
            self.logger.error(f"Error ingesting from API source {source.id}: {str(e)}")
            # Record the error in source metadata
            source.metadata["last_error"] = str(e)
            source.metadata["last_error_time"] = datetime.now().isoformat()
    
    def _ingest_from_file(self, source: SignalSource):
        """Ingest signals from a file source."""
        try:
            import os
            import json
            import csv
            import yaml
            from datetime import datetime
            
            # Get file configuration
            file_config = source.config.get("file", {})
            file_path = file_config.get("path")
            file_type = file_config.get("type", "").lower()
            encoding = file_config.get("encoding", "utf-8")
            
            # Validate required config
            if not file_path:
                self.logger.error(f"Missing file path for file source: {source.id}")
                return
            
            # Expand path if needed (e.g., for ~, environment variables)
            file_path = os.path.expanduser(os.path.expandvars(file_path))
            
            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path} for source: {source.id}")
                return
            
            # Determine file type if not specified
            if not file_type:
                extension = os.path.splitext(file_path)[1].lower()
                if extension in ['.json', '.js']:
                    file_type = 'json'
                elif extension in ['.csv']:
                    file_type = 'csv'
                elif extension in ['.yaml', '.yml']:
                    file_type = 'yaml'
                elif extension in ['.txt', '.md', '.log']:
                    file_type = 'text'
                else:
                    file_type = 'text'  # default
            
            # Process file based on type
            signals_count = 0
            
            if file_type == 'json':
                # Read and parse JSON file
                with open(file_path, 'r', encoding=encoding) as f:
                    data = json.load(f)
                
                # Handle array or object
                items = data if isinstance(data, list) else [data]
                
                # Process each item as a signal
                for item in items:
                    signal_id = item.get("id", f"file-{source.id}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{signals_count}")
                    signal = Signal(
                        id=signal_id,
                        source_id=source.id,
                        content=item,
                        signal_type=source.signal_type,
                        metadata={
                            "ingested_at": datetime.now().isoformat(),
                            "file_source": file_path,
                            "file_type": file_type
                        }
                    )
                    self._add_signal(signal)
                    signals_count += 1
            
            elif file_type == 'csv':
                # Read and parse CSV file
                with open(file_path, 'r', encoding=encoding, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        signal_id = row.get("id", f"file-{source.id}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{signals_count}")
                        signal = Signal(
                            id=signal_id,
                            source_id=source.id,
                            content=dict(row),  # Convert OrderedDict to dict
                            signal_type=source.signal_type,
                            metadata={
                                "ingested_at": datetime.now().isoformat(),
                                "file_source": file_path,
                                "file_type": file_type
                            }
                        )
                        self._add_signal(signal)
                        signals_count += 1
            
            elif file_type == 'yaml':
                # Read and parse YAML file
                with open(file_path, 'r', encoding=encoding) as f:
                    data = yaml.safe_load(f)
                
                # Handle array or object
                items = data if isinstance(data, list) else [data]
                
                # Process each item as a signal
                for item in items:
                    signal_id = item.get("id", f"file-{source.id}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{signals_count}")
                    signal = Signal(
                        id=signal_id,
                        source_id=source.id,
                        content=item,
                        signal_type=source.signal_type,
                        metadata={
                            "ingested_at": datetime.now().isoformat(),
                            "file_source": file_path,
                            "file_type": file_type
                        }
                    )
                    self._add_signal(signal)
                    signals_count += 1
            
            else:  # Default to text
                # Read text file
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                # Split into lines if configured
                split_lines = file_config.get("split_lines", False)
                if split_lines:
                    lines = content.splitlines()
                    for line in lines:
                        if line.strip():  # Skip empty lines
                            signal = Signal(
                                id=f"file-{source.id}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{signals_count}",
                                source_id=source.id,
                                content=line,
                                signal_type=source.signal_type,
                                metadata={
                                    "ingested_at": datetime.now().isoformat(),
                                    "file_source": file_path,
                                    "file_type": file_type
                                }
                            )
                            self._add_signal(signal)
                            signals_count += 1
                else:
                    # Process entire file as a single signal
                    signal = Signal(
                        id=f"file-{source.id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        source_id=source.id,
                        content=content,
                        signal_type=source.signal_type,
                        metadata={
                            "ingested_at": datetime.now().isoformat(),
                            "file_source": file_path,
                            "file_type": file_type
                        }
                    )
                    self._add_signal(signal)
                    signals_count += 1
            
            self.logger.info(f"Ingested {signals_count} signals from file source: {source.id}")
            
            # Update last read time
            source.metadata["last_read_time"] = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"Error ingesting from file source {source.id}: {str(e)}")
            # Record the error in source metadata
            source.metadata["last_error"] = str(e)
            source.metadata["last_error_time"] = datetime.now().isoformat()
    
    def _ingest_from_database(self, source: SignalSource):
        """Ingest signals from a database source."""
        try:
            import sqlalchemy
            from datetime import datetime
            
            # Get database configuration
            db_config = source.config.get("database", {})
            db_type = db_config.get("type", "").lower()
            db_host = db_config.get("host", "localhost")
            db_port = db_config.get("port")
            db_name = db_config.get("database")
            db_user = db_config.get("username", "")
            db_pass = db_config.get("password", "")
            db_query = db_config.get("query")
            
            # Validate required config
            if not db_name or not db_query:
                self.logger.error(f"Missing database name or query for database source: {source.id}")
                return
            
            # Build connection string based on database type
            connection_string = ""
            
            if db_type == "mysql":
                port_str = f":{db_port}" if db_port else ""
                connection_string = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}{port_str}/{db_name}"
            elif db_type == "postgresql" or db_type == "postgres":
                port_str = f":{db_port}" if db_port else ""
                connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}{port_str}/{db_name}"
            elif db_type == "sqlite":
                connection_string = f"sqlite:///{db_name}"
            elif db_type == "mssql":
                port_str = f":{db_port}" if db_port else ""
                connection_string = f"mssql+pyodbc://{db_user}:{db_pass}@{db_host}{port_str}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server"
            else:
                self.logger.error(f"Unsupported database type '{db_type}' for source: {source.id}")
                return
            
            # Create engine and connect
            engine = sqlalchemy.create_engine(connection_string)
            connection = engine.connect()
            
            # Execute query
            result = connection.execute(sqlalchemy.text(db_query))
            
            # Process results
            signals_count = 0
            for row in result:
                # Convert row to dict
                if hasattr(row, "_asdict"):
                    row_dict = row._asdict()  # For SQLAlchemy 1.4+
                else:
                    row_dict = dict(row)  # For SQLAlchemy 1.3 and earlier
                
                # Create signal
                signal_id = row_dict.get("id", f"db-{source.id}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{signals_count}")
                signal = Signal(
                    id=signal_id,
                    source_id=source.id,
                    content=row_dict,
                    signal_type=source.signal_type,
                    metadata={
                        "ingested_at": datetime.now().isoformat(),
                        "db_source": f"{db_type}:{db_name}",
                        "query_hash": hash(db_query) % 10000  # Simple hash of the query for reference
                    }
                )
                self._add_signal(signal)
                signals_count += 1
            
            # Close connection
            connection.close()
            
            self.logger.info(f"Ingested {signals_count} signals from database source: {source.id}")
            
            # Update last read time
            source.metadata["last_read_time"] = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"Error ingesting from database source {source.id}: {str(e)}")
            # Record the error in source metadata
            source.metadata["last_error"] = str(e)
            source.metadata["last_error_time"] = datetime.now().isoformat()
    
    def _ingest_from_web(self, source: SignalSource):
        """Ingest signals from web scraping."""
        try:
            import requests
            from bs4 import BeautifulSoup
            from datetime import datetime
            import re
            
            # Get web scraping configuration
            web_config = source.config.get("web", {})
            url = web_config.get("url")
            selectors = web_config.get("selectors", [])
            extract_links = web_config.get("extract_links", False)
            follow_links = web_config.get("follow_links", False)
            max_links = web_config.get("max_links", 5)
            
            # Validate required config
            if not url:
                self.logger.error(f"Missing URL for web source: {source.id}")
                return
            
            # Helper function to extract content from a page
            def extract_from_url(page_url, is_followed_link=False):
                nonlocal signals_count
                
                # Setup headers to mimic a browser
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0'
                }
                
                # Fetch the page
                response = requests.get(page_url, headers=headers, timeout=30)
                if response.status_code != 200:
                    self.logger.warning(f"Failed to fetch URL {page_url}: Status code {response.status_code}")
                    return []
                
                # Parse the HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract content based on selectors
                extracted_contents = []
                
                if not selectors:
                    # If no selectors specified, use the page title and body text
                    title = soup.title.string if soup.title else ""
                    body_text = soup.body.get_text(separator=' ', strip=True) if soup.body else ""
                    
                    extracted_contents.append({
                        "title": title,
                        "content": body_text,
                        "url": page_url
                    })
                else:
                    # Use the provided selectors
                    for selector in selectors:
                        selector_type = selector.get("type", "css")
                        selector_path = selector.get("path", "")
                        selector_attr = selector.get("attribute", "")
                        selector_name = selector.get("name", "content")
                        
                        if not selector_path:
                            continue
                        
                        if selector_type == "css":
                            elements = soup.select(selector_path)
                        elif selector_type == "xpath":
                            # Using a simple approximation for XPath with CSS
                            # For complex XPath, a different library would be needed
                            elements = soup.select(selector_path)
                        else:
                            elements = []
                        
                        for element in elements:
                            content = ""
                            if selector_attr:
                                content = element.get(selector_attr, "")
                            else:
                                content = element.get_text(strip=True)
                            
                            if content:
                                # Check if we already have a content object started
                                content_obj = next((c for c in extracted_contents if c["url"] == page_url), None)
                                
                                if content_obj:
                                    # Add as a new field to existing object
                                    content_obj[selector_name] = content
                                else:
                                    # Create a new content object
                                    content_obj = {
                                        "url": page_url,
                                        selector_name: content
                                    }
                                    extracted_contents.append(content_obj)
                
                # Create signals from extracted content
                for content in extracted_contents:
                    signal_id = f"web-{source.id}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{signals_count}"
                    signal = Signal(
                        id=signal_id,
                        source_id=source.id,
                        content=content,
                        signal_type=source.signal_type,
                        metadata={
                            "ingested_at": datetime.now().isoformat(),
                            "web_source": page_url,
                            "is_followed_link": is_followed_link
                        }
                    )
                    self._add_signal(signal)
                    signals_count += 1
                
                # Extract links if requested
                links = []
                if extract_links or follow_links:
                    # Find all links
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        
                        # Make absolute URLs
                        if href.startswith('/'):
                            from urllib.parse import urlparse
                            base_url = "{0.scheme}://{0.netloc}".format(urlparse(page_url))
                            href = base_url + href
                        elif not href.startswith(('http://', 'https://')):
                            continue
                        
                        # Filter out common non-content links
                        if re.search(r'(login|signup|register|logout|admin|contact|about|terms|privacy|javascript:|mailto:|tel:)', href, re.I):
                            continue
                            
                        links.append(href)
                
                return links
            
            # Process the main URL
            signals_count = 0
            links = extract_from_url(url)
            
            # Follow links if configured
            followed_count = 0
            if follow_links and links:
                # Limit the number of links to follow
                for link in links[:max_links]:
                    if followed_count >= max_links:
                        break
                        
                    try:
                        extract_from_url(link, is_followed_link=True)
                        followed_count += 1
                    except Exception as link_error:
                        self.logger.warning(f"Error following link {link}: {str(link_error)}")
            
            self.logger.info(f"Ingested {signals_count} signals from web source: {source.id} (followed {followed_count} links)")
            
            # Update last read time
            source.metadata["last_read_time"] = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"Error ingesting from web source {source.id}: {str(e)}")
            # Record the error in source metadata
            source.metadata["last_error"] = str(e)
            source.metadata["last_error_time"] = datetime.now().isoformat()
    def _miner_coordination_loop(self):
        """Background loop for coordinating opportunity miner agents."""
        while self.running:
            try:
                # Check if we need to create/update miners
                for heuristic_id in self.miner_heuristics:
                    current_miners = self._get_miners_for_heuristic(heuristic_id)
                    desired_miners = min(
                        self.config["miners"]["max_miners_per_heuristic"],
                        3  # Start with a reasonable number
                    )
                    
                    if len(current_miners) < desired_miners:
                        # Create additional miners
                        for _ in range(desired_miners - len(current_miners)):
                            self._create_miner_agent(heuristic_id)
                
                # Adjust miner count based on performance
                self._optimize_miner_allocation()
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logging.error(f"Error in miner coordination loop: {str(e)}")
                time.sleep(60)  # Longer sleep on error
    
    def _reasoning_loop(self):
        """Background loop for reasoning about and evaluating opportunities."""
        while self.running:
            try:
                # Process opportunities in CANDIDATE status
                candidates = [
                    opp for opp in self.opportunities.values()
                    if opp.status == OpportunityStatus.CANDIDATE
                ]
                
                for candidate in candidates:
                    self._queue_for_evaluation(candidate.id)
                
                # Consolidate similar opportunities
                self._consolidate_similar_opportunities()
                
                time.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                logging.error(f"Error in reasoning loop: {str(e)}")
                time.sleep(120)  # Longer sleep on error
    
    def _queue_for_evaluation(self, opportunity_id: str):
        """Queue an opportunity for evaluation by the reasoning team."""
        if opportunity_id not in self.opportunities:
            return
            
        opportunity = self.opportunities[opportunity_id]
        
        # Update status
        opportunity.status = OpportunityStatus.UNDER_EVALUATION
        self._save_opportunities()
        
        # Create evaluation task
        if hasattr(self.kernel, "task_planner") and self.reasoning_team_id:
            self.kernel.task_planner.create_task(
                title=f"Evaluate opportunity: {opportunity.title}",
                description=f"Evaluate the candidate opportunity {opportunity_id}, checking evidence and assigning confidence level",
                priority="medium",
                team_id=self.reasoning_team_id,
                metadata={
                    "action": "evaluate_opportunity",
                    "opportunity_id": opportunity_id
                }
            )
    def _consolidate_similar_opportunities(self):
        """Find and consolidate similar opportunities."""
        if len(self.opportunities) < 2:
            return
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Collect all opportunities
            opportunities = list(self.opportunities.values())
            
            # Prepare text data for similarity comparison
            texts = []
            for opp in opportunities:
                # Combine title, description, and tags into a single text representation
                title = opp.title if hasattr(opp, 'title') else ""
                description = opp.description if hasattr(opp, 'description') else ""
                tags = " ".join(opp.tags) if hasattr(opp, 'tags') and opp.tags else ""
                
                combined_text = f"{title} {description} {tags}"
                texts.append(combined_text)
            
            # Calculate similarity matrix using TF-IDF vectorization
            vectorizer = TfidfVectorizer(stop_words='english')
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
            except ValueError as e:
                # If vectorization fails (e.g., empty documents), use a backup approach
                self.logger.warning(f"TF-IDF vectorization failed: {e}. Using tag-based similarity.")
                return self._consolidate_by_tags()
            
            # Find pairs of similar opportunities
            similar_pairs = []
            threshold = 0.8  # Similarity threshold for consolidation
            
            for i in range(len(opportunities)):
                for j in range(i+1, len(opportunities)):
                    if similarity_matrix[i, j] > threshold:
                        similar_pairs.append((i, j, similarity_matrix[i, j]))
            
            # Sort pairs by similarity (highest first)
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Consolidate similar opportunities
            consolidated_count = 0
            processed = set()
            
            for i, j, score in similar_pairs:
                if i in processed or j in processed:
                    continue
                
                # Get the two opportunities
                opp1 = opportunities[i]
                opp2 = opportunities[j]
                
                # Skip if they're already linked as parent/child
                if (hasattr(opp1, 'parent_id') and opp1.parent_id == opp2.id) or \
                   (hasattr(opp2, 'parent_id') and opp2.parent_id == opp1.id):
                    continue
                
                # Determine which one should be the parent (usually the older one)
                if getattr(opp1, 'created_at', 0) <= getattr(opp2, 'created_at', 0):
                    parent, child = opp1, opp2
                else:
                    parent, child = opp2, opp1
                
                # Merge the opportunities
                self._merge_opportunities(parent.id, child.id)
                
                processed.add(i)
                processed.add(j)
                consolidated_count += 1
            
            if consolidated_count > 0:
                self.logger.info(f"Consolidated {consolidated_count} pairs of similar opportunities")
            
            # Save changes
            self._save_opportunities()
            
        except ImportError:
            self.logger.warning("scikit-learn not installed, falling back to tag-based similarity")
            return self._consolidate_by_tags()
        except Exception as e:
            self.logger.error(f"Error consolidating opportunities: {str(e)}")

    def _consolidate_by_tags(self):
        """Fallback method to consolidate opportunities based on tags only."""
        opportunities = list(self.opportunities.values())
        
        # Group opportunities by tags
        tag_groups = {}
        for opp in opportunities:
            if not hasattr(opp, 'tags') or not opp.tags:
                continue
                
            # Use frozenset to make tags hashable
            tag_set = frozenset(opp.tags)
            if tag_set not in tag_groups:
                tag_groups[tag_set] = []
            tag_groups[tag_set].append(opp)
        
        # Consolidate opportunities with identical tag sets
        consolidated_count = 0
        for tag_set, opps in tag_groups.items():
            if len(opps) < 2:
                continue
                
            # Sort by creation date (oldest first)
            opps.sort(key=lambda x: getattr(x, 'created_at', 0))
            
            # First opportunity becomes the parent
            parent = opps[0]
            
            # Merge the rest as children
            for child in opps[1:]:
                self._merge_opportunities(parent.id, child.id)
                consolidated_count += 1
        
        if consolidated_count > 0:
            self.logger.info(f"Consolidated {consolidated_count} opportunities by tags")
            self._save_opportunities()
    
    def _merge_opportunities(self, parent_id, child_id):
        """Merge two opportunities by making one a child of the other."""
        if parent_id not in self.opportunities or child_id not in self.opportunities:
            return False
            
        parent = self.opportunities[parent_id]
        child = self.opportunities[child_id]
        
        # Set parent-child relationship
        child.parent_id = parent_id
        child.status = OpportunityStatus.MERGED
        
        # Update parent with additional data from child
        if hasattr(parent, 'tags') and hasattr(child, 'tags'):
            parent.tags = list(set(parent.tags + child.tags))
        
        if hasattr(parent, 'signals') and hasattr(child, 'signals'):
            parent.signals = list(set(parent.signals + child.signals))
        
        if hasattr(parent, 'references') and hasattr(child, 'references'):
            parent.references = list(set(parent.references + child.references))
        
        # Add note about the merge
        if hasattr(parent, 'notes'):
            if not isinstance(parent.notes, list):
                parent.notes = []
            parent.notes.append(f"Merged with opportunity {child_id} on {datetime.now().isoformat()}")
        
        return True
    
    def _optimize_miner_allocation(self):
        """Optimize allocation of miners based on heuristic performance."""
        if not self.miner_heuristics:
            return
            
        # Calculate success rate for each heuristic
        heuristics = list(self.miner_heuristics.values())
        for h in heuristics:
            h.success_rate = h.calculate_success_rate()
        
        # Sort by success rate (high to low)
        heuristics.sort(key=lambda h: h.success_rate, reverse=True)
        
        # Calculate allocation based on performance
        total_miners = self.config["miners"]["max_concurrent_miners"]
        allocated = 0
        
        for i, heuristic in enumerate(heuristics):
            # Allocate miners proportionally to success rate, with minimum of 1
            # Last one gets remainder
            if i == len(heuristics) - 1:
                desired = total_miners - allocated
            else:
                # Weight by success rate, ensure at least 1
                weight = max(0.1, heuristic.success_rate) if heuristic.success_rate > 0 else 0.1
                desired = max(1, int(total_miners * weight / len(heuristics)))
                desired = min(desired, total_miners - allocated)  # Don't exceed total
            
            # Update miner count for this heuristic
            current = len(self._get_miners_for_heuristic(heuristic.id))
            
            if current < desired:
                # Create additional miners
                for _ in range(desired - current):
                    self._create_miner_agent(heuristic.id)
            elif current > desired and current > 1:
                # Reduce miners (keep at least 1)
                to_remove = current - desired
                miners = self._get_miners_for_heuristic(heuristic.id)
                for _ in range(to_remove):
                    if miners:
                        self._terminate_miner_agent(miners.pop())
            
            allocated += desired
            
        logging.info(f"Optimized miner allocation: {allocated} miners across {len(heuristics)} heuristics")
        return allocated
    
    def _get_miners_for_heuristic(self, heuristic_id: str) -> List[str]:
        """Get agent IDs of miners using a specific heuristic."""
        if not hasattr(self.kernel, "agent_manager") or not self.miner_team_id:
            return []
            
        result = []
        team = self.kernel.agent_manager.teams.get(self.miner_team_id)
        if not team:
            return []
            
        for agent_id, agent_info in team.agents.items():
            agent = agent_info.get("agent")
            if agent and agent.attributes.get("miner_heuristic_id") == heuristic_id:
                result.append(agent_id)
                
        return result
    
    def _create_miner_agents_for_heuristic(self, heuristic_id: str):
        """Create initial miner agents for a new heuristic."""
        if heuristic_id not in self.miner_heuristics:
            return
            
        heuristic = self.miner_heuristics[heuristic_id]
        
        # Create miners up to the minimum count
        target_count = min(
            self.config["miners"]["max_miners_per_heuristic"],
            2  # Start with a reasonable number
        )
        
        for _ in range(target_count):
            self._create_miner_agent(heuristic_id)
    
    def _create_miner_agent(self, heuristic_id: str) -> Optional[str]:
        """Create a new miner agent for a specific heuristic."""
        if not hasattr(self.kernel, "agent_manager") or heuristic_id not in self.miner_heuristics:
            return None
            
        heuristic = self.miner_heuristics[heuristic_id]
        
        # Create an agent with the appropriate configuration
        name = f"Miner-{heuristic.name[:20]}-{str(uuid.uuid4())[:8]}"
        miner = self.kernel.agent_manager.create_agent(
            agent_type="researcher",
            name=name,
            capabilities=["data_analysis", "pattern_recognition", "opportunity_identification"],
            miner_heuristic_id=heuristic_id,
            prompt_template=heuristic.prompt_template
        )
        
        # Add to miner team
        if self.miner_team_id and miner:
            team = self.kernel.agent_manager.teams.get(self.miner_team_id)
            if team:
                team.add_agent(miner, role="miner")
            
        logging.info(f"Created miner agent: {name} for heuristic: {heuristic.name}")
        return miner.agent_id if miner else None
    
    def _terminate_miner_agent(self, agent_id: str) -> bool:
        """Terminate a miner agent."""
        if not hasattr(self.kernel, "agent_manager"):
            return False
            
        return self.kernel.agent_manager.terminate_agent(agent_id)
    
    def _mutate_prompt(self, prompt: str, mutation_factor: float) -> str:
        """Mutate a prompt template with the given mutation factor."""
        if mutation_factor == 0:
            return prompt
            
        # For non-zero mutation, ask LLM to mutate the prompt
        if hasattr(self.kernel, "llm_orchestrator"):
            try:
                mutation_prompt = f"""
                Mutate the following prompt template, applying a mutation factor of {mutation_factor} 
                (where 0.0 means no change and 1.0 means complete rewrite).
                Focus on improving the prompt's effectiveness for discovering strategic opportunities.
                
                ORIGINAL PROMPT:
                {prompt}
                
                MUTATED PROMPT:
                """
                
                response = self.kernel.llm_orchestrator.execute_prompt(
                    task_type="prompt_optimization",
                    prompt_template="direct",
                    params={"prompt": mutation_prompt},
                    max_tokens=2000
                )
                
                mutated_prompt = response.get("result", prompt)
                return mutated_prompt
                
            except Exception as e:
                logging.error(f"Error mutating prompt: {str(e)}")
                return prompt
        else:
            return prompt
