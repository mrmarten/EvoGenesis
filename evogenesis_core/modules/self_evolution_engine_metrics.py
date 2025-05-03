"""
Self-Evolution Engine Metrics Module - Provides metrics collection for evolutionary improvements.

This module works with the Performance Monitoring System to collect and analyze metrics
specifically for the Self-Evolution Engine's A/B tests, enabling:
1. Real-time comparison of different code versions
2. Data-driven decisions on which implementations to adopt
3. Historical performance tracking for evolutionary improvements
"""

import logging
import time
import json
import os
import uuid
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Set, Tuple
from datetime import datetime, timedelta

from evogenesis_core.modules.performance_monitoring import (
    PerformanceMonitoringSystem,
    ABTestMetricsCollector,
    MetricType,
    CollectorType
)

class TestVersionType(str, Enum):
    """Types of versions in an A/B test."""
    CONTROL = "control"        # The current/baseline version
    EXPERIMENTAL = "experimental"  # The new/experimental version


class SelfEvolutionMetrics:
    """
    Metrics collection and analysis for the Self-Evolution Engine.
    
    This class provides specialized metrics collection for:
    - A/B tests between code versions
    - Framework update performance analysis
    - Agent improvement tracking
    """
    
    def __init__(self, engine, kernel=None):
        """
        Initialize the metrics collection system.
        
        Args:
            engine: The Self-Evolution Engine instance
            kernel: Optional reference to the EvoGenesis kernel
        """
        self.engine = engine
        self.kernel = kernel or engine.kernel
        self.logger = logging.getLogger(__name__)
        
        # Get access to the monitoring system
        self.monitoring_system = None
        if hasattr(self.kernel, "performance_monitoring"):
            self.monitoring_system = self.kernel.performance_monitoring
        else:
            # Create a new monitoring system if one doesn't exist
            self.monitoring_system = PerformanceMonitoringSystem(self.kernel)
            self.logger.info("Created new performance monitoring system for Self-Evolution metrics")
        
        # Track active collectors for A/B tests
        self.active_test_collectors = {}
        
        # Track historical test results
        self.test_history_dir = os.path.join(os.getcwd(), "data", "evolution", "test_results")
        os.makedirs(self.test_history_dir, exist_ok=True)
        
        self.logger.info("Self-Evolution metrics initialized")
    
    def create_ab_test_collector(self, test_id: str, feature: str, 
                                version_a: str, version_b: str,
                                metrics: List[str] = None) -> str:
        """
        Create a metrics collector for an A/B test.
        
        Args:
            test_id: ID of the A/B test
            feature: Name of the feature being tested
            version_a: Identifier for version A (control)
            version_b: Identifier for version B (experimental)
            metrics: List of metrics to collect
            
        Returns:
            ID of the created collector
        """
        # Default metrics if none provided
        if not metrics:
            metrics = ["request_count", "error_count", "latency", "success_rate", "cpu_usage", "memory_usage"]
        
        # Create the collector
        collector_id = self.monitoring_system.create_abtest_collector(
            test_id=test_id,
            feature_name=feature,
            version_a=version_a,
            version_b=version_b,
            metrics_to_collect=metrics,
            sampling_interval=1.0,  # Sample every second
            aggregate_interval=5.0   # Aggregate every 5 seconds for real-time comparison
        )
        
        # Store for easy access
        self.active_test_collectors[test_id] = collector_id
        
        self.logger.info(f"Created A/B test collector for {feature} ({test_id})")
        return collector_id
    
    def record_ab_test_metric(self, test_id: str, version: str, 
                            metric_name: str, value: Union[int, float],
                            context: Dict[str, Any] = None):
        """
        Record a metric for an A/B test.
        
        Args:
            test_id: ID of the A/B test
            version: Version identifier ("version_a" or "version_b")
            metric_name: Name of the metric to record
            value: Value to record
            context: Optional context information
        """
        if test_id not in self.active_test_collectors:
            self.logger.warning(f"Cannot record metric for unknown test: {test_id}")
            return
        
        collector_id = self.active_test_collectors[test_id]
        collector = self.monitoring_system.get_collector(collector_id)
        
        if not collector:
            self.logger.warning(f"Collector not found for test: {test_id}")
            return
        
        # Record the metric
        try:
            collector.record_metric(f"{version}_{metric_name}", value, context=context)
        except Exception as e:
            self.logger.error(f"Error recording A/B test metric: {e}")
    
    def batch_record_ab_test_metrics(self, test_id: str, version: str, 
                                   metrics: Dict[str, Union[int, float]],
                                   context: Dict[str, Any] = None):
        """
        Record multiple metrics for an A/B test.
        
        Args:
            test_id: ID of the A/B test
            version: Version identifier ("version_a" or "version_b")
            metrics: Dictionary of metric names to values
            context: Optional context information
        """
        if test_id not in self.active_test_collectors:
            self.logger.warning(f"Cannot record metrics for unknown test: {test_id}")
            return
        
        # Use the monitoring system's API for batch recording
        try:
            self.monitoring_system.record_abtest_metrics(test_id, version, metrics)
        except Exception as e:
            self.logger.error(f"Error batch recording A/B test metrics: {e}")
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Get the current results of an A/B test.
        
        Args:
            test_id: ID of the A/B test
            
        Returns:
            Dictionary with test results and comparison data
        """
        if test_id not in self.active_test_collectors:
            self.logger.warning(f"Results not available for unknown test: {test_id}")
            return {"error": f"Test not found: {test_id}"}
        
        try:
            # Get comprehensive results from the monitoring system
            return self.monitoring_system.get_abtest_results(test_id)
        except Exception as e:
            self.logger.error(f"Error getting A/B test results: {e}")
            return {"error": str(e)}
    
    def analyze_ab_test(self, test_id: str) -> Dict[str, Any]:
        """
        Perform detailed analysis of an A/B test.
        
        Args:
            test_id: ID of the A/B test
            
        Returns:
            Dictionary with analysis results and recommendation
        """
        # Get basic results
        results = self.get_ab_test_results(test_id)
        if "error" in results:
            return results
        
        # Extract key results
        comparison = results.get("comparison", {})
        metrics = results.get("current_metrics", {}).get("metrics", {})
        
        # Additional analysis beyond the basic comparison
        try:
            # Get test configuration from the engine
            test_config = self.engine.active_ab_tests.get(test_id, {})
            feature = test_config.get("feature", "unknown")
            version_a = test_config.get("version_a", "control")
            version_b = test_config.get("version_b", "experimental")
            
            # Analyze key metrics for this feature type
            feature_category = self._determine_feature_category(feature)
            key_metrics = self._get_key_metrics_for_feature_category(feature_category)
            
            # Compute weighted score based on feature category
            version_a_score = 0
            version_b_score = 0
            metric_weights = {}
            metric_results = {}
            
            for metric_name, weight in key_metrics.items():
                if f"{metric_name}_avg" in comparison:
                    metric_key = f"{metric_name}_avg"
                elif metric_name in comparison:
                    metric_key = metric_name
                else:
                    continue
                
                comp_data = comparison[metric_key]
                is_better = comp_data.get("improved", False)
                rel_diff = abs(comp_data.get("relative_diff", 0))
                
                # Scoring based on importance and magnitude of improvement
                score_value = weight * (rel_diff / 100)
                
                if is_better:
                    version_b_score += score_value
                else:
                    version_a_score += score_value
                
                metric_weights[metric_key] = weight
                metric_results[metric_key] = {
                    "weight": weight,
                    "is_better_in_b": is_better,
                    "relative_diff": rel_diff,
                    "score_contribution": score_value
                }
            
            # Normalize scores
            total_weight = sum(metric_weights.values())
            if total_weight > 0:
                version_a_score = version_a_score / total_weight * 100
                version_b_score = version_b_score / total_weight * 100
            
            # Get confidence from comparison
            confidence = comparison.get("confidence", 0.5)
            
            # Make a recommendation
            if version_b_score > version_a_score:
                if confidence >= 0.9:
                    recommendation = "strongly_adopt_b"
                    recommendation_text = f"Strongly recommend adopting version {version_b}"
                elif confidence >= 0.7:
                    recommendation = "adopt_b"
                    recommendation_text = f"Recommend adopting version {version_b}"
                else:
                    recommendation = "tentatively_adopt_b"
                    recommendation_text = f"Tentatively recommend adopting version {version_b}, but continue testing"
            else:
                if confidence >= 0.9:
                    recommendation = "strongly_retain_a"
                    recommendation_text = f"Strongly recommend keeping version {version_a}"
                elif confidence >= 0.7:
                    recommendation = "retain_a"
                    recommendation_text = f"Recommend keeping version {version_a}"
                else:
                    recommendation = "tentatively_retain_a"
                    recommendation_text = f"Tentatively recommend keeping version {version_a}, but continue testing"
            
            # Create result
            analysis_result = {
                "test_id": test_id,
                "feature": feature,
                "category": feature_category,
                "version_a": version_a,
                "version_b": version_b,
                "version_a_score": version_a_score,
                "version_b_score": version_b_score,
                "confidence": confidence,
                "key_metrics": metric_results,
                "recommendation": recommendation,
                "recommendation_text": recommendation_text,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save results to history
            self._save_test_results(test_id, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing A/B test: {e}")
            return {"error": str(e), "base_results": results}
    
    def stop_ab_test_collector(self, test_id: str):
        """
        Stop and cleanup the collector for an A/B test.
        
        Args:
            test_id: ID of the A/B test
        """
        if test_id not in self.active_test_collectors:
            return
        
        collector_id = self.active_test_collectors[test_id]
        collector = self.monitoring_system.get_collector(collector_id)
        
        if collector:
            try:
                # Get final results before stopping
                results = self.analyze_ab_test(test_id)
                
                # Stop the collector
                collector.stop()
                self.logger.info(f"Stopped A/B test collector for test {test_id}")
                
                # Clean up
                del self.active_test_collectors[test_id]
                
            except Exception as e:
                self.logger.error(f"Error stopping A/B test collector: {e}")
    
    def _determine_feature_category(self, feature: str) -> str:
        """
        Determine the category of a feature for metric weighting.
        
        Args:
            feature: Name of the feature
            
        Returns:
            Category name
        """
        # Map feature to categories based on name patterns
        feature_lower = feature.lower()
        
        if any(term in feature_lower for term in ["memory", "storage", "cache", "database"]):
            return "storage"
        
        if any(term in feature_lower for term in ["llm", "model", "ai", "predict"]):
            return "ai_model"
        
        if any(term in feature_lower for term in ["interface", "ui", "ux", "display"]):
            return "interface"
        
        if any(term in feature_lower for term in ["network", "api", "connect", "http"]):
            return "network"
        
        if any(term in feature_lower for term in ["security", "auth", "permission"]):
            return "security"
        
        if any(term in feature_lower for term in ["tool", "utility", "function"]):
            return "tooling"
        
        if any(term in feature_lower for term in ["agent", "task", "plan"]):
            return "agent"
        
        # Default category
        return "general"
    
    def _get_key_metrics_for_feature_category(self, category: str) -> Dict[str, float]:
        """
        Get the key metrics and their weights for a feature category.
        
        Args:
            category: Feature category
            
        Returns:
            Dictionary mapping metric names to their weights
        """
        # Define metric weights by category
        category_metrics = {
            "storage": {
                "latency": 0.3,
                "memory_usage": 0.25,
                "error_rate": 0.25,
                "success_rate": 0.2
            },
            "ai_model": {
                "latency": 0.25,
                "error_rate": 0.2,
                "success_rate": 0.3,
                "cpu_usage": 0.25
            },
            "interface": {
                "latency": 0.4,
                "error_rate": 0.3,
                "success_rate": 0.3
            },
            "network": {
                "latency": 0.4,
                "error_rate": 0.3,
                "success_rate": 0.2,
                "request_count": 0.1
            },
            "security": {
                "error_rate": 0.4,
                "success_rate": 0.4,
                "latency": 0.2
            },
            "tooling": {
                "success_rate": 0.3,
                "error_rate": 0.3,
                "latency": 0.25,
                "cpu_usage": 0.15
            },
            "agent": {
                "success_rate": 0.4,
                "latency": 0.3,
                "error_rate": 0.2,
                "memory_usage": 0.1
            },
            "general": {
                "success_rate": 0.25,
                "error_rate": 0.25,
                "latency": 0.25,
                "cpu_usage": 0.15,
                "memory_usage": 0.1
            }
        }
        
        return category_metrics.get(category, category_metrics["general"])
    
    def _save_test_results(self, test_id: str, results: Dict[str, Any]):
        """
        Save test results to the history directory.
        
        Args:
            test_id: ID of the A/B test
            results: Test results to save
        """
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{test_id}_{timestamp}.json"
            filepath = os.path.join(self.test_history_dir, filename)
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
                
            self.logger.info(f"Saved A/B test results to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving test results: {e}")
    
    def get_historical_test_results(self, feature: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get historical A/B test results.
        
        Args:
            feature: Optional feature name to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of historical test results
        """
        results = []
        
        try:
            # List all result files
            files = sorted(os.listdir(self.test_history_dir), reverse=True)
            
            # Read each file
            for filename in files[:limit * 2]:  # Read more than needed for filtering
                if not filename.endswith(".json"):
                    continue
                    
                filepath = os.path.join(self.test_history_dir, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Filter by feature if specified
                if feature and data.get("feature") != feature:
                    continue
                
                results.append(data)
                
                # Stop if we have enough results
                if len(results) >= limit:
                    break
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting historical test results: {e}")
            return []
    
    def get_performance_improvement_trend(self, feature: str = None, 
                                        metric: str = "success_rate",
                                        days: int = 30) -> Dict[str, Any]:
        """
        Analyze historical A/B test results to identify performance trends.
        
        Args:
            feature: Optional feature name to filter by
            metric: Metric to analyze trends for
            days: Number of days to include in the analysis
            
        Returns:
            Dictionary with trend analysis
        """        # Query historical metrics data and perform trend analysis
        try:
            import json
            import os
            import datetime
            import numpy as np
            from scipy import stats
            
            # Calculate date range
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            # Path to metrics data files
            metrics_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "data", "metrics", feature)
            
            # Initialize data collection arrays
            timestamps = []
            values = []
            
            # Attempt to load from metrics database first
            if hasattr(self, "metrics_db") and self.metrics_db:
                db_metrics = self.metrics_db.query(
                    feature=feature,
                    metric=metric,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat()
                )
                
                if db_metrics and len(db_metrics) > 1:
                    # Extract timestamps and values from database results
                    for record in db_metrics:
                        timestamps.append(datetime.datetime.fromisoformat(record["timestamp"]).timestamp())
                        values.append(record["value"])
            
            # If no database or insufficient data, try loading from JSON files
            if len(values) < 2 and os.path.exists(metrics_dir):
                # Load data from all relevant metric files within the date range
                for filename in os.listdir(metrics_dir):
                    if not filename.endswith('.json'):
                        continue
                    
                    try:
                        file_date_str = filename.split('_')[0]  # Assuming format: YYYYMMDD_metrics.json
                        file_date = datetime.datetime.strptime(file_date_str, "%Y%m%d")
                        
                        # Check if file is within date range
                        if file_date >= start_date and file_date <= end_date:
                            file_path = os.path.join(metrics_dir, filename)
                            
                            with open(file_path, 'r') as f:
                                daily_metrics = json.load(f)
                                
                                # Extract the specific metric we're analyzing
                                if metric in daily_metrics:
                                    timestamps.append(file_date.timestamp())
                                    values.append(daily_metrics[metric])
                    except Exception as e:
                        print(f"Error processing metrics file {filename}: {str(e)}")
            
            # Analyze trends with at least 2 data points
            if len(values) >= 2:
                # Convert to numpy arrays for analysis
                timestamps_array = np.array(timestamps)
                values_array = np.array(values)
                
                # Sort by timestamp to ensure chronological order
                sort_indices = np.argsort(timestamps_array)
                timestamps_array = timestamps_array[sort_indices]
                values_array = values_array[sort_indices]
                
                # Perform linear regression to find trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps_array, values_array)
                
                # Calculate relative change over the period
                first_value = values_array[0] if values_array[0] != 0 else 0.001  # Avoid division by zero
                total_change = (values_array[-1] - values_array[0]) / first_value
                days_interval = (timestamps_array[-1] - timestamps_array[0]) / (24 * 3600)  # Convert seconds to days
                
                # Calculate monthly rate of change
                monthly_change = total_change * (30 / days_interval) if days_interval > 0 else 0
                
                # Determine trend direction
                if slope > 0.001:
                    trend_direction = "improving"
                elif slope < -0.001:
                    trend_direction = "declining"
                else:
                    trend_direction = "stable"
                
                # Calculate confidence from r-squared value
                confidence = min(0.99, max(0.5, abs(r_value)))
                
                return {
                    "feature": feature,
                    "metric": metric,
                    "days_analyzed": days,
                    "trend_direction": trend_direction,
                    "improvement_rate": f"{monthly_change*100:.1f}% per month",
                    "slope": slope,
                    "monthly_change": monthly_change,
                    "total_change": total_change,
                    "r_squared": r_value ** 2,
                    "confidence": confidence,
                    "data_points": len(values),
                    "last_value": float(values_array[-1]) if len(values_array) > 0 else None,
                    "first_value": float(values_array[0]) if len(values_array) > 0 else None
                }
            else:
                # Not enough data for trend analysis
                return {
                    "feature": feature,
                    "metric": metric,
                    "days_analyzed": days,
                    "trend_direction": "unknown",
                    "improvement_rate": "unknown",
                    "confidence": 0.0,
                    "error": "Insufficient data points for trend analysis",
                    "data_points": len(values)
                }
                
        except Exception as e:
            print(f"Error analyzing trends for {feature}.{metric}: {str(e)}")
            # Fallback response with error information
            return {
                "feature": feature,
                "metric": metric,
                "days_analyzed": days,
                "trend_direction": "unknown",
                "improvement_rate": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
