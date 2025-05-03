"""
Performance Monitoring Module - Provides metrics collection and telemetry for EvoGenesis.

This module implements a comprehensive monitoring system that collects performance
metrics from various components of the EvoGenesis framework, enabling:
1. Real-time performance tracking
2. Threshold-based alerting
3. Historical data collection for analysis
4. A/B test metrics collection and comparison
"""

import logging
import threading
import time
import json
import os
import uuid
import statistics
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable, Set
from datetime import datetime, timedelta
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"        # Cumulative value that only increases (e.g., request count)
    GAUGE = "gauge"            # Value that can go up and down (e.g., memory usage)
    HISTOGRAM = "histogram"    # Distribution of values (e.g., latency measurements)
    SUMMARY = "summary"        # Similar to histogram but with calculated quantiles

class AlertSeverity(str, Enum):
    """Severity levels for performance alerts."""
    INFO = "info"              # Informational, no action needed
    WARNING = "warning"        # Potential issue, monitoring recommended
    ERROR = "error"            # Serious issue, action required
    CRITICAL = "critical"      # Critical issue, immediate action required

class CollectorType(str, Enum):
    """Types of metric collectors."""
    SYSTEM = "system"          # Collector for system-level metrics (CPU, memory, etc.)
    COMPONENT = "component"    # Collector for component-specific metrics 
    AB_TEST = "ab_test"        # Collector specifically for A/B test metrics
    CUSTOM = "custom"          # Custom collector for specialized metrics

class MetricsCollector:
    """
    Base class for metrics collection in EvoGenesis.
    
    This class provides a thread-safe way to collect various types of performance
    metrics from system components, with support for aggregation, alerting, and
    telemetry logging.
    """
    
    def __init__(self, 
                collector_id: str,
                name: str,
                collector_type: CollectorType,
                sampling_interval: float = 1.0,
                aggregate_interval: float = 60.0,
                telemetry_enabled: bool = True,
                alerting_enabled: bool = True):
        """
        Initialize a metrics collector.
        
        Args:
            collector_id: Unique ID for this collector
            name: Human-readable name for this collector
            collector_type: Type of collector (system, component, etc.)
            sampling_interval: How often to sample metrics (seconds)
            aggregate_interval: How often to aggregate metrics (seconds)
            telemetry_enabled: Whether to log telemetry data
            alerting_enabled: Whether to generate alerts
        """
        self.collector_id = collector_id
        self.name = name
        self.collector_type = collector_type
        self.sampling_interval = sampling_interval
        self.aggregate_interval = aggregate_interval
        self.telemetry_enabled = telemetry_enabled
        self.alerting_enabled = alerting_enabled
        
        # Metric storage
        self.metrics = {}
        self.raw_samples = {}
        self.aggregated_metrics = {}
        self.last_aggregate_time = time.time()
        
        # Alert thresholds
        self.alert_thresholds = {}
        self.triggered_alerts = set()
        
        # Thread management
        self.running = False
        self.collection_thread = None
        self.thread_lock = threading.RLock()
        self.metrics_queue = queue.Queue()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Telemetry log path (create directory if it doesn't exist)
        self.telemetry_dir = os.path.join(os.getcwd(), "data", "telemetry")
        os.makedirs(self.telemetry_dir, exist_ok=True)
        self.telemetry_log_path = os.path.join(
            self.telemetry_dir, 
            f"{self.collector_type}_{self.collector_id}.jsonl"
        )
    
    def start(self):
        """Start metrics collection in a background thread."""
        with self.thread_lock:
            if self.running:
                return
            
            self.running = True
            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                name=f"metrics-{self.collector_id}",
                daemon=True
            )
            self.collection_thread.start()
            self.logger.info(f"Started metrics collector: {self.name} ({self.collector_id})")
    
    def stop(self):
        """Stop metrics collection."""
        with self.thread_lock:
            if not self.running:
                return
                
            self.running = False
            if self.collection_thread:
                self.collection_thread.join(timeout=2.0)
                self.collection_thread = None
            
            # Perform final aggregation
            self._aggregate_metrics()
            self.logger.info(f"Stopped metrics collector: {self.name} ({self.collector_id})")
    
    def register_metric(self, metric_name: str, metric_type: MetricType, 
                       description: str = "", unit: str = "", 
                       alert_thresholds: Dict[str, Any] = None):
        """
        Register a new metric to be collected.
        
        Args:
            metric_name: Name of the metric
            metric_type: Type of metric (counter, gauge, etc.)
            description: Human-readable description
            unit: Unit of measurement (e.g., "ms", "bytes", etc.)
            alert_thresholds: Dictionary of alert thresholds for this metric
        """
        with self.thread_lock:
            self.metrics[metric_name] = {
                "type": metric_type,
                "description": description,
                "unit": unit,
                "created_at": datetime.now().isoformat()
            }
            
            # Initialize storage based on metric type
            if metric_type == MetricType.COUNTER:
                self.raw_samples[metric_name] = 0
            elif metric_type == MetricType.GAUGE:
                self.raw_samples[metric_name] = []
            elif metric_type in (MetricType.HISTOGRAM, MetricType.SUMMARY):
                self.raw_samples[metric_name] = []
            
            # Store alert thresholds if provided
            if alert_thresholds:
                self.alert_thresholds[metric_name] = alert_thresholds
    
    def record_metric(self, metric_name: str, value: Union[int, float], 
                     timestamp: Optional[float] = None,
                     context: Optional[Dict[str, Any]] = None):
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Value to record
            timestamp: Optional timestamp (defaults to current time)
            context: Optional context data to store with the metric
        """
        if metric_name not in self.metrics:
            self.logger.warning(f"Attempted to record unregistered metric: {metric_name}")
            return
        
        # Add to queue for thread-safe processing
        self.metrics_queue.put({
            "metric_name": metric_name,
            "value": value,
            "timestamp": timestamp or time.time(),
            "context": context or {}
        })
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get the current metric values.
        
        Returns:
            Dictionary of current metric values and metadata
        """
        with self.thread_lock:
            result = {
                "collector_id": self.collector_id,
                "name": self.name,
                "type": self.collector_type,
                "timestamp": datetime.now().isoformat(),
                "metrics": {}
            }
            
            # Copy the latest aggregated metrics
            for metric_name, metric_info in self.metrics.items():
                if metric_name in self.aggregated_metrics:
                    result["metrics"][metric_name] = {
                        **metric_info,
                        "values": self.aggregated_metrics[metric_name]
                    }
            
            return result
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get any triggered alerts.
        
        Returns:
            List of active alerts
        """
        with self.thread_lock:
            return list(self.triggered_alerts)
    
    def set_alert_threshold(self, metric_name: str, threshold_config: Dict[str, Any]):
        """
        Set alert thresholds for a specific metric.
        
        Args:
            metric_name: Name of the metric
            threshold_config: Dictionary with threshold configuration
        """
        if metric_name not in self.metrics:
            self.logger.warning(f"Attempted to set threshold for unregistered metric: {metric_name}")
            return
        
        with self.thread_lock:
            self.alert_thresholds[metric_name] = threshold_config
    
    def _collection_loop(self):
        """Main collection loop running in background thread."""
        last_sample_time = time.time()
        last_aggregate_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Process metrics from the queue
            self._process_queued_metrics()
            
            # Check if it's time to collect samples
            if current_time - last_sample_time >= self.sampling_interval:
                self._collect_samples()
                last_sample_time = current_time
            
            # Check if it's time to aggregate
            if current_time - last_aggregate_time >= self.aggregate_interval:
                self._aggregate_metrics()
                last_aggregate_time = current_time
                
                # Check alert thresholds
                if self.alerting_enabled:
                    self._check_alert_thresholds()
                
                # Log telemetry data
                if self.telemetry_enabled:
                    self._log_telemetry()
            
            # Sleep a bit to reduce CPU usage
            time.sleep(min(0.1, self.sampling_interval / 10))
    
    def _process_queued_metrics(self):
        """Process metrics from the queue."""
        # Process all available metrics in the queue
        while not self.metrics_queue.empty():
            try:
                metric_data = self.metrics_queue.get_nowait()
                self._update_raw_samples(
                    metric_data["metric_name"],
                    metric_data["value"],
                    metric_data["timestamp"],
                    metric_data["context"]
                )
                self.metrics_queue.task_done()
            except queue.Empty:
                break
    
    def _update_raw_samples(self, metric_name: str, value: Union[int, float], 
                          timestamp: float, context: Dict[str, Any]):
        """Update raw samples with a new metric value."""
        if metric_name not in self.metrics:
            return
        
        with self.thread_lock:
            metric_type = self.metrics[metric_name]["type"]
            
            if metric_type == MetricType.COUNTER:
                # For counters, we just increase the value
                self.raw_samples[metric_name] += value
            
            elif metric_type == MetricType.GAUGE:
                # For gauges, we record the value with timestamp
                self.raw_samples[metric_name].append({
                    "value": value,
                    "timestamp": timestamp,
                    "context": context
                })
                
                # Keep only recent values for gauges (last minute)
                cutoff_time = timestamp - 60
                self.raw_samples[metric_name] = [
                    sample for sample in self.raw_samples[metric_name]
                    if sample["timestamp"] > cutoff_time
                ]
            
            elif metric_type in (MetricType.HISTOGRAM, MetricType.SUMMARY):
                # For histograms and summaries, we record all values
                self.raw_samples[metric_name].append({
                    "value": value,
                    "timestamp": timestamp,
                    "context": context
                })
    
    def _collect_samples(self):
        """Collect samples from the source system or component."""
        # This method can be overridden by subclasses to collect specific metrics
        pass
    
    def _aggregate_metrics(self):
        """Aggregate raw samples into meaningful metrics."""
        with self.thread_lock:
            current_time = time.time()
            aggregated = {}
            
            for metric_name, metric_info in self.metrics.items():
                metric_type = metric_info["type"]
                raw_data = self.raw_samples.get(metric_name)
                
                if raw_data is None:
                    continue
                
                if metric_type == MetricType.COUNTER:
                    # For counters, just use the current value
                    aggregated[metric_name] = {
                        "value": raw_data,
                        "timestamp": current_time
                    }
                
                elif metric_type == MetricType.GAUGE:
                    # For gauges, calculate recent stats
                    if not raw_data:
                        continue
                    
                    values = [sample["value"] for sample in raw_data]
                    if not values:
                        continue
                    
                    aggregated[metric_name] = {
                        "current": values[-1],
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "count": len(values),
                        "timestamp": current_time
                    }
                
                elif metric_type in (MetricType.HISTOGRAM, MetricType.SUMMARY):
                    # For histograms/summaries, calculate distribution stats
                    if not raw_data:
                        continue
                        
                    values = [sample["value"] for sample in raw_data]
                    if not values:
                        continue
                    
                    try:
                        aggregated[metric_name] = {
                            "count": len(values),
                            "min": min(values),
                            "max": max(values),
                            "avg": sum(values) / len(values),
                            "median": statistics.median(values),
                            "p95": self._percentile(values, 95),
                            "p99": self._percentile(values, 99),
                            "stddev": statistics.stdev(values) if len(values) > 1 else 0,
                            "timestamp": current_time
                        }
                    except (statistics.StatisticsError, ValueError) as e:
                        self.logger.warning(f"Error calculating statistics for {metric_name}: {e}")
            
            # Update the aggregated metrics
            self.aggregated_metrics = aggregated
            self.last_aggregate_time = current_time
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate a percentile from a list of values."""
        if not values:
            return 0.0
            
        # Sort values
        sorted_values = sorted(values)
        
        # Calculate index
        k = (len(sorted_values) - 1) * (percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_values[int(k)]
        
        # Interpolate
        d0 = sorted_values[int(f)] * (c - k)
        d1 = sorted_values[int(c)] * (k - f)
        return d0 + d1
    
    def _check_alert_thresholds(self):
        """Check for threshold violations and generate alerts."""
        with self.thread_lock:
            for metric_name, threshold in self.alert_thresholds.items():
                if metric_name not in self.aggregated_metrics:
                    continue
                
                metric_data = self.aggregated_metrics[metric_name]
                
                # Critical thresholds
                if "critical" in threshold:
                    if self._check_threshold_breach(metric_data, threshold["critical"]):
                        self._create_alert(metric_name, metric_data, AlertSeverity.CRITICAL)
                        continue
                
                # Error thresholds
                if "error" in threshold:
                    if self._check_threshold_breach(metric_data, threshold["error"]):
                        self._create_alert(metric_name, metric_data, AlertSeverity.ERROR)
                        continue
                
                # Warning thresholds
                if "warning" in threshold:
                    if self._check_threshold_breach(metric_data, threshold["warning"]):
                        self._create_alert(metric_name, metric_data, AlertSeverity.WARNING)
                        continue
                
                # Info thresholds
                if "info" in threshold:
                    if self._check_threshold_breach(metric_data, threshold["info"]):
                        self._create_alert(metric_name, metric_data, AlertSeverity.INFO)
                        continue
                
                # Clear alert if no threshold is breached
                self._clear_alert(metric_name)
    
    def _check_threshold_breach(self, metric_data: Dict[str, Any], 
                              threshold: Dict[str, Any]) -> bool:
        """
        Check if a metric breaches a threshold.
        
        Args:
            metric_data: The aggregated metric data
            threshold: The threshold configuration
            
        Returns:
            True if threshold is breached, False otherwise
        """
        # Simple single value comparison
        if "value" in metric_data and "min" in threshold and metric_data["value"] < threshold["min"]:
            return True
        
        if "value" in metric_data and "max" in threshold and metric_data["value"] > threshold["max"]:
            return True
        
        # More complex comparisons for histograms/summaries
        for key in ["min", "max", "avg", "median", "p95", "p99"]:
            if key in metric_data:
                if f"{key}_min" in threshold and metric_data[key] < threshold[f"{key}_min"]:
                    return True
                if f"{key}_max" in threshold and metric_data[key] > threshold[f"{key}_max"]:
                    return True
        
        return False
    
    def _create_alert(self, metric_name: str, metric_data: Dict[str, Any], 
                    severity: AlertSeverity):
        """
        Create a new alert for a breached threshold.
        
        Args:
            metric_name: Name of the metric
            metric_data: The aggregated metric data
            severity: Alert severity level
        """
        alert_id = f"{self.collector_id}_{metric_name}_{severity.value}"
        
        alert = {
            "alert_id": alert_id,
            "collector_id": self.collector_id,
            "metric_name": metric_name,
            "severity": severity.value,
            "message": f"{severity.value.upper()}: {self.name} - {metric_name} threshold breached",
            "metric_data": metric_data,
            "timestamp": time.time(),
            "created_at": datetime.now().isoformat()
        }
        
        # Check if this is a new alert or an update
        for existing_alert in self.triggered_alerts:
            if existing_alert["alert_id"] == alert_id:
                # Update existing alert
                existing_alert.update(alert)
                self.logger.warning(f"Alert updated: {alert['message']}")
                return
        
        # Create new alert
        self.triggered_alerts.add(alert)
        self.logger.warning(f"Alert created: {alert['message']}")
    
    def _clear_alert(self, metric_name: str):
        """
        Clear alerts for a specific metric.
        
        Args:
            metric_name: Name of the metric
        """
        alert_ids_to_remove = set()
        
        for alert in self.triggered_alerts:
            if alert["metric_name"] == metric_name:
                alert_ids_to_remove.add(alert["alert_id"])
        
        if alert_ids_to_remove:
            self.triggered_alerts = {
                alert for alert in self.triggered_alerts
                if alert["alert_id"] not in alert_ids_to_remove
            }
    
    def _log_telemetry(self):
        """Log telemetry data to a file."""
        if not self.telemetry_enabled:
            return
        
        with self.thread_lock:
            try:
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "collector_id": self.collector_id,
                    "name": self.name,
                    "type": self.collector_type,
                    "metrics": self.aggregated_metrics
                }
                
                with open(self.telemetry_log_path, "a") as f:
                    f.write(json.dumps(telemetry_data) + "\n")
            
            except Exception as e:
                self.logger.error(f"Error logging telemetry: {e}")


class SystemMetricsCollector(MetricsCollector):
    """Collector for system-level metrics (CPU, memory, disk, network)."""
    
    def __init__(self, name: str = "System Metrics", **kwargs):
        collector_id = kwargs.pop("collector_id", f"system-{uuid.uuid4().hex[:8]}")
        super().__init__(collector_id=collector_id, name=name, 
                         collector_type=CollectorType.SYSTEM, **kwargs)
        
        # Register system metrics
        self.register_metric("cpu_usage", MetricType.GAUGE, 
                           description="CPU usage percentage", 
                           unit="%",
                           alert_thresholds={
                               "warning": {"max": 85.0},
                               "critical": {"max": 95.0}
                           })
        
        self.register_metric("memory_usage", MetricType.GAUGE, 
                           description="Memory usage percentage", 
                           unit="%",
                           alert_thresholds={
                               "warning": {"max": 85.0},
                               "critical": {"max": 95.0}
                           })
        
        self.register_metric("disk_usage", MetricType.GAUGE, 
                           description="Disk usage percentage", 
                           unit="%",
                           alert_thresholds={
                               "warning": {"max": 90.0},
                               "critical": {"max": 95.0}
                           })
        
        self.register_metric("network_in", MetricType.COUNTER, 
                           description="Network bytes received", 
                           unit="bytes")
        
        self.register_metric("network_out", MetricType.COUNTER, 
                           description="Network bytes sent", 
                           unit="bytes")
    
    def _collect_samples(self):
        """Collect system metrics."""
        try:
            # Import these here to avoid requiring them if not using this collector
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_metric("cpu_usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("memory_usage", memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.record_metric("disk_usage", disk.percent)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.record_metric("network_in", net_io.bytes_recv)
            self.record_metric("network_out", net_io.bytes_sent)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")


class ComponentMetricsCollector(MetricsCollector):
    """Collector for component-specific metrics."""
    
    def __init__(self, component_name: str, **kwargs):
        collector_id = kwargs.pop("collector_id", f"{component_name.lower()}-{uuid.uuid4().hex[:8]}")
        super().__init__(collector_id=collector_id, name=f"{component_name} Metrics", 
                         collector_type=CollectorType.COMPONENT, **kwargs)
        
        self.component_name = component_name
        
        # Register common component metrics
        self.register_metric("request_count", MetricType.COUNTER,
                           description=f"Total requests processed by {component_name}",
                           unit="requests")
        
        self.register_metric("error_count", MetricType.COUNTER,
                           description=f"Total errors in {component_name}",
                           unit="errors")
        
        self.register_metric("latency", MetricType.HISTOGRAM,
                           description=f"Request latency in {component_name}",
                           unit="ms",
                           alert_thresholds={
                               "warning": {"avg_max": 500.0, "p95_max": 1000.0},
                               "critical": {"avg_max": 1000.0, "p95_max": 2000.0}
                           })
        
        self.register_metric("active_operations", MetricType.GAUGE,
                           description=f"Active operations in {component_name}",
                           unit="operations")
    
    def record_request(self, duration_ms: float, status: str = "success", 
                      operation_type: str = "request", context: Dict[str, Any] = None):
        """
        Record a request to the component.
        
        Args:
            duration_ms: Request duration in milliseconds
            status: Request status ("success", "error", etc.)
            operation_type: Type of operation
            context: Additional context for the request
        """
        # Increment request counter
        self.record_metric("request_count", 1)
        
        # Record latency
        self.record_metric("latency", duration_ms)
        
        # If error, increment error counter
        if status.lower() == "error":
            self.record_metric("error_count", 1)
        
        # Record active operations (assuming this is called at end of request)
        # For accurate tracking, you'd need separate start/end methods
        self.record_metric("active_operations", -1)
    
    def start_operation(self, operation_type: str = "request", context: Dict[str, Any] = None):
        """
        Record the start of an operation.
        
        Args:
            operation_type: Type of operation
            context: Additional context
            
        Returns:
            Operation ID for tracking
        """
        op_id = str(uuid.uuid4())
        
        # Increment active operations
        self.record_metric("active_operations", 1)
        
        return {
            "op_id": op_id,
            "start_time": time.time(),
            "operation_type": operation_type,
            "context": context or {}
        }
    
    def end_operation(self, operation: Dict[str, Any], status: str = "success"):
        """
        Record the end of an operation.
        
        Args:
            operation: Operation data from start_operation
            status: Operation status
        """
        duration_ms = (time.time() - operation["start_time"]) * 1000
        self.record_request(
            duration_ms=duration_ms,
            status=status,
            operation_type=operation["operation_type"],
            context=operation["context"]
        )


class ABTestMetricsCollector(MetricsCollector):
    """Collector specifically designed for A/B testing metrics."""
    
    def __init__(self, test_id: str, feature_name: str, version_a: str, version_b: str, 
                metrics_to_collect: List[str] = None, **kwargs):
        """
        Initialize an A/B test metrics collector.
        
        Args:
            test_id: ID of the A/B test
            feature_name: Name of the feature being tested
            version_a: Identifier for version A (control)
            version_b: Identifier for version B (experimental)
            metrics_to_collect: List of metrics to collect for each version
            **kwargs: Additional arguments for the base collector
        """
        collector_id = kwargs.pop("collector_id", f"abtest-{test_id}")
        super().__init__(collector_id=collector_id, 
                         name=f"A/B Test: {feature_name} ({version_a} vs {version_b})",
                         collector_type=CollectorType.AB_TEST, **kwargs)
        
        self.test_id = test_id
        self.feature_name = feature_name
        self.version_a = version_a
        self.version_b = version_b
        
        # Default metrics if none specified
        metrics_to_collect = metrics_to_collect or [
            "request_count", "error_count", "latency", "memory_usage", "cpu_usage"
        ]
        
        # Register metrics for version A
        for metric in metrics_to_collect:
            self.register_metric(f"version_a_{metric}", self._get_metric_type(metric),
                               description=f"{metric} for {version_a}",
                               unit=self._get_metric_unit(metric))
            
        # Register metrics for version B
        for metric in metrics_to_collect:
            self.register_metric(f"version_b_{metric}", self._get_metric_type(metric),
                               description=f"{metric} for {version_b}",
                               unit=self._get_metric_unit(metric))
            
        # Track routing decisions
        self.register_metric("routing_decisions", MetricType.COUNTER,
                           description="Number of routing decisions made",
                           unit="decisions")
                           
        # Register success rate metrics for easier comparison
        self.register_metric("version_a_success_rate", MetricType.GAUGE,
                           description=f"Success rate for {version_a}",
                           unit="%")
        
        self.register_metric("version_b_success_rate", MetricType.GAUGE,
                           description=f"Success rate for {version_b}",
                           unit="%")
    
    def _get_metric_type(self, metric_name: str) -> MetricType:
        """Determine the appropriate metric type based on name."""
        if metric_name in ["latency", "response_time", "duration"]:
            return MetricType.HISTOGRAM
        elif metric_name in ["memory_usage", "cpu_usage", "active_operations", "success_rate"]:
            return MetricType.GAUGE
        else:
            return MetricType.COUNTER
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Determine the appropriate unit based on metric name."""
        if metric_name in ["latency", "response_time", "duration"]:
            return "ms"
        elif metric_name in ["memory_usage", "cpu_usage", "success_rate"]:
            return "%"
        elif metric_name in ["bytes_processed", "memory_used"]:
            return "bytes"
        else:
            return "count"
    
    def record_request(self, version: str, metrics: Dict[str, Any]):
        """
        Record metrics for a specific version.
        
        Args:
            version: Version identifier ("version_a" or "version_b")
            metrics: Dictionary of metrics to record
        """
        # Validate version
        if version not in ["version_a", "version_b"]:
            self.logger.warning(f"Invalid version for A/B test: {version}")
            return
        
        # Record each metric
        for metric_name, value in metrics.items():
            self.record_metric(f"{version}_{metric_name}", value)
        
        # Track success rate if we have both requests and errors
        if "request_count" in metrics and "error_count" in metrics:
            requests = metrics["request_count"]
            errors = metrics["error_count"]
            
            if requests > 0:
                success_rate = ((requests - errors) / requests) * 100
                self.record_metric(f"{version}_success_rate", success_rate)
    
    def record_routing_decision(self, selected_version: str):
        """
        Record which version was selected for a request.
        
        Args:
            selected_version: The version that was selected
        """
        self.record_metric("routing_decisions", 1)
        
        # Directly increment for the specific version
        if selected_version == self.version_a:
            self.record_metric("version_a_request_count", 1)
        elif selected_version == self.version_b:
            self.record_metric("version_b_request_count", 1)
    
    def get_comparison_results(self) -> Dict[str, Any]:
        """
        Get a comparison of the results between versions.
        
        Returns:
            Dictionary with comparison metrics
        """
        with self.thread_lock:
            # Ensure metrics are up to date
            self._aggregate_metrics()
            
            metrics_a = {}
            metrics_b = {}
            comparisons = {}
            
            # Extract metrics for each version
            for name, value in self.aggregated_metrics.items():
                if name.startswith("version_a_"):
                    metrics_a[name.replace("version_a_", "")] = value
                elif name.startswith("version_b_"):
                    metrics_b[name.replace("version_b_", "")] = value
            
            # Calculate relative differences
            for metric in set(metrics_a.keys()).intersection(set(metrics_b.keys())):
                if metric in ["request_count", "error_count"]:
                    # For counters, compare totals
                    val_a = metrics_a[metric].get("value", 0)
                    val_b = metrics_b[metric].get("value", 0)
                    
                    if val_a > 0:
                        relative_diff = ((val_b - val_a) / val_a) * 100
                        comparisons[metric] = {
                            "version_a": val_a,
                            "version_b": val_b,
                            "relative_diff": relative_diff,
                            "improved": (relative_diff < 0) if metric == "error_count" else (relative_diff > 0)
                        }
                
                elif metric in ["latency", "duration", "response_time"]:
                    # For latency metrics, compare avg and p95
                    for stat in ["avg", "p95"]:
                        if stat in metrics_a[metric] and stat in metrics_b[metric]:
                            val_a = metrics_a[metric][stat]
                            val_b = metrics_b[metric][stat]
                            
                            if val_a > 0:
                                relative_diff = ((val_b - val_a) / val_a) * 100
                                comparisons[f"{metric}_{stat}"] = {
                                    "version_a": val_a,
                                    "version_b": val_b,
                                    "relative_diff": relative_diff,
                                    "improved": relative_diff < 0  # Lower latency is better
                                }
                
                elif metric == "success_rate":
                    # Compare success rates
                    val_a = metrics_a[metric].get("current", 0)
                    val_b = metrics_b[metric].get("current", 0)
                    
                    absolute_diff = val_b - val_a
                    comparisons["success_rate"] = {
                        "version_a": val_a,
                        "version_b": val_b,
                        "absolute_diff": absolute_diff,
                        "improved": absolute_diff > 0  # Higher success rate is better
                    }
            
            # Determine overall result
            improvements = [comp["improved"] for comp in comparisons.values()]
            overall_improved = sum(improvements) > len(improvements) / 2
            
            result = {
                "test_id": self.test_id,
                "feature_name": self.feature_name,
                "version_a": self.version_a,
                "version_b": self.version_b,
                "request_counts": {
                    "version_a": metrics_a.get("request_count", {}).get("value", 0),
                    "version_b": metrics_b.get("request_count", {}).get("value", 0)
                },
                "metrics_compared": len(comparisons),
                "metrics_improved": sum(improvements),
                "comparisons": comparisons,
                "overall_improved": overall_improved,
                "confidence": self._calculate_confidence(metrics_a, metrics_b),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
    
    def _calculate_confidence(self, metrics_a: Dict[str, Any], 
                            metrics_b: Dict[str, Any]) -> float:
        """
        Calculate a confidence score for the test results.
        
        Args:
            metrics_a: Metrics for version A
            metrics_b: Metrics for version B
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence on sample size
        requests_a = metrics_a.get("request_count", {}).get("value", 0)
        requests_b = metrics_b.get("request_count", {}).get("value", 0)
        
        # Need a minimum number of requests for any confidence
        if requests_a < 30 or requests_b < 30:
            return min(0.5, max(0.1, (min(requests_a, requests_b) / 30) * 0.5))
        
        # Higher confidence with more samples
        sample_confidence = min(0.9, 0.5 + (min(requests_a, requests_b) / 1000) * 0.4)
        
        # Factor in success rate difference
        success_a = metrics_a.get("success_rate", {}).get("current", 0)
        success_b = metrics_b.get("success_rate", {}).get("current", 0)
        diff = abs(success_a - success_b)
        
        # Larger differences give more confidence
        if diff > 5.0:
            diff_confidence = 0.2
        elif diff > 2.0:
            diff_confidence = 0.1
        elif diff > 1.0:
            diff_confidence = 0.05
        else:
            diff_confidence = 0.0
        
        # Factor in latency differences
        latency_a = metrics_a.get("latency", {}).get("avg", 0)
        latency_b = metrics_b.get("latency", {}).get("avg", 0)
        
        if latency_a > 0:
            latency_diff_pct = abs((latency_b - latency_a) / latency_a) * 100
            
            if latency_diff_pct > 20:
                latency_confidence = 0.2
            elif latency_diff_pct > 10:
                latency_confidence = 0.1
            elif latency_diff_pct > 5:
                latency_confidence = 0.05
            else:
                latency_confidence = 0.0
        else:
            latency_confidence = 0.0
        
        return min(1.0, sample_confidence + diff_confidence + latency_confidence)


class PerformanceMonitoringSystem:
    """
    System-wide performance monitoring manager.
    
    This class manages all metrics collectors, provides aggregated results,
    and handles alerting and telemetry for the entire system.
    """
    
    def __init__(self, kernel=None):
        """
        Initialize the performance monitoring system.
        
        Args:
            kernel: Optional reference to the EvoGenesis kernel
        """
        self.kernel = kernel
        self.logger = logging.getLogger(__name__)
        
        # Collectors
        self.collectors = {}
        self.system_collector = None
        self.component_collectors = {}
        self.abtest_collectors = {}
        
        # Alert subscribers
        self.alert_subscribers = []
        
        # Thread pool for parallel metric collection
        self.thread_pool = ThreadPoolExecutor(max_workers=min(10, os.cpu_count() * 2))
        
        # Storage for historical metrics (in-memory cache)
        self.metric_history = {}
        self.max_history_hours = 24  # Keep data for 24 hours
        
        # Create default system metrics collector
        self.initialize_system_collector()
        
        self.logger.info("Performance monitoring system initialized")
    
    def initialize_system_collector(self):
        """Initialize the system metrics collector."""
        try:
            self.system_collector = SystemMetricsCollector(
                sampling_interval=5.0,  # Sample every 5 seconds
                aggregate_interval=60.0  # Aggregate every minute
            )
            self.register_collector(self.system_collector)
            self.system_collector.start()
            self.logger.info("System metrics collector initialized")
        except Exception as e:
            self.logger.error(f"Error initializing system metrics collector: {e}")
    
    def register_collector(self, collector: MetricsCollector):
        """
        Register a metrics collector with the monitoring system.
        
        Args:
            collector: The metrics collector to register
        """
        collector_id = collector.collector_id
        
        if collector_id in self.collectors:
            self.logger.warning(f"Replacing existing collector with ID: {collector_id}")
        
        self.collectors[collector_id] = collector
        
        # Also track by type for easier access
        if collector.collector_type == CollectorType.SYSTEM:
            self.system_collector = collector
        elif collector.collector_type == CollectorType.COMPONENT:
            self.component_collectors[collector.name] = collector
        elif collector.collector_type == CollectorType.AB_TEST:
            test_id = getattr(collector, "test_id", "unknown-test")
            self.abtest_collectors[test_id] = collector
    
    def create_component_collector(self, component_name: str, **kwargs) -> str:
        """
        Create and register a component metrics collector.
        
        Args:
            component_name: Name of the component
            **kwargs: Additional arguments for the collector
            
        Returns:
            ID of the created collector
        """
        collector = ComponentMetricsCollector(component_name=component_name, **kwargs)
        self.register_collector(collector)
        collector.start()
        return collector.collector_id
    
    def create_abtest_collector(self, test_id: str, feature_name: str, 
                              version_a: str, version_b: str, 
                              metrics_to_collect: List[str] = None, **kwargs) -> str:
        """
        Create and register an A/B test metrics collector.
        
        Args:
            test_id: ID of the A/B test
            feature_name: Name of the feature being tested
            version_a: Identifier for version A
            version_b: Identifier for version B
            metrics_to_collect: List of metrics to collect
            **kwargs: Additional arguments for the collector
            
        Returns:
            ID of the created collector
        """
        collector = ABTestMetricsCollector(
            test_id=test_id,
            feature_name=feature_name,
            version_a=version_a,
            version_b=version_b,
            metrics_to_collect=metrics_to_collect,
            **kwargs
        )
        self.register_collector(collector)
        collector.start()
        return collector.collector_id
    
    def get_collector(self, collector_id: str) -> Optional[MetricsCollector]:
        """
        Get a collector by ID.
        
        Args:
            collector_id: ID of the collector
            
        Returns:
            The collector if found, None otherwise
        """
        return self.collectors.get(collector_id)
    
    def get_component_collector(self, component_name: str) -> Optional[ComponentMetricsCollector]:
        """
        Get a component collector by name.
        
        Args:
            component_name: Name of the component
            
        Returns:
            The collector if found, None otherwise
        """
        return self.component_collectors.get(component_name)
    
    def get_abtest_collector(self, test_id: str) -> Optional[ABTestMetricsCollector]:
        """
        Get an A/B test collector by test ID.
        
        Args:
            test_id: ID of the A/B test
            
        Returns:
            The collector if found, None otherwise
        """
        return self.abtest_collectors.get(test_id)
    
    def start_all_collectors(self):
        """Start all registered collectors."""
        for collector_id, collector in self.collectors.items():
            try:
                collector.start()
            except Exception as e:
                self.logger.error(f"Error starting collector {collector_id}: {e}")
    
    def stop_all_collectors(self):
        """Stop all registered collectors."""
        for collector_id, collector in self.collectors.items():
            try:
                collector.stop()
            except Exception as e:
                self.logger.error(f"Error stopping collector {collector_id}: {e}")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from all collectors.
        
        Returns:
            Dictionary with all current metrics
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "collectors": {},
            "system": {},
            "components": {},
            "ab_tests": {}
        }
        
        # Get metrics from each collector
        futures = {}
        for collector_id, collector in self.collectors.items():
            futures[collector_id] = self.thread_pool.submit(collector.get_current_metrics)
        
        # Wait for all to complete and gather results
        for collector_id, future in futures.items():
            try:
                collector_metrics = future.result(timeout=2.0)
                result["collectors"][collector_id] = collector_metrics
                
                # Organize by type
                collector = self.collectors[collector_id]
                if collector.collector_type == CollectorType.SYSTEM:
                    result["system"] = collector_metrics
                elif collector.collector_type == CollectorType.COMPONENT:
                    result["components"][collector.name] = collector_metrics
                elif collector.collector_type == CollectorType.AB_TEST:
                    test_id = getattr(collector, "test_id", "unknown-test")
                    result["ab_tests"][test_id] = collector_metrics
            except Exception as e:
                self.logger.error(f"Error getting metrics from collector {collector_id}: {e}")
        
        return result
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all active alerts from all collectors.
        
        Returns:
            List of active alerts
        """
        all_alerts = []
        
        for collector_id, collector in self.collectors.items():
            try:
                collector_alerts = collector.get_alerts()
                all_alerts.extend(collector_alerts)
            except Exception as e:
                self.logger.error(f"Error getting alerts from collector {collector_id}: {e}")
        
        return sorted(all_alerts, key=lambda a: a.get("timestamp", 0), reverse=True)
    
    def subscribe_to_alerts(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to alerts.
        
        Args:
            callback: Function to call when an alert is triggered
        """
        if callback not in self.alert_subscribers:
            self.alert_subscribers.append(callback)
    
    def unsubscribe_from_alerts(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Unsubscribe from alerts.
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self.alert_subscribers:
            self.alert_subscribers.remove(callback)
    
    async def check_alert_threshold(self, collector_id: str, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        Check if a specific metric is currently triggering an alert.
        
        Args:
            collector_id: ID of the collector
            metric_name: Name of the metric
            
        Returns:
            Alert info if triggered, None otherwise
        """
        collector = self.get_collector(collector_id)
        if not collector:
            return None
        
        alerts = collector.get_alerts()
        for alert in alerts:
            if alert["metric_name"] == metric_name:
                return alert
        
        return None
    
    def get_component_metrics(self, component_name: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get metrics for a specific component, including historical data.
        
        Args:
            component_name: Name of the component
            hours: How many hours of historical data to include
            
        Returns:
            Dictionary with component metrics
        """
        # Get current metrics
        collector = self.get_component_collector(component_name)
        if not collector:
            return {"error": f"Component collector not found: {component_name}"}
        
        current_metrics = collector.get_current_metrics()
        
        # Try to get historical data
        historical = self._get_historical_metrics(
            collector_id=collector.collector_id, 
            hours=min(hours, self.max_history_hours)
        )
        
        return {
            "current": current_metrics,
            "historical": historical,
            "component_name": component_name
        }
    
    def get_abtest_results(self, test_id: str) -> Dict[str, Any]:
        """
        Get results for a specific A/B test.
        
        Args:
            test_id: ID of the A/B test
            
        Returns:
            Dictionary with test results and comparison
        """
        collector = self.get_abtest_collector(test_id)
        if not collector:
            return {"error": f"A/B test collector not found: {test_id}"}
        
        try:
            # Get current metrics
            current_metrics = collector.get_current_metrics()
            
            # Get comparison results
            comparison = collector.get_comparison_results()
            
            return {
                "test_id": test_id,
                "current_metrics": current_metrics,
                "comparison": comparison
            }
        except Exception as e:
            self.logger.error(f"Error getting A/B test results for {test_id}: {e}")
            return {"error": str(e)}
    
    def record_abtest_metrics(self, test_id: str, version: str, metrics: Dict[str, Any]):
        """
        Record metrics for an A/B test.
        
        Args:
            test_id: ID of the A/B test
            version: Version identifier (must be "version_a" or "version_b")
            metrics: Dictionary of metrics to record
        """
        collector = self.get_abtest_collector(test_id)
        if not collector:
            self.logger.warning(f"A/B test collector not found: {test_id}")
            return
        
        try:
            collector.record_request(version, metrics)
        except Exception as e:
            self.logger.error(f"Error recording A/B test metrics for {test_id}: {e}")
    
    def _get_historical_metrics(self, collector_id: str, 
                              hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get historical metrics for a collector.
        
        Args:
            collector_id: ID of the collector
            hours: How many hours of historical data to include
            
        Returns:
            List of historical metric data points
        """
        # This implementation would typically use a database or time-series
        # storage system. For simplicity, we'll just return empty data here.
        # In a production system, this would retrieve historical data from
        # a database or telemetry files.
        
        # For now, just return an empty list
        return []
    
    def export_metrics(self, file_path: str, collectors: List[str] = None):
        """
        Export metrics to a file.
        
        Args:
            file_path: Path to save the metrics
            collectors: Optional list of collector IDs to export (all if None)
        """
        try:
            # Get metrics
            if collectors:
                collector_data = {}
                for collector_id in collectors:
                    if collector_id in self.collectors:
                        collector_data[collector_id] = self.collectors[collector_id].get_current_metrics()
            else:
                collector_data = {
                    collector_id: collector.get_current_metrics()
                    for collector_id, collector in self.collectors.items()
                }
            
            # Export data
            with open(file_path, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "collectors": collector_data
                }, f, indent=2)
                
            self.logger.info(f"Exported metrics to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    def import_metrics(self, file_path: str) -> bool:
        """
        Import metrics from a file.
        
        Args:
            file_path: Path to the metrics file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open and load the metrics file
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Validate the imported data format
                if not isinstance(data, dict) or 'timestamp' not in data or 'collectors' not in data:
                    self.logger.error(f"Invalid metrics file format: {file_path}")
                    return False
                
                # Parse timestamp
                try:
                    import_timestamp = datetime.fromisoformat(data['timestamp'])
                except (ValueError, TypeError):
                    self.logger.error(f"Invalid timestamp format in metrics file: {file_path}")
                    return False
            
            # Process each collector's data
            for collector_id, metrics in data['collectors'].items():
                # Check if this collector exists in our system
                if collector_id in self.collectors:
                    # Store in metric history
                    if collector_id not in self.metric_history:
                        self.metric_history[collector_id] = []
                    
                    self.metric_history[collector_id].append({
                        'timestamp': import_timestamp,
                        'metrics': metrics
                    })
                    
                    # Trim history if it exceeds max_history_hours
                    cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)
                    self.metric_history[collector_id] = [
                        entry for entry in self.metric_history[collector_id]
                        if entry['timestamp'] >= cutoff_time
                    ]
                else:
                    self.logger.warning(f"Imported metrics for unknown collector: {collector_id}")
            
                self.logger.info(f"Successfully imported metrics from {file_path}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error importing metrics: {e}")
            return False
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            self.stop_all_collectors()
            self.thread_pool.shutdown(wait=False)
        except:
            pass

# Make sure math is imported for the percentile function
import math
