"""
LLM Orchestration Module - Manages interactions with language models.

This module is responsible for selecting appropriate language models for tasks,
managing API keys, optimizing prompts, and standardizing API calls across providers.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import json
import time
import yaml
import logging
from enum import Enum
import hashlib
import asyncio
# from datetime import datetime, timedelta # Unused imports

# Provider-specific libraries
# Note: These would be conditionally imported in production code
# import openai # Unused import
from openai import AsyncOpenAI
import anthropic
import google.generativeai as genai

# Try to import llama-cpp, but don't fail if it's not available
# Note: To enable local models, install the 'llama-cpp-python' dependency
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("llama_cpp package not available. Local LLM functionality will be disabled.")


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    CUSTOM = "custom"


class ModelCapability(str, Enum):
    """Capabilities that models can have."""
    GENERAL = "general"
    CODE = "code"
    PLANNING = "planning"
    REASONING = "reasoning"
    CREATIVE = "creative"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    CHAT = "chat"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    TOOL_USE = "tool_use"


class ModelPriority(str, Enum):
    """Priority factors for model selection."""
    QUALITY = "quality"
    COST = "cost"
    SPEED = "speed"
    BALANCED = "balanced"


class APIKeyManager:
    """Manages API keys for different providers securely."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the API Key Manager.
        
        Args:
            config_path: Path to the configuration file with API keys
        """
        self.config_path = config_path or os.path.join("config", "llm_providers.yaml")
        self.api_keys = {}
        self.load_api_keys()
    
    def load_api_keys(self):
        """Load API keys from configuration file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Store keys in memory (in a real system, use a more secure approach)
                if config and "api_keys" in config:
                    self.api_keys = config["api_keys"]
            except Exception as e:
                logging.error(f"Error loading API keys: {str(e)}")
    
    def get_api_key(self, provider: ModelProvider) -> Optional[str]:
        """Get API key for a specific provider."""
        return self.api_keys.get(provider.value)
    
    def set_api_key(self, provider: ModelProvider, key: str) -> bool:
        """Set API key for a specific provider."""
        try:
            self.api_keys[provider.value] = key
            
            # Update config file
            config = {"api_keys": self.api_keys}
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f)
            return True # Corrected indentation
        except Exception as e:
            logging.error(f"Error setting API key: {str(e)}")
            return False
            return False
    
    def configure_client(self, provider: ModelProvider) -> Any:
        """Configure the client for a specific provider using the stored API key."""
        api_key = self.get_api_key(provider)
        
        if not api_key:
            raise ValueError(f"No API key found for provider {provider}")
        
        if provider == ModelProvider.OPENAI:
            client = AsyncOpenAI(api_key=api_key)
            return client
        
        elif provider == ModelProvider.AZURE_OPENAI:
            # For Azure OpenAI, we need more configuration parameters
            # Expecting api_key to be a dict with required Azure OpenAI parameters
            if isinstance(api_key, str):
                # Try to parse JSON if stored as string
                try:
                    config = json.loads(api_key)
                except json.JSONDecodeError:
                    raise ValueError("Azure OpenAI credentials must be a JSON object")
            else:
                config = api_key
            
            # Check for required Azure OpenAI parameters
            required_params = ["api_key", "azure_endpoint", "api_version"]
            if not all(param in config for param in required_params):
                missing = [param for param in required_params if param not in config]
                raise ValueError(f"Missing required Azure OpenAI parameters: {', '.join(missing)}")
            
            # Create Azure OpenAI client
            client = AsyncOpenAI(
                api_key=config["api_key"],
                azure_endpoint=config["azure_endpoint"],
                api_version=config["api_version"],
                azure_deployment=config.get("azure_deployment"),  # Optional, can be specified per request
                azure_ad_token=config.get("azure_ad_token"),      # Optional, for Azure AD authentication
            )
            return client
        
        elif provider == ModelProvider.ANTHROPIC:
            client = anthropic.Anthropic(api_key=api_key)
            return client
        
        elif provider == ModelProvider.GOOGLE:
            genai.configure(api_key=api_key)
            return genai
        
        elif provider == ModelProvider.LOCAL:
            # No API key needed for local models
            return None
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class ModelRegistry:
    """Registry of available models and their capabilities."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Model Registry.
        
        Args:
            config_path: Path to the configuration file with model information
        """
        self.config_path = config_path or os.path.join("config", "models.yaml")
        self.models = {}
        self.benchmarks = {}
        self.cost_models = {}
        self.load_model_data()
    
    def load_model_data(self):
        """Load model data from configuration file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                if config:
                    self.models = config.get("models", {})
                    self.benchmarks = config.get("benchmarks", {})
                    self.cost_models = config.get("cost_models", {})
            except Exception as e:
                logging.error(f"Error loading model data: {str(e)}")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return self.models.get(model_name)
    
    def get_model_capabilities(self, model_name: str) -> List[str]:
        """Get capabilities of a specific model."""
        model_info = self.get_model_info(model_name)
        if model_info:
            return model_info.get("capabilities", [])
        return []
    
    def get_models_by_capability(self, capability: str) -> List[str]:
        """Get models that have a specific capability."""
        return [
            model_name
            for model_name, model_info in self.models.items()
            if capability in model_info.get("capabilities", [])
        ]
    
    def get_model_benchmark(self, model_name: str, benchmark_type: str) -> Optional[float]:
        """Get benchmark score for a specific model and benchmark type."""
        if benchmark_type in self.benchmarks and model_name in self.benchmarks[benchmark_type]:
            return self.benchmarks[benchmark_type][model_name]
        return None
    
    def get_model_cost(self, model_name: str, tokens: int) -> float:
        """Calculate the cost of using a specific model for a given number of tokens."""
        if model_name in self.cost_models:
            cost_info = self.cost_models[model_name]
            input_cost = cost_info.get("input_cost_per_1k", 0) * (tokens / 1000)
            output_cost = cost_info.get("output_cost_per_1k", 0) * (tokens / 1000)
            return input_cost + output_cost
        return 0.0
    
    def update_benchmark(self, model_name: str, benchmark_type: str, score: float):
        """Update benchmark score for a specific model."""
        if benchmark_type not in self.benchmarks:
            self.benchmarks[benchmark_type] = {}
        
        self.benchmarks[benchmark_type][model_name] = score
        self._save_model_data()
    
    def add_model(self, model_name: str, model_info: Dict[str, Any]):
        """Add a new model to the registry."""
        self.models[model_name] = model_info
        self._save_model_data()
    
    def _save_model_data(self):
        """Save model data to configuration file."""
        try:
            config = {
                "models": self.models,
                "benchmarks": self.benchmarks,
                "cost_models": self.cost_models
            }
            
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f)
        except Exception as e:
            logging.error(f"Error saving model data: {str(e)}")


class PromptManager:
    """Manages prompt templates and optimization."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the Prompt Manager.
        
        Args:
            templates_dir: Directory containing prompt templates
        """
        self.templates_dir = templates_dir or os.path.join("config", "prompts")
        self.templates = {}
        self.load_templates()
        
        # Cache for optimized prompts
        self.optimized_prompts = {}
    
    def load_templates(self):
        """Load prompt templates from directory."""
        if os.path.exists(self.templates_dir):
            for filename in os.listdir(self.templates_dir):
                if filename.endswith((".yaml", ".yml", ".jinja2")):
                    template_name = os.path.splitext(filename)[0]
                    template_path = os.path.join(self.templates_dir, filename)
                    
                    try:
                        with open(template_path, 'r') as f:
                            if filename.endswith((".yaml", ".yml")):
                                self.templates[template_name] = yaml.safe_load(f)
                            else:
                                self.templates[template_name] = f.read()
                    except Exception as e:
                        logging.error(f"Error loading template {template_name}: {str(e)}")
    
    def get_template(self, template_name: str) -> Optional[Union[str, Dict[str, Any]]]:
        """Get a prompt template by name."""
        return self.templates.get(template_name)
    
    def format_prompt(self, template_name: str, params: Dict[str, Any]) -> str:
        """Format a prompt template with parameters."""
        template = self.get_template(template_name)
        
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        if isinstance(template, str):
            # Simple string template
            for key, value in params.items():
                template = template.replace(f"{{{{{key}}}}}", str(value))
            
            return template
        
        elif isinstance(template, dict) and "prompt" in template:
            # YAML template with prompt field
            prompt = template["prompt"]
            
            for key, value in params.items():
                prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
            
            return prompt
        
        else:
            raise ValueError(f"Invalid template format for {template_name}")
    
    def get_optimized_prompt(self, template_name: str, params: Dict[str, Any], 
                            model_name: str) -> str:
        """
        Get an optimized version of a prompt for a specific model.
        
        This could use A/B testing or other optimization techniques.
        """
        # Create a cache key based on template, params, and model
        param_str = json.dumps(params, sort_keys=True)
        cache_key = f"{template_name}:{model_name}:{hashlib.md5(param_str.encode()).hexdigest()}"
        
        # Check if we have a cached optimized prompt
        if cache_key in self.optimized_prompts:
            return self.optimized_prompts[cache_key]
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")

        prompt_text_to_format = None

        # Prioritize model-specific prompt if available within a dictionary template
        if isinstance(template, dict):
            model_specific_config = template.get("model_specific", {}).get(model_name, {})
            # Check if model_specific_config itself is the prompt string or contains a 'prompt' key
            if isinstance(model_specific_config, str):
             prompt_text_to_format = model_specific_config
            elif isinstance(model_specific_config, dict):
             prompt_text_to_format = model_specific_config.get("prompt")

            # Fallback to the main 'prompt' key if no model-specific one found
            if prompt_text_to_format is None:
                prompt_text_to_format = template.get("prompt")

        # If it's a simple string template, use the template itself
        elif isinstance(template, str):
             prompt_text_to_format = template

        # If still no specific prompt text identified, raise error
        if prompt_text_to_format is None or not isinstance(prompt_text_to_format, str):
             raise ValueError(f"Could not determine prompt string for template {template_name} and model {model_name}. Ensure the template is a string or a dict with a 'prompt' key (and optionally model-specific prompts).")

        # Format the selected prompt string using the provided parameters
        # Replicates the formatting logic from format_prompt for a simple string
        prompt = prompt_text_to_format
        for key, value in params.items():
            # Basic placeholder replacement, consider using more robust templating if needed
            prompt = prompt.replace(f"{{{{{key}}}}}", str(value))

        # Cache the result
        self.optimized_prompts[cache_key] = prompt
        
        return prompt


class ExecutionTracker:
    """Tracks LLM executions for performance monitoring and cost analysis."""
    
    def __init__(self, kernel):
        """
        Initialize the Execution Tracker.
        
        Args:
            kernel: The EvoGenesis kernel instance
        """
        self.kernel = kernel
        self.executions = []
        self.execution_stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_latency": 0.0
        }
    
    def record_execution(self, execution_data: Dict[str, Any]):
        """Record an LLM execution."""
        self.executions.append(execution_data)
        
        # Update stats
        self.execution_stats["total_calls"] += 1
        self.execution_stats["total_tokens"] += execution_data.get("total_tokens", 0)
        self.execution_stats["total_cost"] += execution_data.get("cost", 0.0)
        
        # Update average latency
        latency = execution_data.get("latency", 0.0)
        prev_avg = self.execution_stats["average_latency"]
        prev_count = self.execution_stats["total_calls"] - 1
        
        if prev_count > 0:
            self.execution_stats["average_latency"] = (
                (prev_avg * prev_count + latency) / 
                self.execution_stats["total_calls"]
            )
        else:
            self.execution_stats["average_latency"] = latency
        
        # Store in memory manager if available
        if hasattr(self.kernel, "memory_manager"):
            self.kernel.memory_manager.store_llm_execution(execution_data)
    
    def get_execution_stats(self, model_name: Optional[str] = None, 
                          time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Get execution statistics, optionally filtered by model and time range."""
        if not model_name and not time_range:
            return self.execution_stats
        
        # Filter executions
        filtered_executions = self.executions
        
        if model_name:
            filtered_executions = [
                e for e in filtered_executions
                if e.get("model_name") == model_name
            ]
        
        if time_range:
            start_time, end_time = time_range
            filtered_executions = [
                e for e in filtered_executions
                if start_time <= e.get("timestamp", 0) <= end_time
            ]
        
        # Calculate stats for filtered executions
        if not filtered_executions:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "average_latency": 0.0
            }
        
        total_calls = len(filtered_executions)
        total_tokens = sum(e.get("total_tokens", 0) for e in filtered_executions)
        total_cost = sum(e.get("cost", 0.0) for e in filtered_executions)
        average_latency = sum(e.get("latency", 0.0) for e in filtered_executions) / total_calls
        
        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "average_latency": average_latency
        }


class LLMOrchestrator:
    """
    Manages interactions with language models in the EvoGenesis framework.
    
    Responsible for:
    - Selecting appropriate models for tasks
    - Managing API keys and interactions with model providers
    - Executing LLM calls and standardizing responses
    - Optimizing prompts for different models
    - Tracking usage, cost, and performance
    """
    
    def __init__(self, kernel):
        """
        Initialize the LLM Orchestrator.
        
        Args:
            kernel: The EvoGenesis kernel instance
        """
        self.kernel = kernel
        self.api_key_manager = APIKeyManager()
        self.model_registry = ModelRegistry()
        self.prompt_manager = PromptManager()
        self.execution_tracker = ExecutionTracker(kernel)
        
        # Configuration
        self.default_model = "gpt-4o"
        self.fallback_models = ["gpt-3.5-turbo", "claude-3-haiku"]
        self.cost_limits = {
            "default": 0.05,  # Default cost limit per call
            "high_priority": 0.25  # Higher limit for important tasks
        }
    
    def start(self):
        """Start the LLM Orchestrator module."""
        # Initialize clients for each provider
        for provider in ModelProvider:
            try:
                # Skip local provider check for API key
                if provider == ModelProvider.LOCAL:
                    if not LLAMA_CPP_AVAILABLE:
                        logging.warning(f"Skipping {provider} initialization: llama_cpp package not available.")
                        continue
                # Skip other providers if no API key is configured
                elif not self.api_key_manager.get_api_key(provider):
                    logging.info(f"Skipping {provider} initialization: No API key configured.")
                    continue
                
                # Configure client
                self.api_key_manager.configure_client(provider)
                logging.info(f"Successfully initialized client for {provider}") # Added info log
            except Exception as e:
                logging.warning(f"Failed to initialize client for {provider}: {str(e)}")
    
    def stop(self):
        """Stop the LLM Orchestrator module."""
        # Cleanup if needed
        pass
    
    def get_status(self):
        """Get the current status of the LLM Orchestrator."""
        stats = self.execution_tracker.get_execution_stats()
        
        return {
            "status": "active",
            "total_calls": stats["total_calls"],
            "total_tokens": stats["total_tokens"],
            "total_cost": stats["total_cost"],
            "available_providers": [
                provider.value for provider in ModelProvider
                if self.api_key_manager.get_api_key(provider) or provider == ModelProvider.LOCAL
            ]
        }
    def select_model(self, task_type: str, capabilities: Optional[List[str]] = None,
                   performance_requirements: Optional[Dict[str, Any]] = None,
                   budget_constraints: Optional[Dict[str, Any]] = None,
                   agent_type: Optional[str] = None) -> Dict[str, Any]: # TODO: agent_type parameter is unused
        """
        Select the most appropriate model for a task.
        Select the most appropriate model for a task.
        
        Args:
            task_type: Type of task (e.g., "summarization", "code_generation")
            capabilities: Required capabilities for the task
            performance_requirements: Performance requirements (e.g., priority, latency)
            budget_constraints: Budget constraints for model selection
            agent_type: Type of agent that will use the model
            
        Returns:
            Dictionary with model selection information
        """
        # Determine required capabilities based on task type
        required_capabilities = capabilities or []
        if task_type:
            # Map task types to capabilities
            task_capability_map = {
                "code_generation": [ModelCapability.CODE],
                "code_review": [ModelCapability.CODE, ModelCapability.REASONING],
                "summarization": [ModelCapability.SUMMARIZATION],
                "planning": [ModelCapability.PLANNING],
                "reasoning": [ModelCapability.REASONING],
                "creative": [ModelCapability.CREATIVE],
                "extraction": [ModelCapability.EXTRACTION],
                "classification": [ModelCapability.CLASSIFICATION],
                "agent_execution": [ModelCapability.REASONING, ModelCapability.TOOL_USE]
            }
            
            if task_type in task_capability_map:
                for cap in task_capability_map[task_type]:
                    if cap.value not in required_capabilities:
                        required_capabilities.append(cap.value)
        
        # Determine priority (quality vs. cost vs. speed)
        if performance_requirements and "priority" in performance_requirements:
            priority = performance_requirements["priority"]
        else:
            priority = ModelPriority.BALANCED
        
        # Get cost limit
        if budget_constraints and "cost_limit" in budget_constraints:
            cost_limit = budget_constraints["cost_limit"]
        elif priority == ModelPriority.QUALITY:
            cost_limit = self.cost_limits.get("high_priority", 0.25)
        else:
            cost_limit = self.cost_limits.get("default", 0.05)
        
        # Find models with required capabilities
        candidate_models = []
        
        for model_name, model_info in self.model_registry.models.items():
            model_capabilities = model_info.get("capabilities", [])
            
            # Check if model has all required capabilities
            if all(cap in model_capabilities for cap in required_capabilities):
                # Calculate a score for this model
                score = self._calculate_model_score(
                    model_name=model_name,
                    priority=priority,
                    task_type=task_type,
                    cost_limit=cost_limit
                )
                
                candidate_models.append({
                    "model_name": model_name,
                    "score": score,
                    "provider": model_info.get("provider"),
                    "config": model_info.get("config", {})
                })
        
        # Sort by score (higher is better)
        candidate_models.sort(key=lambda m: m["score"], reverse=True)
        
        # If no suitable models found, use default model
        if not candidate_models:
            default_model_info = self.model_registry.get_model_info(self.default_model)
            if default_model_info:
                return {
                    "model_name": self.default_model,
                    "provider": default_model_info.get("provider"),
                    "config": default_model_info.get("config", {}),
                    "fallback": True
                }
            else:
                # Last resort
                return {
                    "model_name": "gpt-3.5-turbo",
                    "provider": ModelProvider.OPENAI.value,
                    "config": {},
                    "fallback": True
                }
        
        # Return the best model
        best_model = candidate_models[0]
        best_model["fallback"] = False
        return best_model
    
    def _calculate_model_score(self, model_name: str, priority: str, 
                             task_type: str, cost_limit: float) -> float:
        """
        Calculate a score for a model based on various factors.
        
        Args:
            model_name: Name of the model
            priority: Priority for selection (quality, cost, speed, balanced)
            task_type: Type of task
            cost_limit: Maximum cost allowed
            
        Returns:
            Score for the model (higher is better)
        """
        model_info = self.model_registry.get_model_info(model_name)
        if not model_info:
            return 0.0
        
        # Get benchmark scores
        quality_score = 0.0
        benchmark_types = {
            "code_generation": "code_bench",
            "summarization": "summarization_bench",
            "reasoning": "reasoning_bench",
            "general": "general_bench"
        }
        
        # Try to get a benchmark score specific to the task type
        if task_type in benchmark_types:
            quality_score = self.model_registry.get_model_benchmark(
                model_name, benchmark_types[task_type]
            ) or 0.0
        
        # Fall back to general benchmark if no specific one found
        if quality_score == 0.0:
            quality_score = self.model_registry.get_model_benchmark(
                model_name, "general_bench"
            ) or 0.5  # Default to middle score
        
        # Calculate cost score (higher for lower cost)
        # Assume average request is 1000 tokens input, 500 tokens output
        cost = self.model_registry.get_model_cost(model_name, 1500)
        cost_score = 1.0 - min(1.0, cost / cost_limit)
        
        # Get speed score
        speed_score = model_info.get("average_latency_score", 0.5)
        
        # Calculate final score based on priority
        if priority == ModelPriority.QUALITY:
            return quality_score * 0.7 + cost_score * 0.1 + speed_score * 0.2
        
        elif priority == ModelPriority.COST:
            return quality_score * 0.2 + cost_score * 0.7 + speed_score * 0.1
        
        elif priority == ModelPriority.SPEED:
            return quality_score * 0.2 + cost_score * 0.1 + speed_score * 0.7
        
        else:  # BALANCED
            return quality_score * 0.4 + cost_score * 0.3 + speed_score * 0.3
    
    async def execute_prompt_async(self, task_type: str, prompt_template: str,
                                  params: Dict[str, Any], model_selection: Optional[Dict[str, Any]] = None,
                                  max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Execute a prompt using a selected LLM asynchronously.
        
        Args:
            task_type: Type of task
            prompt_template: Name of the prompt template to use
            params: Parameters for the prompt template
            model_selection: Optional model selection override
            max_tokens: Maximum tokens for the response
            
        Returns:
            Dictionary with the LLM response and metadata
        """
        # Select model if not provided
        if not model_selection:
            model_selection = self.select_model(task_type=task_type)
        
        model_name = model_selection["model_name"]
        provider = model_selection["provider"]
        
        # Format the prompt
        prompt = self.prompt_manager.get_optimized_prompt(
            template_name=prompt_template,
            params=params,
            model_name=model_name
        )
          # Track timing
        start_time = time.time()
        
        try:
            # Execute the prompt based on provider
            if provider == ModelProvider.OPENAI.value:
                client = self.api_key_manager.configure_client(ModelProvider.OPENAI)
                
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                             {"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                
                # Extract response text
                result = response.choices[0].message.content
                
                # Calculate usage and cost
                total_tokens = response.usage.total_tokens
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                  # Calculate cost
                cost_info = self.model_registry.cost_models.get(model_name, {})
                input_cost = cost_info.get("input_cost_per_1k", 0) * (input_tokens / 1000)
                output_cost = cost_info.get("output_cost_per_1k", 0) * (output_tokens / 1000)
                total_cost = input_cost + output_cost
            
            elif provider == ModelProvider.AZURE_OPENAI.value:
                client = self.api_key_manager.configure_client(ModelProvider.AZURE_OPENAI)
                
                # Get Azure deployment from model_selection if available
                config = model_selection.get("config", {})
                deployment = config.get("deployment")
                
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                             {"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7,
                    deployment_id=deployment  # Use deployment specified in model config if available
                )
                
                # Extract response text
                result = response.choices[0].message.content
                
                # Calculate usage and cost
                total_tokens = response.usage.total_tokens
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                
                # Calculate cost (using custom Azure cost model if available)
                cost_info = self.model_registry.cost_models.get(f"azure_{model_name}", {})
                if not cost_info:
                    # Fall back to standard model cost if no Azure-specific cost defined
                    cost_info = self.model_registry.cost_models.get(model_name, {})
                input_cost = cost_info.get("input_cost_per_1k", 0) * (input_tokens / 1000)
                output_cost = cost_info.get("output_cost_per_1k", 0) * (output_tokens / 1000)
                total_cost = input_cost + output_cost
            
            elif provider == ModelProvider.ANTHROPIC.value:
                client = self.api_key_manager.configure_client(ModelProvider.ANTHROPIC)
                
                response = client.messages.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                
                # Extract response text
                result = response.content[0].text
                
                # Calculate usage and cost
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_tokens = input_tokens + output_tokens
                
                # Calculate cost
                cost_info = self.model_registry.cost_models.get(model_name, {})
                input_cost = cost_info.get("input_cost_per_1k", 0) * (input_tokens / 1000)
                output_cost = cost_info.get("output_cost_per_1k", 0) * (output_tokens / 1000)
                total_cost = input_cost + output_cost
            
            elif provider == ModelProvider.GOOGLE.value:
                client = self.api_key_manager.configure_client(ModelProvider.GOOGLE)
                
                model = client.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                
                # Extract response text
                result = response.text
                
                # Estimate usage (Google doesn't provide token counts)
                # This is a very rough estimate
                input_tokens = len(prompt) / 4  # Rough estimate: 4 chars per token
                output_tokens = len(result) / 4
                total_tokens = input_tokens + output_tokens
                
                # Calculate cost                cost_info = self.model_registry.cost_models.get(model_name, {})
                input_cost = cost_info.get("input_cost_per_1k", 0) * (input_tokens / 1000)
                output_cost = cost_info.get("output_cost_per_1k", 0) * (output_tokens / 1000)
                total_cost = input_cost + output_cost
            
            elif provider == ModelProvider.LOCAL.value:
                # Check if llama_cpp is available
                if not LLAMA_CPP_AVAILABLE:
                    raise ValueError(f"Cannot use local model {model_name}: llama_cpp package is not installed")
                
                # Use llama.cpp for local models
                model_path = model_selection.get("config", {}).get("model_path")
                if not model_path:
                    raise ValueError(f"Model path not specified for local model {model_name}")
                
                # Initialize the model
                llm = Llama(model_path=model_path)
                
                # Generate response
                response = llm(prompt, max_tokens=max_tokens)
                
                # Extract response text
                result = response["choices"][0]["text"]
                
                # Estimate usage
                input_tokens = len(prompt) / 4
                output_tokens = len(result) / 4
                total_tokens = input_tokens + output_tokens
                
                # Local models have no cost
                total_cost = 0.0
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Parse result (assume JSON if task expects it)
            parsed_result = result
            if task_type in ["extraction", "classification", "goal_decomposition", "capability_extraction"]:
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    import re
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', result)
                    if json_match:
                        try:
                            parsed_result = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            # Fall back to text
                            parsed_result = result
                    else:
                        # Fall back to text
                        parsed_result = result
            
            # Record execution
            execution_data = {
                "task_type": task_type,
                "model_name": model_name,
                "provider": provider,
                "timestamp": start_time,
                "latency": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost": total_cost,
                "success": True
            }
            
            self.execution_tracker.record_execution(execution_data)
            
            # Return result
            return {
                "result": parsed_result,
                "model_name": model_name,
                "provider": provider,
                "latency": latency,
                "tokens": total_tokens,
                "cost": total_cost
            }
            
        except Exception as e:
            # Record failed execution
            latency = time.time() - start_time
            
            execution_data = {
                "task_type": task_type,
                "model_name": model_name,
                "provider": provider,
                "timestamp": start_time,
                "latency": latency,
                "error": str(e),
                "success": False
            }
            
            self.execution_tracker.record_execution(execution_data)
            
            # Try fallback models if available
            if model_name not in self.fallback_models:
                for fallback_model in self.fallback_models:
                    fallback_model_info = self.model_registry.get_model_info(fallback_model)
                    if fallback_model_info:
                        logging.warning(f"Using fallback model {fallback_model} due to error: {str(e)}")
                        
                        return await self.execute_prompt_async(
                            task_type=task_type,
                            prompt_template=prompt_template,
                            params=params,
                            model_selection={
                                "model_name": fallback_model,
                                "provider": fallback_model_info.get("provider"),
                                "config": fallback_model_info.get("config", {}),
                                "fallback": True
                            },
                            max_tokens=max_tokens
                        )
            
            # If no fallback or fallback failed, raise the error
            raise
    
    def execute_prompt(self, task_type: str, prompt_template: str,
                      params: Dict[str, Any], model_selection: Optional[Dict[str, Any]] = None,
                      max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Synchronous wrapper for execute_prompt_async.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.execute_prompt_async(
                    task_type=task_type,
                    prompt_template=prompt_template,
                    params=params,
                    model_selection=model_selection,
                    max_tokens=max_tokens
                )
            )
        finally:
            loop.close()
    
    def run_a_b_test(self, task_type: str, prompt_template: str, params: Dict[str, Any],
                    model_a: str, model_b: str, samples: int = 5) -> Dict[str, Any]:
        """
        Run an A/B test comparing two models on the same prompt.
        
        Args:
            task_type: Type of task
            prompt_template: Name of the prompt template to use
            params: Parameters for the prompt template
            model_a: First model to test
            model_b: Second model to test
            samples: Number of samples to run for each model
            
        Returns:
            Dictionary with test results
        """
        model_a_info = self.model_registry.get_model_info(model_a)
        model_b_info = self.model_registry.get_model_info(model_b)
        
        if not model_a_info or not model_b_info:
            raise ValueError("One or both models not found in registry")
        
        results_a = []
        results_b = []
        
        # Run tests for model A
        for i in range(samples):
            try:
                result = self.execute_prompt(
                    task_type=task_type,
                    prompt_template=prompt_template,
                    params=params,
                    model_selection={
                        "model_name": model_a,
                        "provider": model_a_info.get("provider"),
                        "config": model_a_info.get("config", {})
                    }
                )
                results_a.append(result)
            except Exception as e:
                logging.error(f"Error running model A ({model_a}) test {i+1}: {str(e)}")
        
        # Run tests for model B
        for i in range(samples):
            try:
                result = self.execute_prompt(
                    task_type=task_type,
                    prompt_template=prompt_template,
                    params=params,
                    model_selection={
                        "model_name": model_b,
                        "provider": model_b_info.get("provider"),
                        "config": model_b_info.get("config", {})
                    }
                )
                results_b.append(result)
            except Exception as e:
                logging.error(f"Error running model B ({model_b}) test {i+1}: {str(e)}")
        
        # Calculate metrics
        model_a_metrics = {
            "avg_latency": sum(r["latency"] for r in results_a) / max(1, len(results_a)),
            "avg_tokens": sum(r["tokens"] for r in results_a) / max(1, len(results_a)),
            "avg_cost": sum(r["cost"] for r in results_a) / max(1, len(results_a)),
            "success_rate": len(results_a) / samples
        }
        
        model_b_metrics = {
            "avg_latency": sum(r["latency"] for r in results_b) / max(1, len(results_b)),
            "avg_tokens": sum(r["tokens"] for r in results_b) / max(1, len(results_b)),
            "avg_cost": sum(r["cost"] for r in results_b) / max(1, len(results_b)),
            "success_rate": len(results_b) / samples
        }
        
        return {
            "model_a": {
                "name": model_a,
                "metrics": model_a_metrics,
                "results": results_a
            },
            "model_b": {
                "name": model_b,
                "metrics": model_b_metrics,
            }
        }

    def update_models_from_benchmarks(self, benchmark_url: str = None):
        """
        Update model benchmarks from external sources.

        Args:
            benchmark_url: URL of benchmark data (optional)
        """
        try:
            # First try to fetch from provided URL
            if benchmark_url:
                logging.info(f"Fetching benchmark data from {benchmark_url}")
                try:
                    import requests
                    response = requests.get(benchmark_url, timeout=10)
                    response.raise_for_status()  # Raise exception for non-200 response
                    benchmark_data = response.json()
                    logging.info(f"Successfully fetched benchmark data from {benchmark_url}")
                except Exception as e:
                    logging.error(f"Failed to fetch benchmark data from {benchmark_url}: {str(e)}")
                    benchmark_data = self._get_fallback_benchmark_data()
            else:
                # Try to fetch from default benchmark sources
                benchmark_data = self._fetch_from_default_sources()

            # Update registry with fetched benchmark data
            for bench_type, bench_data in benchmark_data.items():
                for model_name, score in bench_data.items():
                    self.model_registry.update_benchmark(model_name, bench_type, score)

            # Save benchmarks to cache for future use
            self._save_benchmark_cache(benchmark_data)

            logging.info(f"Updated model benchmarks for {sum(len(models) for models in benchmark_data.values())} models")
            return benchmark_data

        except Exception as e:
            logging.error(f"Error updating model benchmarks: {str(e)}")
            return {}

    def _fetch_from_default_sources(self) -> Dict[str, Dict[str, float]]:
        """
        Fetch benchmark data from default sources.
        
        Returns:
            Dictionary of benchmark data
        """
        benchmark_data = {}
        sources = [
            {"name": "lmsys_arena", "url": "https://huggingface.co/datasets/lmsys/chatbot_arena_archive/raw/main/summary_stats.json"},
            {"name": "open_llm_leaderboard", "url": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/raw/main/leaderboard_data.json"},
            {"name": "evogenesis_internal", "url": os.path.join("data", "benchmarks", "model_benchmarks.json")}
        ]
        
        for source in sources:
            try:
                if source["name"] == "evogenesis_internal":
                    # Local file
                    if os.path.exists(source["url"]):
                        with open(source["url"], 'r') as f:
                            data = json.load(f)
                            self._merge_benchmark_data(benchmark_data, data)
                            logging.info(f"Loaded benchmark data from local file: {source['url']}")
                else:
                    # Remote URL
                    import requests
                    response = requests.get(source["url"], timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        processed_data = self._process_benchmark_source(source["name"], data)
                        self._merge_benchmark_data(benchmark_data, processed_data)
                        logging.info(f"Loaded benchmark data from {source['name']}")
            except Exception as e:
                logging.warning(f"Failed to fetch benchmark data from {source['name']}: {str(e)}")
        
        # If no data was fetched, use fallback data
        if not benchmark_data:
            return self._get_fallback_benchmark_data()
            
        return benchmark_data
    
    def _process_benchmark_source(self, source_name: str, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Process benchmark data from a specific source.
        
        Args:
            source_name: Name of the benchmark source
            data: Raw benchmark data
            
        Returns:
            Processed benchmark data
        """
        processed_data = {}
        
        if source_name == "lmsys_arena":
            # Process LMSYS Chatbot Arena data
            processed_data["general_bench"] = {}
            
            # Extract Elo ratings and normalize them to 0-1 range
            if "elo_ratings" in data:
                ratings = [(model["model"], model["rating"]) for model in data["elo_ratings"]]
                if ratings:
                    min_rating = min(r[1] for r in ratings)
                    max_rating = max(r[1] for r in ratings)
                    rating_range = max_rating - min_rating
                    
                    for model_name, rating in ratings:
                        # Normalize and convert model name to match our registry
                        normalized_name = self._normalize_model_name(model_name)
                        if normalized_name:
                            normalized_score = (rating - min_rating) / rating_range
                            processed_data["general_bench"][normalized_name] = min(0.99, max(0.01, normalized_score))
        
        elif source_name == "open_llm_leaderboard":
            # Process Open LLM Leaderboard data
            processed_data["general_bench"] = {}
            processed_data["reasoning_bench"] = {}
            
            if "results" in data:
                for result in data["results"]:
                    model_name = self._normalize_model_name(result.get("model", ""))
                    if model_name:
                        # Overall average score
                        if "average_score" in result:
                            processed_data["general_bench"][model_name] = min(0.99, max(0.01, result["average_score"] / 100))
                        
                        # Task-specific scores
                        if "reasoning" in result.get("results", {}):
                            processed_data["reasoning_bench"][model_name] = min(0.99, max(0.01, result["results"]["reasoning"] / 100))
        
        # Add more source processors as needed
        
        return processed_data
    
    def _normalize_model_name(self, source_model_name: str) -> Optional[str]:
        """
        Normalize model names from external sources to match our registry.
        
        Args:
            source_model_name: Model name from external source
            
        Returns:
            Normalized model name or None if no match
        """
        # Use dynamically loaded mappings (assuming self.name_mappings is loaded, e.g., in __init__)
        # Ensure mappings are available, potentially loading defaults if not already loaded.
        if not hasattr(self, 'name_mappings') or not self.name_mappings:
             # In a full implementation, this might call a method like _load_or_set_default_name_mappings()
             logging.warning("Model name mappings not loaded before calling _normalize_model_name. Normalization might be incomplete.")
             # Provide a minimal default or empty dict to avoid crashing
             current_mappings = {}
        else:
             current_mappings = self.name_mappings

        # Try direct match using the loaded mappings
        if source_model_name in current_mappings:
            return current_mappings[source_model_name]

        # Try fuzzy matching: check if known mapping keys are substrings of the source name.
        # Sort keys by length descending to prioritize longer, more specific matches
        # (e.g., match "gpt-4-turbo" before "gpt-4").
        source_model_name_lower = source_model_name.lower()
        sorted_mapping_keys = sorted(current_mappings.keys(), key=len, reverse=True)

        for pattern in sorted_mapping_keys:
            pattern_lower = pattern.lower()
            if pattern_lower in source_model_name_lower:
             # Found a potential match based on substring.
             # Consider adding word boundary checks for more accuracy if needed.
             normalized_name = current_mappings[pattern]
             logging.debug(f"Normalized '{source_model_name}' to '{normalized_name}' using fuzzy match on pattern '{pattern}'")
             return normalized_name

        # As a fallback, check if the source name itself is directly usable
        # (i.e., already exists in our model registry)
        if hasattr(self, 'model_registry') and source_model_name in self.model_registry.models:
             logging.debug(f"Normalized '{source_model_name}' to itself as it exists in the model registry.")
             return source_model_name

        logging.warning(f"Could not normalize model name: '{source_model_name}'. No direct or fuzzy match found in mappings, and not found directly in registry.")
        # No match found after checking mappings and registry
        return None

    def _merge_benchmark_data(self, target: Dict[str, Dict[str, float]], source: Dict[str, Dict[str, float]]):
        """
        Merge benchmark data from source into target.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for bench_type, bench_data in source.items():
            if bench_type not in target:
                target[bench_type] = {}
            
            for model_name, score in bench_data.items():
                # Only update if model not already in target or if score is higher
                if model_name not in target[bench_type] or score > target[bench_type][model_name]:
                    target[bench_type][model_name] = score
    
    def _save_benchmark_cache(self, benchmark_data: Dict[str, Dict[str, float]]):
        """
        Save benchmark data to cache.
        
        Args:
            benchmark_data: Benchmark data to save
        """
        try:
            cache_dir = os.path.join("data", "benchmarks")
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_path = os.path.join(cache_dir, "benchmark_cache.json")
            with open(cache_path, 'w') as f:
                json.dump({
                    "timestamp": time.time(),
                    "data": benchmark_data
                }, f, indent=2)
            
            logging.info(f"Saved benchmark cache to {cache_path}")
        except Exception as e:
            logging.warning(f"Failed to save benchmark cache: {str(e)}")
    
    def _get_fallback_benchmark_data(self) -> Dict[str, Dict[str, float]]:
        """
        Get fallback benchmark data when external sources are unavailable.
        
        Returns:
            Dictionary of fallback benchmark data
        """
        # First try to use cached data if available
        try:
            cache_path = os.path.join("data", "benchmarks", "benchmark_cache.json")
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
                
                # Check if cache is still valid (less than 7 days old)
                if time.time() - cache.get("timestamp", 0) < 604800:  # 7 days in seconds
                    logging.info("Using cached benchmark data")
                    return cache["data"]
        except Exception as e:
            logging.warning(f"Failed to load benchmark cache: {str(e)}")
        
        # Fallback benchmark data based on research and model evaluations
        return {
            "general_bench": {
                "gpt-4o": 0.95,
                "gpt-4-turbo": 0.92,
                "claude-3-opus": 0.93,
                "claude-3-sonnet": 0.88,
                "claude-3-haiku": 0.82,
                "gpt-3.5-turbo": 0.80,
                "gemini-pro": 0.84,
                "llama-2-70b": 0.79,
                "mixtral-8x7b": 0.82
            },
            "code_bench": {
                "gpt-4o": 0.96,
                "gpt-4-turbo": 0.93,
                "claude-3-opus": 0.91,
                "claude-3-sonnet": 0.85,
                "claude-3-haiku": 0.78,
                "gpt-3.5-turbo": 0.75,
                "gemini-pro": 0.82,
                "llama-2-70b": 0.73,
                "mixtral-8x7b": 0.76
            },
            "reasoning_bench": {
                "gpt-4o": 0.94,
                "gpt-4-turbo": 0.91,
                "claude-3-opus": 0.92,
                "claude-3-sonnet": 0.87,
                "claude-3-haiku": 0.80,
                "gpt-3.5-turbo": 0.78,
                "gemini-pro": 0.82,
                "llama-2-70b": 0.76,
                "mixtral-8x7b": 0.79
            },
            "planning_bench": {
                "gpt-4o": 0.93,
                "gpt-4-turbo": 0.90,
                "claude-3-opus": 0.91,
                "claude-3-sonnet": 0.86,
                "claude-3-haiku": 0.79,
                "gpt-3.5-turbo": 0.75,
                "gemini-pro": 0.80,
                "llama-2-70b": 0.74,
                "mixtral-8x7b": 0.76
            }
        }
