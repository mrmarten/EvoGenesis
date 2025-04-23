"""
Framework Introspection Module - Provides detailed inspection of agent frameworks.

This module enhances the adapter factory by providing deeper introspection capabilities
for analyzing agent frameworks, including class hierarchies, method signatures, and dependencies.
"""

import inspect
import importlib
import logging
import ast
import re
import pkgutil
from typing import Dict, Any, List, Optional, Union, Callable, Type, Set
import networkx as nx
from packaging import version


class FrameworkIntrospector:
    """
    Advanced introspection tool for analyzing agent frameworks.
    
    Provides detailed information about a framework's structure, classes,
    methods, dependencies, and capabilities to enable more accurate adapter generation.
    """
    
    def __init__(self, framework_name: str, module_name: str = None):
        """
        Initialize a framework introspector.
        
        Args:
            framework_name: Name of the framework to analyze
            module_name: Optional specific module name to import
        """
        self.framework_name = framework_name
        self.module_name = module_name or framework_name
        self.module = None
        self.api_info = {}
        self.class_hierarchy = None
        self.dependency_graph = None
    
    def import_framework(self) -> bool:
        """
        Import the framework module.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.module = importlib.import_module(self.module_name)
            return True
        except ImportError:
            logging.warning(f"Could not import framework: {self.framework_name}")
            return False
    
    def analyze_in_depth(self) -> Dict[str, Any]:
        """
        Perform a deep analysis of the framework.
        
        Returns:
            Comprehensive dictionary with API information
        """
        if not self.module and not self.import_framework():
            return {}
        
        # Basic analysis
        self.api_info = self._analyze_basic()
        
        # Build class hierarchy
        self.class_hierarchy = self._build_class_hierarchy()
        self.api_info["class_hierarchy"] = self.class_hierarchy
        
        # Analyze dependencies
        self.dependency_graph = self._analyze_dependencies()
        self.api_info["dependencies"] = self._serialize_dependency_graph()
        
        # Analyze capabilities
        self.api_info["capabilities"] = self._analyze_capabilities()
        
        # Identify key abstractions
        self.api_info["key_abstractions"] = self._identify_key_abstractions()
        
        return self.api_info
    
    def get_adapter_recommendations(self) -> Dict[str, Any]:
        """
        Generate recommendations for adapter implementation.
        
        Returns:
            Dictionary with recommendations
        """
        if not self.api_info:
            self.analyze_in_depth()
        
        recommendations = {
            "agent_creation": self._recommend_agent_creation(),
            "task_execution": self._recommend_task_execution(),
            "team_coordination": self._recommend_team_coordination(),
            "error_handling": self._recommend_error_handling(),
            "capabilities_declaration": self._recommend_capabilities()
        }
        
        return recommendations
    
    def _analyze_basic(self) -> Dict[str, Any]:
        """
        Perform basic analysis of the framework.
        
        Returns:
            Dictionary with basic API information
        """
        # Extract framework version
        fw_version = getattr(self.module, "__version__", "unknown")
        
        # Extract key classes and their methods
        classes = {}
        for name, obj in inspect.getmembers(self.module):
            if inspect.isclass(obj) and self._is_framework_class(obj):
                methods = {}
                for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                    if not method_name.startswith("_") or method_name in ("__init__", "__call__"):
                        signature = str(inspect.signature(method))
                        docstring = inspect.getdoc(method) or ""
                        methods[method_name] = {
                            "signature": signature,
                            "docstring": docstring,
                            "is_async": inspect.iscoroutinefunction(method)
                        }
                
                classes[name] = {
                    "methods": methods,
                    "docstring": inspect.getdoc(obj) or "",
                    "base_classes": [base.__name__ for base in obj.__bases__ if base is not object]
                }
        
        # Extract key functions
        functions = {}
        for name, obj in inspect.getmembers(self.module):
            if inspect.isfunction(obj) and self._is_framework_function(obj):
                signature = str(inspect.signature(obj))
                docstring = inspect.getdoc(obj) or ""
                functions[name] = {
                    "signature": signature,
                    "docstring": docstring,
                    "is_async": inspect.iscoroutinefunction(obj)
                }
        
        # Store basic API info
        return {
            "name": self.framework_name,
            "version": fw_version,
            "classes": classes,
            "functions": functions
        }
    
    def _is_framework_class(self, cls) -> bool:
        """Determine if a class belongs to the framework."""
        module = cls.__module__
        return module.startswith(self.module_name) or module == self.module.__name__
    
    def _is_framework_function(self, func) -> bool:
        """Determine if a function belongs to the framework."""
        module = func.__module__
        return module.startswith(self.module_name) or module == self.module.__name__
    
    def _build_class_hierarchy(self) -> Dict[str, Any]:
        """
        Build the class hierarchy for the framework.
        
        Returns:
            Dictionary representation of the class hierarchy
        """
        hierarchy = {}
        
        # Get all classes
        all_classes = {}
        for name, obj in inspect.getmembers(self.module):
            if inspect.isclass(obj) and self._is_framework_class(obj):
                all_classes[name] = obj
        
        # For each class, find its immediate children
        for name, cls in all_classes.items():
            hierarchy[name] = {
                "parents": [base.__name__ for base in cls.__bases__ 
                           if base.__name__ != "object" and base.__name__ in all_classes],
                "children": []
            }
        
        # Populate children based on parent information
        for name, info in hierarchy.items():
            for parent in info["parents"]:
                if parent in hierarchy:
                    hierarchy[parent]["children"].append(name)
        
        return hierarchy
    
    def _analyze_dependencies(self) -> nx.DiGraph:
        """
        Analyze dependencies between classes in the framework.
        
        Returns:
            Directed graph of dependencies
        """
        G = nx.DiGraph()
        
        # Get all classes
        all_classes = {}
        for name, obj in inspect.getmembers(self.module):
            if inspect.isclass(obj) and self._is_framework_class(obj):
                all_classes[name] = obj
                G.add_node(name)
        
        # Analyze class dependencies through method signatures
        for class_name, cls in all_classes.items():
            for method_name, method in inspect.getmembers(cls, inspect.isfunction):
                if method_name.startswith('_') and method_name != '__init__':
                    continue
                
                # Check method signature for references to other classes
                sig = inspect.signature(method)
                for param_name, param in sig.parameters.items():
                    if param.annotation != inspect.Parameter.empty:
                        annotation = str(param.annotation)
                        for other_class in all_classes:
                            if other_class in annotation and other_class != class_name:
                                G.add_edge(class_name, other_class)
                
                # Check method body for references to other classes
                try:
                    source = inspect.getsource(method)
                    for other_class in all_classes:
                        if other_class != class_name and re.search(r'\b' + other_class + r'\b', source):
                            G.add_edge(class_name, other_class)
                except (IOError, TypeError):
                    pass
        
        return G
    
    def _serialize_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Serialize the dependency graph for JSON representation.
        
        Returns:
            Dictionary mapping classes to their dependencies
        """
        if not self.dependency_graph:
            return {}
        
        result = {}
        for node in self.dependency_graph.nodes():
            result[node] = list(self.dependency_graph.successors(node))
        
        return result
    
    def _analyze_capabilities(self) -> Dict[str, Any]:
        """
        Analyze the capabilities of the framework.
        
        Returns:
            Dictionary of capabilities and their availability
        """
        capabilities = {
            "multi_agent": False,
            "agent_creation": False,
            "tool_use": False,
            "planning": False,
            "memory": False,
            "async_support": False,
            "vector_db": False,
            "code_execution": False,
            "human_feedback": False,
            "team_coordination": False
        }
        
        # Check for multi-agent support
        multi_agent_keywords = ["MultiAgent", "AgentGroup", "Team", "Conversation", "GroupChat"]
        for keyword in multi_agent_keywords:
            if any(keyword in cls_name for cls_name in self.api_info.get("classes", {})):
                capabilities["multi_agent"] = True
                break
        
        # Check for agent creation
        agent_creation_keywords = ["Agent", "Assistant", "createAgent", "initialize_agent", "AgentExecutor"]
        for keyword in agent_creation_keywords:
            if any(keyword in item for item in list(self.api_info.get("classes", {})) + list(self.api_info.get("functions", {}))):
                capabilities["agent_creation"] = True
                break
        
        # Check for tool use
        tool_keywords = ["Tool", "ToolKit", "tools", "use_tools"]
        for keyword in tool_keywords:
            if any(keyword in item for item in list(self.api_info.get("classes", {})) + list(self.api_info.get("functions", {}))):
                capabilities["tool_use"] = True
                break
        
        # Check for planning
        planning_keywords = ["Plan", "Planner", "Planning", "task_decomposition"]
        for keyword in planning_keywords:
            if any(keyword in item for item in list(self.api_info.get("classes", {})) + list(self.api_info.get("functions", {}))):
                capabilities["planning"] = True
                break
        
        # Check for memory
        memory_keywords = ["Memory", "Conversation", "History", "Buffer"]
        for keyword in memory_keywords:
            if any(keyword in item for item in list(self.api_info.get("classes", {}))):
                capabilities["memory"] = True
                break
        
        # Check for async support
        capabilities["async_support"] = any(
            method.get("is_async", False) 
            for cls in self.api_info.get("classes", {}).values() 
            for method in cls.get("methods", {}).values()
        ) or any(
            func.get("is_async", False) 
            for func in self.api_info.get("functions", {}).values()
        )
        
        # Check for vector DB support
        vector_keywords = ["Vector", "Embedding", "Retrieval", "Search"]
        for keyword in vector_keywords:
            if any(keyword in item for item in list(self.api_info.get("classes", {}))):
                capabilities["vector_db"] = True
                break
        
        # Check for code execution
        code_keywords = ["Code", "Execution", "Execute", "Executor", "Docker", "Sandbox"]
        for keyword in code_keywords:
            if any(keyword in item for item in list(self.api_info.get("classes", {})) + list(self.api_info.get("functions", {}))):
                capabilities["code_execution"] = True
                break
        
        # Check for human feedback
        human_keywords = ["Human", "Feedback", "Interactive", "HITL", "HumanInputMode"]
        for keyword in human_keywords:
            if any(keyword in item for item in list(self.api_info.get("classes", {})) + list(self.api_info.get("functions", {}))):
                capabilities["human_feedback"] = True
                break
        
        # Check for team coordination
        team_keywords = ["Team", "Group", "Coordinate", "Collaboration", "GroupChat"]
        for keyword in team_keywords:
            if any(keyword in item for item in list(self.api_info.get("classes", {})) + list(self.api_info.get("functions", {}))):
                capabilities["team_coordination"] = True
                break
        
        return capabilities
    
    def _identify_key_abstractions(self) -> Dict[str, List[str]]:
        """
        Identify key abstractions in the framework relevant for adaptation.
        
        Returns:
            Dictionary mapping abstraction types to class names
        """
        abstractions = {
            "agents": [],
            "tools": [],
            "memory": [],
            "chains": [],
            "models": [],
            "planners": []
        }
        
        # Identify agent classes
        agent_keywords = ["Agent", "Assistant", "UserProxy", "Executor"]
        for cls_name in self.api_info.get("classes", {}):
            if any(keyword in cls_name for keyword in agent_keywords):
                abstractions["agents"].append(cls_name)
        
        # Identify tool classes
        tool_keywords = ["Tool", "ToolKit", "Action"]
        for cls_name in self.api_info.get("classes", {}):
            if any(keyword in cls_name for keyword in tool_keywords):
                abstractions["tools"].append(cls_name)
        
        # Identify memory classes
        memory_keywords = ["Memory", "History", "Buffer", "Storage"]
        for cls_name in self.api_info.get("classes", {}):
            if any(keyword in cls_name for keyword in memory_keywords):
                abstractions["memory"].append(cls_name)
        
        # Identify chain classes
        chain_keywords = ["Chain", "Pipeline", "Sequence"]
        for cls_name in self.api_info.get("classes", {}):
            if any(keyword in cls_name for keyword in chain_keywords):
                abstractions["chains"].append(cls_name)
        
        # Identify model classes
        model_keywords = ["Model", "LLM", "LLMChain", "Completion", "GPT", "Claude"]
        for cls_name in self.api_info.get("classes", {}):
            if any(keyword in cls_name for keyword in model_keywords):
                abstractions["models"].append(cls_name)
        
        # Identify planner classes
        planner_keywords = ["Plan", "Planner", "TaskDecomposition", "Strategy"]
        for cls_name in self.api_info.get("classes", {}):
            if any(keyword in cls_name for keyword in planner_keywords):
                abstractions["planners"].append(cls_name)
        
        return abstractions
    
    def _recommend_agent_creation(self) -> Dict[str, Any]:
        """Generate recommendations for implementing agent creation."""
        if not self.api_info:
            return {}
        
        recommendations = {
            "primary_classes": [],
            "factory_functions": [],
            "configuration_approach": "",
            "example_code": "",
        }
        
        # Identify primary agent classes
        agent_classes = self.api_info.get("key_abstractions", {}).get("agents", [])
        if agent_classes:
            recommendations["primary_classes"] = agent_classes[:3]  # Top 3 most relevant
        
        # Identify factory functions
        for func_name, func_info in self.api_info.get("functions", {}).items():
            if any(cls in func_info.get("signature", "") for cls in agent_classes):
                recommendations["factory_functions"].append(func_name)
        
        # Determine configuration approach
        has_config_classes = any("Config" in cls for cls in self.api_info.get("classes", {}))
        if has_config_classes:
            recommendations["configuration_approach"] = "Use configuration objects for agent setup"
        else:
            recommendations["configuration_approach"] = "Use direct parameter passing for agent setup"
        
        # Generate example code template (simplified)
        if agent_classes and recommendations["factory_functions"]:
            recommendations["example_code"] = f"""
# Example agent creation using {self.framework_name}
{recommendations["factory_functions"][0]}(
    name="agent_name",
    # Add configuration parameters here
)
"""
        elif agent_classes:
            recommendations["example_code"] = f"""
# Example agent creation using {self.framework_name}
{agent_classes[0]}(
    name="agent_name",
    # Add configuration parameters here
)
"""
        
        return recommendations
    
    def _recommend_task_execution(self) -> Dict[str, Any]:
        """Generate recommendations for implementing task execution."""
        async_support = self.api_info.get("capabilities", {}).get("async_support", False)
        
        return {
            "use_async": async_support,
            "execution_approach": "Direct method call" if not async_support else "Async method call",
            "recommended_pattern": "to_thread wrapper" if async_support else "synchronous execution"
        }
    
    def _recommend_team_coordination(self) -> Dict[str, Any]:
        """Generate recommendations for implementing team coordination."""
        team_support = self.api_info.get("capabilities", {}).get("team_coordination", False)
        multi_agent = self.api_info.get("capabilities", {}).get("multi_agent", False)
        
        if team_support:
            return {
                "native_support": True,
                "implementation_approach": "Use native team classes/methods"
            }
        elif multi_agent:
            return {
                "native_support": False,
                "implementation_approach": "Implement custom routing between agents"
            }
        else:
            return {
                "native_support": False,
                "implementation_approach": "Implement full team coordination from scratch"
            }
    
    def _recommend_error_handling(self) -> Dict[str, Any]:
        """Generate recommendations for implementing error handling."""
        # Check if framework has specific error classes
        error_classes = [cls for cls in self.api_info.get("classes", {}) 
                      if "Error" in cls or "Exception" in cls]
        
        return {
            "framework_errors": error_classes,
            "recommended_approach": "Use specific error handling" if error_classes else "Use generic try/except blocks",
            "retry_mechanism": "Implement custom retry logic"
        }
    
    def _recommend_capabilities(self) -> Dict[str, Any]:
        """Generate recommendations for capabilities declaration."""
        capabilities = self.api_info.get("capabilities", {})
        key_abstractions = self.api_info.get("key_abstractions", {})
        
        return {
            "declared_capabilities": {k: v for k, v in capabilities.items() if v},
            "key_abstractions": {k: v[:3] for k, v in key_abstractions.items() if v}  # Top 3 of each
        }
