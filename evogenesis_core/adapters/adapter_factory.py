"""
Adapter Factory Module - Dynamically generates and updates framework adapters.

This module provides a factory system for generating framework adapters on demand,
allowing EvoGenesis to adapt to new agent frameworks and updates to existing ones.
"""

import os
import importlib
import inspect
import logging
import ast
import re
import pkgutil
from typing import Dict, Any, List, Optional, Union, Callable, Type
import jinja2
import difflib
import subprocess
import json
import asyncio

from evogenesis_core.adapters.base_adapter import AgentExecutionAdapter


class AdapterTemplate:
    """Represents a template for generating framework adapters."""
    
    def __init__(self, name: str, template_path: str, metadata: Dict[str, Any]):
        """
        Initialize an adapter template.
        
        Args:
            name: Name of the template
            template_path: Path to the template file
            metadata: Template metadata including required framework info
        """
        self.name = name
        self.template_path = template_path
        self.metadata = metadata
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.dirname(template_path)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.template = self.jinja_env.get_template(os.path.basename(template_path))
    
    def render(self, context: Dict[str, Any]) -> str:
        """
        Render the template with the provided context.
        
        Args:
            context: Context variables for the template
            
        Returns:
            Rendered template as a string
        """
        return self.template.render(**context)


class FrameworkAnalyzer:
    """Analyzes agent frameworks to extract API information."""
    
    def __init__(self, framework_name: str, module_name: str = None):
        """
        Initialize a framework analyzer.
        
        Args:
            framework_name: Name of the framework to analyze
            module_name: Optional specific module name to import
        """
        self.framework_name = framework_name
        self.module_name = module_name or framework_name
        self.module = None
        self.api_info = {}
    
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
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the framework and extract API information.
        
        Returns:
            Dictionary with API information
        """
        if not self.module and not self.import_framework():
            return {}
        
        # Extract framework version
        version = getattr(self.module, "__version__", "unknown")
        
        # Extract key classes and their methods
        classes = {}
        for name, obj in inspect.getmembers(self.module):
            if inspect.isclass(obj) and obj.__module__.startswith(self.module_name):
                methods = {}
                for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                    if not method_name.startswith("_"):
                        signature = str(inspect.signature(method))
                        docstring = inspect.getdoc(method) or ""
                        methods[method_name] = {
                            "signature": signature,
                            "docstring": docstring
                        }
                
                classes[name] = {
                    "methods": methods,
                    "docstring": inspect.getdoc(obj) or ""
                }
        
        # Extract key functions
        functions = {}
        for name, obj in inspect.getmembers(self.module):
            if inspect.isfunction(obj) and obj.__module__.startswith(self.module_name):
                signature = str(inspect.signature(obj))
                docstring = inspect.getdoc(obj) or ""
                functions[name] = {
                    "signature": signature,
                    "docstring": docstring
                }
        
        # Store and return the API info
        self.api_info = {
            "name": self.framework_name,
            "version": version,
            "classes": classes,
            "functions": functions
        }
        
        return self.api_info
    
    def compare_with_previous(self, previous_api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current API with a previous version.
        
        Args:
            previous_api_info: Previous API information
            
        Returns:
            Dictionary with differences
        """
        if not self.api_info:
            self.analyze()
        
        differences = {
            "added_classes": [],
            "removed_classes": [],
            "modified_classes": [],
            "added_functions": [],
            "removed_functions": [],
            "modified_functions": []
        }
        
        # Compare classes
        current_classes = set(self.api_info.get("classes", {}).keys())
        previous_classes = set(previous_api_info.get("classes", {}).keys())
        
        differences["added_classes"] = list(current_classes - previous_classes)
        differences["removed_classes"] = list(previous_classes - current_classes)
        
        # Check for method changes in common classes
        for class_name in current_classes.intersection(previous_classes):
            current_methods = set(self.api_info["classes"][class_name]["methods"].keys())
            previous_methods = set(previous_api_info["classes"][class_name]["methods"].keys())
            
            added_methods = current_methods - previous_methods
            removed_methods = previous_methods - current_methods
            
            # Check for signature changes in common methods
            modified_methods = []
            for method_name in current_methods.intersection(previous_methods):
                current_sig = self.api_info["classes"][class_name]["methods"][method_name]["signature"]
                previous_sig = previous_api_info["classes"][class_name]["methods"][method_name]["signature"]
                
                if current_sig != previous_sig:
                    modified_methods.append(method_name)
            
            if added_methods or removed_methods or modified_methods:
                differences["modified_classes"].append({
                    "class": class_name,
                    "added_methods": list(added_methods),
                    "removed_methods": list(removed_methods),
                    "modified_methods": modified_methods
                })
        
        # Compare functions
        current_functions = set(self.api_info.get("functions", {}).keys())
        previous_functions = set(previous_api_info.get("functions", {}).keys())
        
        differences["added_functions"] = list(current_functions - previous_functions)
        differences["removed_functions"] = list(previous_functions - current_functions)
        
        # Check for signature changes in common functions
        for func_name in current_functions.intersection(previous_functions):
            current_sig = self.api_info["functions"][func_name]["signature"]
            previous_sig = previous_api_info["functions"][func_name]["signature"]
            
            if current_sig != previous_sig:
                differences["modified_functions"].append(func_name)
        
        return differences


class AdapterFactory:
    """Factory for generating and updating framework adapters."""
    
    def __init__(self, templates_dir: str = None, adapters_dir: str = None):
        """
        Initialize the adapter factory.
        
        Args:
            templates_dir: Directory containing adapter templates
            adapters_dir: Directory where generated adapters will be stored
        """
        self.templates_dir = templates_dir or os.path.join(os.path.dirname(__file__), "templates")
        self.adapters_dir = adapters_dir or os.path.dirname(__file__)
        self.templates = {}
        self.framework_cache = {}
        
        # Ensure templates directory exists
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Load templates
        self.load_templates()
    
    def load_templates(self):
        """Load adapter templates from the templates directory."""
        if not os.path.exists(self.templates_dir):
            logging.warning(f"Templates directory not found: {self.templates_dir}")
            return
        
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".j2") and not filename.startswith("_"):
                template_name = os.path.splitext(filename)[0]
                metadata_path = os.path.join(self.templates_dir, f"{template_name}.json")
                
                # Load metadata if it exists
                metadata = {}
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except json.JSONDecodeError:
                        logging.warning(f"Error parsing metadata for template: {template_name}")
                
                # Create template object
                template_path = os.path.join(self.templates_dir, filename)
                self.templates[template_name] = AdapterTemplate(
                    name=template_name,
                    template_path=template_path,
                    metadata=metadata
                )
                
                logging.info(f"Loaded adapter template: {template_name}")
    
    def get_available_templates(self) -> List[str]:
        """
        Get names of available adapter templates.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def analyze_framework(self, framework_name: str, module_name: str = None) -> Dict[str, Any]:
        """
        Analyze a framework and get its API information.
        
        Args:
            framework_name: Name of the framework
            module_name: Optional specific module name
            
        Returns:
            Dictionary with API information
        """
        analyzer = FrameworkAnalyzer(framework_name, module_name)
        api_info = analyzer.analyze()
        
        # Cache the results
        self.framework_cache[framework_name] = api_info
        
        return api_info
    
    def generate_adapter(self, template_name: str, framework_name: str, 
                        module_name: str = None, output_name: str = None) -> str:
        """
        Generate an adapter for a framework.
        
        Args:
            template_name: Name of the template to use
            framework_name: Name of the framework
            module_name: Optional specific module name
            output_name: Optional output filename
            
        Returns:
            Path to the generated adapter file
        """
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")
        
        # Analyze the framework
        if framework_name not in self.framework_cache:
            self.analyze_framework(framework_name, module_name)
        
        framework_info = self.framework_cache[framework_name]
        
        # Determine output name
        if not output_name:
            output_name = f"{framework_name.lower()}_adapter.py"
        
        # Prepare context for template
        context = {
            "framework": framework_info,
            "generation_timestamp": asyncio.get_event_loop().time(),
            "module_name": module_name or framework_name
        }
        
        # Render the template
        template = self.templates[template_name]
        rendered = template.render(context)
        
        # Write to file
        output_path = os.path.join(self.adapters_dir, output_name)
        with open(output_path, 'w') as f:
            f.write(rendered)
        
        logging.info(f"Generated adapter for {framework_name}: {output_path}")
        return output_path
    
    def update_adapter(self, adapter_path: str, framework_name: str, 
                     module_name: str = None) -> Dict[str, Any]:
        """
        Update an existing adapter after framework changes.
        
        Args:
            adapter_path: Path to the adapter file
            framework_name: Name of the framework
            module_name: Optional specific module name
            
        Returns:
            Dictionary with update information
        """
        # Get current framework info
        current_info = self.analyze_framework(framework_name, module_name)
        
        # Try to extract previous framework info from adapter
        previous_info = self._extract_framework_info_from_adapter(adapter_path)
        
        # Compare and find differences
        analyzer = FrameworkAnalyzer(framework_name, module_name)
        analyzer.api_info = current_info
        differences = analyzer.compare_with_previous(previous_info)
        
        # If there are significant differences, regenerate the adapter
        if self._has_significant_differences(differences):
            # Extract original template if possible
            template_name = self._detect_template_from_adapter(adapter_path)
            if not template_name:
                template_name = next(iter(self.templates))
                logging.warning(f"Could not detect template, using {template_name}")
            
            # Generate updated adapter
            updated_path = self.generate_adapter(
                template_name=template_name,
                framework_name=framework_name,
                module_name=module_name,
                output_name=os.path.basename(adapter_path) + ".updated"
            )
            
            # Generate diff report
            diff_report = self._generate_diff(adapter_path, updated_path)
            
            return {
                "status": "updated",
                "differences": differences,
                "updated_path": updated_path,
                "diff_report": diff_report
            }
        else:
            return {
                "status": "no_update_needed",
                "differences": differences
            }
    
    def adapt_to_framework_changes(self, adapter_path: str, framework_name: str,
                                module_name: str = None, auto_fix: bool = False) -> Dict[str, Any]:
        """
        Adapt an existing adapter to framework changes.
        
        Args:
            adapter_path: Path to the adapter file
            framework_name: Name of the framework
            module_name: Optional specific module name
            auto_fix: Whether to automatically apply fixes
            
        Returns:
            Dictionary with adaptation information
        """
        update_info = self.update_adapter(adapter_path, framework_name, module_name)
        
        if update_info["status"] == "updated" and auto_fix:
            # Apply the update
            updated_path = update_info["updated_path"]
            import shutil
            shutil.move(updated_path, adapter_path)
            update_info["status"] = "auto_fixed"
        
        return update_info
    
    def create_template_from_adapter(self, adapter_path: str, template_name: str = None) -> str:
        """
        Create a new template from an existing adapter.
        
        Args:
            adapter_path: Path to the adapter file
            template_name: Optional name for the new template
            
        Returns:
            Path to the created template
        """
        # Read adapter file
        with open(adapter_path, 'r') as f:
            adapter_code = f.read()
        
        # Determine template name if not provided
        if not template_name:
            base_name = os.path.basename(adapter_path)
            template_name = os.path.splitext(base_name)[0].replace("_adapter", "")
        
        # Extract framework-specific parts and templatize them
        templatized_code = self._templatize_adapter_code(adapter_code)
        
        # Save as template
        template_path = os.path.join(self.templates_dir, f"{template_name}.j2")
        with open(template_path, 'w') as f:
            f.write(templatized_code)
        
        # Create metadata
        framework_info = self._extract_framework_info_from_adapter(adapter_path)
        metadata = {
            "name": template_name,
            "description": f"Template for {template_name} adapters",
            "framework_requirements": {
                "name": framework_info.get("name", "unknown"),
                "min_version": framework_info.get("version", "0.0.0")
            }
        }
        
        # Save metadata
        metadata_path = os.path.join(self.templates_dir, f"{template_name}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Reload templates to include the new one
        self.load_templates()
        
        return template_path
    
    def validate_adapter(self, adapter_path: str) -> Dict[str, Any]:
        """
        Validate an adapter by checking its code and running tests.
        
        Args:
            adapter_path: Path to the adapter file
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "syntax_valid": False,
            "implements_interface": False,
            "test_results": None,
            "errors": []
        }
        
        # Check syntax
        try:
            with open(adapter_path, 'r') as f:
                code = f.read()
            ast.parse(code)
            validation_results["syntax_valid"] = True
        except SyntaxError as e:
            validation_results["errors"].append(f"Syntax error: {str(e)}")
            return validation_results
        
        # Check implementation of interface
        try:
            # Get the module name from the path
            module_dir = os.path.dirname(adapter_path)
            module_name = os.path.splitext(os.path.basename(adapter_path))[0]
            
            # Temporarily add the directory to sys.path
            import sys
            sys.path.insert(0, os.path.dirname(module_dir))
            
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, adapter_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find adapter classes
            adapter_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, AgentExecutionAdapter) and 
                    obj != AgentExecutionAdapter):
                    adapter_classes.append(obj)
            
            # Check if all required methods are implemented
            if adapter_classes:
                for cls in adapter_classes:
                    missing_methods = []
                    for method_name, method in inspect.getmembers(AgentExecutionAdapter, inspect.isfunction):
                        if method_name.startswith('__'):
                            continue
                        if not hasattr(cls, method_name) or not callable(getattr(cls, method_name)):
                            missing_methods.append(method_name)
                    
                    if missing_methods:
                        validation_results["errors"].append(
                            f"Class {cls.__name__} is missing required methods: {', '.join(missing_methods)}"
                        )
                    else:
                        validation_results["implements_interface"] = True
            else:
                validation_results["errors"].append("No adapter classes found")
            
            # Remove the directory from sys.path
            sys.path.remove(os.path.dirname(module_dir))
            
        except Exception as e:
            validation_results["errors"].append(f"Error importing adapter: {str(e)}")
        
        # TODO: Run tests if available
        
        return validation_results
    
    def _extract_framework_info_from_adapter(self, adapter_path: str) -> Dict[str, Any]:
        """
        Extract framework information from an adapter file.
        
        Args:
            adapter_path: Path to the adapter file
            
        Returns:
            Dictionary with framework information
        """
        framework_info = {
            "name": "unknown",
            "version": "unknown",
            "classes": {},
            "functions": {}
        }
        
        try:
            with open(adapter_path, 'r') as f:
                code = f.read()
            
            # Try to find import statements
            import_pattern = r"import\s+([^\s]+)|from\s+([^\s]+)\s+import"
            matches = re.findall(import_pattern, code)
            
            potential_framework_names = []
            for match in matches:
                framework = match[0] or match[1]
                if framework and not framework.startswith(("evogenesis", "typing", "logging")):
                    potential_framework_names.append(framework.split('.')[0])
            
            if potential_framework_names:
                # Try each potential framework
                for name in potential_framework_names:
                    try:
                        analyzer = FrameworkAnalyzer(name)
                        if analyzer.import_framework():
                            framework_info = analyzer.analyze()
                            break
                    except:
                        continue
        except Exception as e:
            logging.warning(f"Error extracting framework info from adapter: {str(e)}")
        
        return framework_info
    
    def _detect_template_from_adapter(self, adapter_path: str) -> Optional[str]:
        """
        Try to detect which template was used to generate an adapter.
        
        Args:
            adapter_path: Path to the adapter file
            
        Returns:
            Template name if detected, None otherwise
        """
        try:
            with open(adapter_path, 'r') as f:
                code = f.read()
            
            # Look for template signatures in comments
            template_pattern = r"# Generated from template: (\w+)"
            match = re.search(template_pattern, code)
            if match:
                return match.group(1)
            
            # Try to match with existing templates
            best_match = None
            highest_score = 0
            
            for name, template in self.templates.items():
                # Render an empty template for comparison
                empty_render = template.render({
                    "framework": {"name": "test", "version": "0.0.0", "classes": {}, "functions": {}},
                    "generation_timestamp": 0,
                    "module_name": "test"
                })
                
                # Calculate similarity score
                similarity = difflib.SequenceMatcher(None, code, empty_render).ratio()
                if similarity > highest_score:
                    highest_score = similarity
                    best_match = name
            
            if highest_score > 0.5:  # Arbitrary threshold
                return best_match
                
        except Exception as e:
            logging.warning(f"Error detecting template: {str(e)}")
        
        return None
    
    def _templatize_adapter_code(self, code: str) -> str:
        """
        Convert adapter code to a template by replacing framework-specific parts.
        
        Args:
            code: Adapter code
            
        Returns:
            Templatized code
        """
        # Replace import statements
        import_pattern = r"(import\s+)([^\s\.]+)(.*?)$|from\s+([^\s\.]+)(.*?)$"
        
        def import_replacer(match):
            if match.group(2):  # import form
                framework = match.group(2)
                if not framework.startswith(("evogenesis", "typing", "logging", "json", "uuid", "os")):
                    return f"{match.group(1)}{{{{ module_name }}}}{match.group(3)}"
                return match.group(0)
            else:  # from form
                framework = match.group(4)
                if not framework.startswith(("evogenesis", "typing", "logging", "json", "uuid", "os")):
                    return f"from {{{{ module_name }}}}{match.group(5)}"
                return match.group(0)
        
        templatized = re.sub(import_pattern, import_replacer, code, flags=re.MULTILINE)
        
        # Add template variables for framework name, version
        templatized = templatized.replace(
            'FRAMEWORK_AVAILABLE = True',
            '{{ framework.name|upper }}_AVAILABLE = True'
        )
        
        # Add generation comment
        templatized = f"""# Generated from template: {{{{ template_name }}}}
# This file was automatically generated. Do not edit directly.
# Generation timestamp: {{{{ generation_timestamp }}}}

{templatized}"""
        
        return templatized
    
    def _has_significant_differences(self, differences: Dict[str, Any]) -> bool:
        """
        Determine if API differences are significant enough to warrant adapter update.
        
        Args:
            differences: Dictionary with API differences
            
        Returns:
            True if differences are significant, False otherwise
        """
        # Check for removed classes or functions which would break the adapter
        if differences.get("removed_classes") or differences.get("removed_functions"):
            return True
        
        # Check for method removals in modified classes
        for class_mod in differences.get("modified_classes", []):
            if class_mod.get("removed_methods"):
                return True
        
        return False
    
    def _generate_diff(self, original_path: str, updated_path: str) -> str:
        """
        Generate a diff between original and updated adapter files.
        
        Args:
            original_path: Path to the original file
            updated_path: Path to the updated file
            
        Returns:
            String with diff output
        """
        with open(original_path, 'r') as f:
            original_lines = f.readlines()
        
        with open(updated_path, 'r') as f:
            updated_lines = f.readlines()
        
        diff = difflib.unified_diff(
            original_lines,
            updated_lines,
            fromfile=original_path,
            tofile=updated_path
        )
        
        return ''.join(diff)
