"""
This script fixes indentation issues in the task_planner.py file.
"""

import re

def fix_task_planner():
    # Read the file
    with open('evogenesis_core/modules/task_planner.py', 'r') as file:
        content = file.read()
    
    # Find the complete_task method and fix its indentation
    pattern = r'(\s+)def complete_task\(self, task_id: str, result: Any = None\) -> bool:'
    match = re.search(pattern, content)
    
    if match:
        # Get the indentation of the method
        proper_indent = '    '  # Standard 4-space indentation
        
        # Find the complete method definition including its docstring and body
        method_pattern = r'(\s+)def complete_task\(self, task_id: str, result: Any = None\) -> bool:(.*?)(?=\n\s+def|\Z)'
        method_match = re.search(method_pattern, content, re.DOTALL)
        
        if method_match:
            old_method = method_match.group(0)
            
            # Extract the content of the method
            method_lines = old_method.split('\n')
            
            # Rebuild with proper indentation
            new_method_lines = []
            for i, line in enumerate(method_lines):
                if i == 0:  # Method definition line
                    new_method_lines.append(proper_indent + "def complete_task(self, task_id: str, result: Any = None) -> bool:")
                else:
                    # Strip all whitespace from the beginning and add proper indentation
                    stripped = line.lstrip()
                    if stripped:  # Not an empty line
                        new_method_lines.append(proper_indent + proper_indent + stripped)
                    else:  # Empty line
                        new_method_lines.append('')
            
            new_method = '\n'.join(new_method_lines)
            
            # Replace the old method with the new one
            new_content = content.replace(old_method, new_method)
            
            # Write the file back
            with open('evogenesis_core/modules/task_planner.py', 'w') as file:
                file.write(new_content)
            
            print("Fixed indentation in complete_task method")
            return True
    
    print("Could not find or fix the complete_task method")
    return False

if __name__ == "__main__":
    fix_task_planner()
