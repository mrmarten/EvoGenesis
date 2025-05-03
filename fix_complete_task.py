"""
This script fixes indentation issues in the task_planner.py file
by recreating the complete_task method with proper indentation.
"""

import re

# The fixed method with proper indentation
FIXED_METHOD = '''    def complete_task(self, task_id: str, result: Any = None) -> bool:
        """
        Mark a task as completed with an optional result.
        
        Args:
            task_id: The ID of the task to complete
            result: The result data from the task execution
            
        Returns:
            True if successful, False otherwise
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status != TaskStatus.IN_PROGRESS:
            return False
            
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.result = result
        
        # Execute callbacks for this task
        for callback in self.task_callbacks.get(task_id, []):
            callback(task)
        
        # Update mission if this task is part of a scheduled mission
        if hasattr(self.kernel, 'mission_scheduler') and task.metadata and task.metadata.get('is_scheduled'):
            try:
                mission_id = task.metadata.get('mission_id')
                if mission_id:
                    self.kernel.mission_scheduler.update_mission_result(
                        task_id=task_id,
                        success=True,
                        result=result if isinstance(result, dict) else {'data': result}
                    )
            except Exception as e:
                self.logger.error(f"Error updating mission for completed task {task_id}: {str(e)}")
        
        # Check if parent task or goal is now complete
        self._check_parent_completion(task)
        
        # Start dependent tasks that are now ready
        self._start_dependent_tasks(task_id)
        
        return True'''

# Read the file
with open('evogenesis_core/modules/task_planner.py', 'r') as file:
    content = file.read()

# Replace the entire method with our fixed version
pattern = r'(\s+)def complete_task\(self.*?return True'
replacement = FIXED_METHOD
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write the fixed file
with open('evogenesis_core/modules/task_planner.py', 'w') as file:
    file.write(new_content)

print("Replaced complete_task method with fixed indentation.")
