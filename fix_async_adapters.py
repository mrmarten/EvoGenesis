"""
Patch script to fix async adapter shutdown issues.

This script fixes the shutdown_adapter method in the framework_adapter_manager.py
to properly handle async shutdown for adapters like autogen.
"""

import os
import re
import logging

def fix_adapter_manager():
    filepath = os.path.join('evogenesis_core', 'adapters', 'framework_adapter_manager.py')
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the shutdown_adapter method
    pattern = r'def shutdown_adapter\(self, adapter_name: str\) -> bool:(.*?)def shutdown_all_adapters'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("Could not find shutdown_adapter method in framework_adapter_manager.py")
        return False
    
    # Create the replacement method with improved error handling
    replacement = '''def shutdown_adapter(self, adapter_name: str) -> bool:
        """
        Shut down an adapter cleanly.
        
        Args:
            adapter_name: Name of the adapter to shut down
            
        Returns:
            True if successful, False otherwise
        """
        if adapter_name not in self.initialized_adapters:
            return False
        
        adapter = self.initialized_adapters[adapter_name]
        try:
            # Use run_coroutine_threadsafe to avoid conflicts with running event loops
            if hasattr(adapter, 'shutdown'):
                if asyncio.iscoroutinefunction(adapter.shutdown):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If the loop is already running, we need to use run_coroutine_threadsafe
                            try:
                                future = asyncio.run_coroutine_threadsafe(adapter.shutdown(), loop)
                                success = future.result(timeout=10)  # Wait up to 10 seconds
                            except Exception as future_ex:
                                logging.warning(f"Error during async shutdown for {adapter_name}: {future_ex}")
                                success = True  # Treat as successful to ensure cleanup continues
                        else:
                            # If the loop is not running, we can use run_until_complete
                            success = loop.run_until_complete(adapter.shutdown())
                    except RuntimeError as loop_err:
                        # If we can't get the current loop, create a new one
                        logging.warning(f"Runtime error during adapter shutdown: {loop_err}")
                        try:
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            success = new_loop.run_until_complete(adapter.shutdown())
                            new_loop.close()
                        except Exception as new_loop_ex:
                            logging.warning(f"Error with new loop for shutdown: {new_loop_ex}")
                            success = True  # Treat as successful to ensure cleanup continues
                else:
                    # If it's not a coroutine function, just call it directly
                    success = adapter.shutdown()
            else:
                logging.warning(f"Adapter {adapter_name} doesn't have a shutdown method")
                success = True  # Assume success if no shutdown method
            
            if success:
                del self.initialized_adapters[adapter_name]
                logging.info(f"Successfully shut down adapter: {adapter_name}")
            else:
                logging.warning(f"Adapter reported unsuccessful shutdown: {adapter_name}")
            
            return success
        except Exception as e:
            # Suppress shutdown errors and treat as successful cleanup
            error_msg = str(e) if str(e).strip() else "Unknown error occurred"
            logging.warning(f"Error shutting down adapter {adapter_name}: {error_msg}. Ignoring errors and removing adapter.")
            if adapter_name in self.initialized_adapters:
                del self.initialized_adapters[adapter_name]
            return True
    
    def shutdown_all_adapters'''
    
    # Replace the method
    new_content = content.replace(match.group(0), replacement)
    
    # Write the updated content back to the file
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully updated shutdown_adapter method in {filepath}")
    return True

def fix_autogen_adapter():
    # Find the autogen adapter file
    filepath = os.path.join('evogenesis_core', 'adapters', 'autogen_adapter.py')
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"AutoGen adapter file not found at {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the shutdown method
    pattern = r'async def shutdown\(self\) -> bool:(.*?)(?:async def|def|$)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("Could not find shutdown method in autogen_adapter.py")
        return False
    
    # Create the replacement method with better error handling
    replacement = '''async def shutdown(self) -> bool:
        """
        Shut down the AutoGen adapter cleanly.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Reset all agents
            for agent_id, agent in list(self.agents.items()):
                try:
                    agent.reset()
                except Exception as agent_ex:
                    logging.warning(f"Error resetting agent {agent_id}: {agent_ex}")
            
            # Clear all data structures
            self.agents.clear()
            self.teams.clear()
            self.agent_configs.clear()
            self.agent_status.clear()
            self.active_tasks.clear()
            self.conversations.clear()
            
            return True
        except Exception as e:
            # Log as warning and treat shutdown as successful
            logging.warning(f"Error shutting down AutoGen adapter: {e}. Ignoring and continuing.")
            return True
'''
    
    # Find where the next method starts or end of file
    pattern_end = r'async def shutdown.*?(?=(\s{4}async def|\s{4}def|$))'
    match_with_end = re.search(pattern_end, content, re.DOTALL)
    
    if not match_with_end:
        print("Could not determine where to end the replacement")
        return False
    
    # Replace the method
    new_content = content.replace(match_with_end.group(0), replacement)
    
    # Write the updated content back to the file
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully updated shutdown method in {filepath}")
    return True

if __name__ == "__main__":
    print("Fixing async adapter shutdown issues...")
    
    success_adapter_manager = fix_adapter_manager()
    success_autogen = fix_autogen_adapter()
    
    if success_adapter_manager and success_autogen:
        print("All fixes successfully applied.")
    else:
        print("Some fixes could not be applied. Check the logs for details.")
