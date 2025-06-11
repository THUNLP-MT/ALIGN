import re

# Global dictionary to store the current location based on the env object id
# This assumes the env object will be consistently passed between calls
_agent_locations = {}
_agent_inventories = {}
_non_closable_receptacles = {"shelf", "counter", "countertop", "table", "diningtable", "sidetable", "coffeetable", "desk"}
_cooling_receptacles = {"fridge", "freezer", "refrigerator"}
_cleaning_receptacles = {"sinkbasin", "sink", "bathtub", "washingmachine"}

def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic for different task types.
    Enhanced with clearer and more comprehensive rules.
    """
    return """
    Important Environment Rules:
    1. You must first move to a receptacle before you can examine it. For example, you must use 'go to kitchen counter' before you can 'examine kitchen counter'.
    2. Objects can only be interacted with when you are at their location.
    3. Follow the logical sequence of actions - move to objects before interacting with them.
    4. When placing objects on receptacles, use the format 'move [object] to [receptacle]' (e.g., 'move bread 1 to countertop 1'). You can also use 'put' or 'place' with 'on' or 'in' (e.g., 'put bread 1 on countertop 1').
    5. You cannot examine or interact with objects that are not at your current location.
    6. You must take an object before you can place it somewhere else.
    7. To cool objects, you need to use appropriate cooling receptacles like fridges or freezers, and you must be at that receptacle's location.
    8. To clean objects, you need to use appropriate cleaning receptacles like sinks or bathtubs, and you must have the object in your inventory.
    9. After placing an object on a receptacle, you can examine it since it's in your current location.
    10. Not all receptacles can be closed. Fixed structures like shelves, countertops, and tables cannot be closed.
    11. Always check your current location before attempting to interact with objects or receptacles.
    12. If you attempt to take an object and receive "Nothing happens", it might mean the object is not available or is already taken. Try examining your surroundings or inventory first.
    13. Some objects might not be interactive or might require specific conditions to be interacted with.
    14. You must have an object in your inventory before you can clean, cool, heat, or slice it.
    15. If an object is visible in a receptacle but cannot be taken, it might be inaccessible or not interactable at this time.
    16. If you're looking for an object mentioned in the task but cannot find it, try examining different receptacles in the environment. The object might be in a location not immediately visible.
    17. Some tasks may require you to find objects that aren't initially visible. Explore the environment thoroughly by examining different receptacles.
    """

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action and return the next observation, reward, and done status.
    Enhanced with better feedback and handling of edge cases.
    """
    global _agent_locations, _agent_inventories, _non_closable_receptacles, _cooling_receptacles, _cleaning_receptacles
    env_id = id(env)
    
    # Initialize location tracking for this environment if not already done
    if env_id not in _agent_locations:
        _agent_locations[env_id] = None
        _agent_inventories[env_id] = set()
    
    # Parse the agent action to determine the action type and target
    agent_action = agent_action.strip().lower()
    action_parts = agent_action.split()
    action_type = action_parts[0] if action_parts else ""
    
    # Update location when "go to" action is performed
    if len(action_parts) >= 3 and action_parts[0] == "go" and action_parts[1] == "to":
        target_location = " ".join(action_parts[2:])
        
        # Process the goto action
        obs, reward, done, info = env.step([agent_action])
        result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
        
        # Check if the action was successful (basic heuristic - if error message isn't returned)
        if not ("You can't" in result_obs or "cannot" in result_obs.lower() or "can not" in result_obs.lower()):
            # If successful, update the agent's location
            _agent_locations[env_id] = target_location
            logger.debug(f"Agent moved to: {target_location}")
        
        return result_obs, result_reward, result_done
    
    # Handle close action - addressing Analysis Result 6
    elif action_type == "close" and len(action_parts) >= 2:
        receptacle_name = " ".join(action_parts[1:])
        current_location = _agent_locations[env_id]
        
        # Check if the receptacle is at the agent's current location
        if current_location != receptacle_name and re.search(r'\d+$', receptacle_name):
            logger.debug(f"Agent attempted to close '{receptacle_name}' while at '{current_location}'")
            return f"You cannot close {receptacle_name} without first moving to it. You need to use 'go to {receptacle_name}' first.", False, False
        
        # Check if this is a non-closable receptacle type
        base_receptacle_type = receptacle_name.rstrip("0123456789 ")
        if base_receptacle_type.strip() in _non_closable_receptacles:
            logger.debug(f"Agent attempted to close non-closable receptacle: {receptacle_name}")
            return f"{receptacle_name} cannot be closed because it is an open structure. Closing is not applicable for this type of receptacle.", False, False
        
        # Proceed with the close action for closable receptacles
        obs, reward, done, info = env.step([agent_action])
        result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
        
        # Provide clearer feedback if the action still fails
        if "Nothing happens" in result_obs:
            return f"You attempted to close {receptacle_name}, but nothing happens. This may indicate that the receptacle is already closed or cannot be closed.", result_reward, result_done
        
        return result_obs, result_reward, result_done
    
    # Handle inventory action to track what objects the agent has
    elif action_type == "inventory":
        obs, reward, done, info = env.step([agent_action])
        result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
        
        # Extract inventory items
        inventory_items = re.findall(r'([\w\s]+\d+)', result_obs.lower())
        if inventory_items:
            _agent_inventories[env_id] = set(item.strip() for item in inventory_items)
            logger.debug(f"Updated inventory: {_agent_inventories[env_id]}")
        else:
            _agent_inventories[env_id] = set()
            logger.debug("Inventory is empty")
        
        return result_obs, result_reward, result_done
    
    # Handle take action to update inventory tracking - Enhanced for Analysis Result 8 and 9
    elif action_type == "take" and "from" in action_parts:
        from_index = action_parts.index("from")
        object_name = " ".join(action_parts[1:from_index])
        receptacle_name = " ".join(action_parts[from_index+1:])
        current_location = _agent_locations[env_id]
        
        # Check if agent is at the receptacle's location
        if current_location != receptacle_name and re.search(r'\d+$', receptacle_name):
            logger.debug(f"Agent attempted to take from '{receptacle_name}' while at '{current_location}'")
            return f"You cannot take objects from {receptacle_name} without first moving to it. You need to use 'go to {receptacle_name}' first.", False, False
        
        # First do a look action to see what's visible
        look_obs, _, _, look_info = env.step(["look"])
        look_result = look_obs[0].lower()
        
        # Enhanced handling for Analysis Result 15 - check if object is mentioned in task
        is_task_object = object_name in task.lower() or object_name.rstrip("0123456789 ").strip() in task.lower()
        
        # Check if the object is mentioned in the look result, init_obs, or task
        object_exists = object_name in look_result or object_name in init_obs.lower() or is_task_object
        object_at_location = False
        
        # Check if the object is at the current location
        if current_location:
            # Do an examine on the receptacle to see what objects are there
            examine_action = f"examine {receptacle_name}"
            examine_obs, _, _, examine_info = env.step([examine_action])
            examine_result = examine_obs[0].lower()
            
            # Check if the object is mentioned in the examination
            object_at_location = object_name in examine_result
            
            # Reset the environment state with a look action (since examine might change state)
            env.step(["look"])
        
        # Check if object is already in inventory
        if object_name in _agent_inventories[env_id]:
            logger.debug(f"Agent attempted to take '{object_name}' but it's already in inventory")
            return f"You already have {object_name} in your inventory.", False, False
        
        # Now try the take action
        obs, reward, done, info = env.step([agent_action])
        result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
        
        # If action failed with generic message but object exists, provide better feedback
        if "Nothing happens" in result_obs:
            # Addressing Analysis Result 15 - handling task-relevant objects
            if is_task_object and not object_at_location:
                logger.debug(f"Object '{object_name}' mentioned in task but not found at '{receptacle_name}'")
                return f"The {object_name} mentioned in your task is not currently in {receptacle_name}. Try exploring other receptacles in the environment to find it.", False, False
            # Addressing Analysis Result 9 - handling objects that are visible but not interactable
            elif object_exists and object_at_location:
                logger.debug(f"Object '{object_name}' exists at '{receptacle_name}' but cannot be taken")
                return f"You can see {object_name} at {receptacle_name}, but it seems to be inaccessible or not interactable at this time. It might be secured or require another action before you can take it.", False, False
            elif object_exists and not object_at_location:
                logger.debug(f"Object '{object_name}' exists but is not at '{receptacle_name}'")
                return f"You cannot take {object_name} from {receptacle_name} because it's not there. It might be located elsewhere.", False, False
            elif not object_exists:
                logger.debug(f"Object '{object_name}' doesn't appear to exist")
                return f"You cannot take {object_name} because it doesn't appear to exist in this environment. Check that you have the correct object name.", False, False
        
        # If successful take action, update inventory
        if not ("You can't" in result_obs or "cannot" in result_obs.lower() or "can not" in result_obs.lower()):
            _agent_inventories[env_id].add(object_name)
            logger.debug(f"Added {object_name} to inventory: {_agent_inventories[env_id]}")
            
            # Provide clear success feedback if not already present
            if "You " not in result_obs:
                return f"You successfully take {object_name} from {receptacle_name}.", result_reward, result_done
        
        return result_obs, result_reward, result_done
    
    # Handle cleaning actions - addressing Analysis Result 10
    elif action_type == "clean" and "with" in action_parts:
        with_index = action_parts.index("with")
        object_name = " ".join(action_parts[1:with_index])
        receptacle_name = " ".join(action_parts[with_index+1:])
        current_location = _agent_locations[env_id]
        
        # Check if agent is at the cleaning receptacle
        if current_location != receptacle_name:
            logger.debug(f"Agent attempted to clean with '{receptacle_name}' while at '{current_location}'")
            return f"You cannot clean {object_name} with {receptacle_name} without first moving to it. You need to use 'go to {receptacle_name}' first.", False, False
        
        # Check if the object is in inventory
        if object_name not in _agent_inventories[env_id]:
            # First check with inventory action to ensure our tracking is accurate
            inventory_check_action = "inventory"
            inv_obs, _, _, inv_info = env.step([inventory_check_action])
            inv_result = inv_obs[0].lower()
            
            # Update our inventory tracking
            inventory_items = re.findall(r'([\w\s]+\d+)', inv_result.lower())
            if inventory_items:
                _agent_inventories[env_id] = set(item.strip() for item in inventory_items)
                logger.debug(f"Updated inventory during cleaning: {_agent_inventories[env_id]}")
            
            # Check again after updating inventory
            if object_name not in _agent_inventories[env_id]:
                logger.debug(f"Agent attempted to clean '{object_name}' but it's not in inventory")
                return f"You cannot clean {object_name} with {receptacle_name} because it's not in your inventory. You need to take {object_name} before you can clean it.", False, False
        
        # Check if receptacle is actually capable of cleaning (simple heuristic)
        is_cleaning_receptacle = any(cleaning_word in receptacle_name for cleaning_word in _cleaning_receptacles)
        
        if not is_cleaning_receptacle:
            logger.debug(f"Agent attempted to clean with non-cleaning receptacle: {receptacle_name}")
            return f"{receptacle_name} cannot be used for cleaning. Try using a sink, bathtub, or washing machine instead.", False, False
        
        # If all checks pass, try the cleaning action
        obs, reward, done, info = env.step([agent_action])
        result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
        
        # Provide clearer feedback if the action still fails
        if "Nothing happens" in result_obs:
            return f"You attempt to clean {object_name} with {receptacle_name}, but nothing happens. This may not be the right approach for this task, or the object may already be clean.", result_reward, result_done
        
        # Provide clear success feedback
        if "You " not in result_obs:
            return f"You successfully clean {object_name} with {receptacle_name}.", result_reward, result_done
        
        return result_obs, result_reward, result_done
    
    # Check if this is an examine action targeting a receptacle or object
    elif action_type == "examine" and len(action_parts) >= 2:
        target = " ".join(action_parts[1:])
        current_location = _agent_locations[env_id]
        
        # Enhanced handling for Analysis Result 16 - check if target is mentioned in task
        base_target = target.rstrip("0123456789 ").strip()
        is_task_object = target in task.lower() or base_target in task.lower()
        
        # Check if we're examining a receptacle or object (things that typically end with a number)
        if re.search(r'\d+$', target):
            # First determine if this is a receptacle or an object
            # If the target matches current location or is in inventory, it's okay to examine
            if target == current_location or target in _agent_inventories[env_id]:
                # It's okay to examine
                obs, reward, done, info = env.step([agent_action])
                result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
                
                # Enhanced feedback for Analysis Result 5 - clarify if examination was successful
                if not ("You can't" in result_obs or "cannot" in result_obs.lower() or "can not" in result_obs.lower()):
                    if "You " not in result_obs and "Contents" not in result_obs:
                        return f"You examine {target} and observe its features.", result_reward, result_done
                
                return result_obs, result_reward, result_done
            else:
                # Need to check if this is an object at the current location or if it exists at all
                # First do a look action to see what's visible in the environment
                look_obs, _, _, look_info = env.step(["look"])
                look_result = look_obs[0].lower()
                
                # Check if the target object is mentioned in the look result
                if target in look_result and current_location:
                    # It's an object at the current location, so it's okay to examine
                    obs, reward, done, info = env.step([agent_action])
                    result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
                    
                    # Enhanced feedback for clarity
                    if "You " not in result_obs and "Contents" not in result_obs:
                        return f"You examine {target} and observe its features.", result_reward, result_done
                    
                    return result_obs, result_reward, result_done
                # Addressing Analysis Result 16 - provide better feedback for task-relevant objects
                elif is_task_object:
                    logger.debug(f"Agent attempted to examine task-relevant '{target}' that's not immediately visible")
                    return f"You don't see {target} in your current location. Since this object is relevant to your task, try exploring different receptacles in the environment to find it.", False, False
                elif target in init_obs.lower() or target in task.lower():
                    # The object exists in the environment but not at current location
                    logger.debug(f"Agent attempted to examine '{target}' while at '{current_location}'")
                    return f"You cannot examine {target} without first moving to it. You need to use 'go to {target}' before you can examine it.", False, False
                else:
                    # The object doesn't appear to exist in the environment
                    logger.debug(f"Agent attempted to examine '{target}' which is not visible in the environment")
                    return f"You don't see {target} in the environment. Make sure you're looking for an object that exists.", False, False
        
        # For examining non-numbered objects, we still need to check if they exist
        # First do a look action to see what's visible
        look_obs, _, _, look_info = env.step(["look"])
        look_result = look_obs[0].lower()
        
        # Check if the target is mentioned in the look result, init_obs, or task
        if target in look_result or target in init_obs.lower() or target in task.lower():
            # The object exists, proceed with examination
            obs, reward, done, info = env.step([agent_action])
            result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
            
            # If failed with generic message, provide more specific feedback
            if "You can't" in result_obs or "Nothing happens" in result_obs:
                # Addressing Analysis Result 16 - provide better guidance for task-relevant objects
                if is_task_object:
                    return f"You need to explore the environment to find {target}. Try examining different receptacles to locate it, as it's needed for your task.", False, False
                else:
                    return f"You need to be at the same location as {target} to examine it. Try moving to where the object is located first.", False, False
            
            return result_obs, result_reward, result_done
        else:
            # The object doesn't appear to exist
            logger.debug(f"Agent attempted to examine '{target}' which is not visible in the environment")
            # Addressing Analysis Result 16 - provide better guidance for task objects
            if is_task_object:
                return f"You don't see {target} in your current view, but it's needed for your task. Try exploring the environment by examining different receptacles to find it.", False, False
            else:
                return f"You don't see {target} in the environment. Make sure you're looking for an object that exists.", False, False
    
    # Handle cooling actions - added logic for Analysis Result 3
    elif action_type == "cool" and "with" in action_parts:
        with_index = action_parts.index("with")
        object_name = " ".join(action_parts[1:with_index])
        receptacle_name = " ".join(action_parts[with_index+1:])
        current_location = _agent_locations[env_id]
        
        # Check if agent is at the cooling receptacle
        if current_location != receptacle_name:
            logger.debug(f"Agent attempted to cool with '{receptacle_name}' while at '{current_location}'")
            return f"You cannot cool {object_name} with {receptacle_name} without first moving to it. You need to use 'go to {receptacle_name}' first.", False, False
        
        # Check if the object is in inventory
        if object_name not in _agent_inventories[env_id]:
            # First check with inventory action to ensure our tracking is accurate
            inventory_check_action = "inventory"
            inv_obs, _, _, inv_info = env.step([inventory_check_action])
            inv_result = inv_obs[0].lower()
            
            # Update our inventory tracking
            inventory_items = re.findall(r'([\w\s]+\d+)', inv_result.lower())
            if inventory_items:
                _agent_inventories[env_id] = set(item.strip() for item in inventory_items)
                logger.debug(f"Updated inventory during cooling: {_agent_inventories[env_id]}")
            
            # Check again after updating
            if object_name not in _agent_inventories[env_id]:
                logger.debug(f"Agent attempted to cool '{object_name}' but it's not in inventory")
                
                # Enhanced for Analysis Result 15 - check if object is mentioned in task
                if object_name in task.lower() or object_name.rstrip("0123456789 ").strip() in task.lower():
                    return f"You cannot cool {object_name} with {receptacle_name} because it's not in your inventory. You need to find and take {object_name} before you can cool it. Try exploring different receptacles in the environment.", False, False
                else:
                    return f"You cannot cool {object_name} with {receptacle_name} because it's not in your inventory. You need to take {object_name} before you can cool it.", False, False
        
        # Check if receptacle is actually capable of cooling
        is_cooling_receptacle = any(cooling_word in receptacle_name for cooling_word in _cooling_receptacles)
        
        if not is_cooling_receptacle:
            logger.debug(f"Agent attempted to cool with non-cooling receptacle: {receptacle_name}")
            return f"{receptacle_name} cannot be used for cooling. Try using a fridge or freezer instead.", False, False
        
        # If all checks pass, try the cooling action
        obs, reward, done, info = env.step([agent_action])
        result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
        
        # Provide clearer feedback if the action still fails
        if "Nothing happens" in result_obs:
            return f"You attempt to cool {object_name} with {receptacle_name}, but nothing happens. This may not be the right approach for this task, or the object may already be cool.", result_reward, result_done
        
        # Provide clear success feedback
        if "You " not in result_obs:
            return f"You successfully cool {object_name} with {receptacle_name}.", result_reward, result_done
        
        return result_obs, result_reward, result_done
    
    # Handle different phrasings for placing objects on receptacles - enhanced for Analysis Result 4 and 7
    elif (action_type == "move" and "to" in action_parts) or \
         (action_type == "put" and "on" in action_parts) or \
         (action_type == "put" and "in" in action_parts) or \
         (action_type == "place" and ("on" in action_parts or "in" in action_parts)):
        
        # Extract object and receptacle based on different phrasings
        if action_type == "move" and "to" in action_parts:
            # Format: move [object] to [receptacle]
            to_index = action_parts.index("to")
            object_name = " ".join(action_parts[1:to_index])
            receptacle_name = " ".join(action_parts[to_index+1:])
            standardized_action = f"move {object_name} to {receptacle_name}"
        elif action_type == "put" and "on" in action_parts:
            # Format: put [object] on [receptacle]
            on_index = action_parts.index("on")
            object_name = " ".join(action_parts[1:on_index])
            receptacle_name = " ".join(action_parts[on_index+1:])
            standardized_action = f"move {object_name} to {receptacle_name}"
        elif action_type == "put" and "in" in action_parts:
            # Format: put [object] in [receptacle]
            in_index = action_parts.index("in")
            object_name = " ".join(action_parts[1:in_index])
            receptacle_name = " ".join(action_parts[in_index+1:])
            standardized_action = f"move {object_name} to {receptacle_name}"
        elif action_type == "place" and "on" in action_parts:
            # Format: place [object] on [receptacle]
            on_index = action_parts.index("on")
            object_name = " ".join(action_parts[1:on_index])
            receptacle_name = " ".join(action_parts[on_index+1:])
            standardized_action = f"move {object_name} to {receptacle_name}"
        elif action_type == "place" and "in" in action_parts:
            # Format: place [object] in [receptacle]
            in_index = action_parts.index("in")
            object_name = " ".join(action_parts[1:in_index])
            receptacle_name = " ".join(action_parts[in_index+1:])
            standardized_action = f"move {object_name} to {receptacle_name}"
        else:
            # Shouldn't reach here based on the conditional, but just in case
            return "I don't understand how to place the object. Please use 'move [object] to [receptacle]'.", False, False
        
        logger.debug(f"Standardized placement action: '{standardized_action}' from original: '{agent_action}'")
        
        # Check if the agent is at the receptacle's location
        current_location = _agent_locations[env_id]
        if current_location != receptacle_name and re.search(r'\d+$', receptacle_name):
            logger.debug(f"Agent attempted to place object on '{receptacle_name}' while at '{current_location}'")
            # Enhanced feedback for Analysis Result 7 - explicitly suggesting the next correct action
            return f"You cannot place objects on {receptacle_name} without first moving to it. Use 'go to {receptacle_name}' first before attempting to place {object_name} there.", False, False
        
        # Check if the object is in inventory
        if object_name not in _agent_inventories[env_id]:
            # First check with inventory action to ensure our tracking is accurate
            inventory_check_action = "inventory"
            inv_obs, _, _, inv_info = env.step([inventory_check_action])
            inv_result = inv_obs[0].lower()
            
            # Update our inventory tracking
            inventory_items = re.findall(r'([\w\s]+\d+)', inv_result.lower())
            if inventory_items:
                _agent_inventories[env_id] = set(item.strip() for item in inventory_items)
                logger.debug(f"Updated inventory during placement: {_agent_inventories[env_id]}")
            
            # Check again after updating
            if object_name not in _agent_inventories[env_id]:
                logger.debug(f"Agent attempted to place '{object_name}' but it's not in inventory")
                
                # Enhanced for Analysis Result 15 - check if object is mentioned in task
                if object_name in task.lower() or object_name.rstrip("0123456789 ").strip() in task.lower():
                    return f"You need to find and take {object_name} before you can place it on {receptacle_name}. Since this object is needed for your task, try exploring different receptacles to locate it.", False, False
                else:
                    return f"You need to take {object_name} before you can place it on {receptacle_name}. Check your inventory first to see what objects you are carrying.", False, False
        
        # Now that we've verified the object is in inventory, proceed with the placement action
        obs, reward, done, info = env.step([standardized_action])
        result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
        
        # If the action still failed for other reasons, log this for debugging
        if "You can't" in result_obs or "cannot" in result_obs.lower() or "can not" in result_obs.lower() or "Nothing happens" in result_obs:
            logger.debug(f"Object placement action failed despite checks: {result_obs}")
            return f"The placement action failed. Make sure {receptacle_name} can hold {object_name} and that you're at the correct location.", result_reward, result_done
        else:
            # Update inventory when successfully placing an object
            if object_name in _agent_inventories[env_id]:
                _agent_inventories[env_id].remove(object_name)
                logger.debug(f"Removed {object_name} from inventory after placement: {_agent_inventories[env_id]}")
            
            # Provide clear success feedback for Analysis Result 4
            if "You " not in result_obs:  # Only modify if no clear feedback
                return f"You successfully place {object_name} on {receptacle_name}.", result_reward, result_done
        
        return result_obs, result_reward, result_done
    
    # Handle heating actions with similar structure to cooling and cleaning
    elif action_type == "heat" and "with" in action_parts:
        with_index = action_parts.index("with")
        object_name = " ".join(action_parts[1:with_index])
        receptacle_name = " ".join(action_parts[with_index+1:])
        current_location = _agent_locations[env_id]
        
        # Check if agent is at the heating receptacle
        if current_location != receptacle_name:
            logger.debug(f"Agent attempted to heat with '{receptacle_name}' while at '{current_location}'")
            return f"You cannot heat {object_name} with {receptacle_name} without first moving to it. You need to use 'go to {receptacle_name}' first.", False, False
        
        # Check if the object is in inventory
        if object_name not in _agent_inventories[env_id]:
            # First check with inventory action to ensure our tracking is accurate
            inventory_check_action = "inventory"
            inv_obs, _, _, inv_info = env.step([inventory_check_action])
            inv_result = inv_obs[0].lower()
            
            # Update our inventory tracking
            inventory_items = re.findall(r'([\w\s]+\d+)', inv_result.lower())
            if inventory_items:
                _agent_inventories[env_id] = set(item.strip() for item in inventory_items)
                logger.debug(f"Updated inventory during heating: {_agent_inventories[env_id]}")
            
            # Check again after updating
            if object_name not in _agent_inventories[env_id]:
                logger.debug(f"Agent attempted to heat '{object_name}' but it's not in inventory")
                
                # Enhanced for Analysis Result 15 - check if object is mentioned in task
                if object_name in task.lower() or object_name.rstrip("0123456789 ").strip() in task.lower():
                    return f"You cannot heat {object_name} with {receptacle_name} because it's not in your inventory. You need to find and take {object_name} before you can heat it. Try exploring different receptacles in the environment.", False, False
                else:
                    return f"You cannot heat {object_name} with {receptacle_name} because it's not in your inventory. You need to take {object_name} before you can heat it.", False, False
        
        # If all checks pass, try the heating action
        obs, reward, done, info = env.step([agent_action])
        result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
        
        # Provide clearer feedback if the action still fails
        if "Nothing happens" in result_obs:
            return f"You attempt to heat {object_name} with {receptacle_name}, but nothing happens. This may not be the right approach for this task, or the object may already be heated.", result_reward, result_done
        
        # Provide clear success feedback
        if "You " not in result_obs:
            return f"You successfully heat {object_name} with {receptacle_name}.", result_reward, result_done
        
        return result_obs, result_reward, result_done
    
    # Handle slicing actions
    elif action_type == "slice" and "with" in action_parts:
        with_index = action_parts.index("with")
        object_name = " ".join(action_parts[1:with_index])
        tool_name = " ".join(action_parts[with_index+1:])
        
        # Check if the object to be sliced is in inventory
        if object_name not in _agent_inventories[env_id]:
            # First check with inventory action to ensure our tracking is accurate
            inventory_check_action = "inventory"
            inv_obs, _, _, inv_info = env.step([inventory_check_action])
            inv_result = inv_obs[0].lower()
            
            # Update our inventory tracking
            inventory_items = re.findall(r'([\w\s]+\d+)', inv_result.lower())
            if inventory_items:
                _agent_inventories[env_id] = set(item.strip() for item in inventory_items)
                logger.debug(f"Updated inventory during slicing: {_agent_inventories[env_id]}")
            
            # Check again after updating
            if object_name not in _agent_inventories[env_id]:
                logger.debug(f"Agent attempted to slice '{object_name}' but it's not in inventory")
                
                # Enhanced for Analysis Result 15 - check if object is mentioned in task
                if object_name in task.lower() or object_name.rstrip("0123456789 ").strip() in task.lower():
                    return f"You cannot slice {object_name} with {tool_name} because it's not in your inventory. You need to find and take {object_name} before you can slice it. Try exploring different receptacles in the environment.", False, False
                else:
                    return f"You cannot slice {object_name} with {tool_name} because it's not in your inventory. You need to take {object_name} before you can slice it.", False, False
        
        # Check if the slicing tool is in inventory
        if tool_name not in _agent_inventories[env_id]:
            # First check with inventory action to ensure our tracking is accurate
            inventory_check_action = "inventory"
            inv_obs, _, _, inv_info = env.step([inventory_check_action])
            inv_result = inv_obs[0].lower()
            
            # Update our inventory tracking
            inventory_items = re.findall(r'([\w\s]+\d+)', inv_result.lower())
            if inventory_items:
                _agent_inventories[env_id] = set(item.strip() for item in inventory_items)
                logger.debug(f"Updated inventory during slicing tool check: {_agent_inventories[env_id]}")
            
            # Check again after updating
            if tool_name not in _agent_inventories[env_id]:
                logger.debug(f"Agent attempted to slice with '{tool_name}' but it's not in inventory")
                
                # Enhanced for Analysis Result 15 - check if tool is mentioned in task
                if tool_name in task.lower() or tool_name.rstrip("0123456789 ").strip() in task.lower():
                    return f"You cannot slice {object_name} with {tool_name} because you don't have {tool_name} in your inventory. You need to find and take {tool_name} first. Try exploring different receptacles.", False, False
                else:
                    return f"You cannot slice {object_name} with {tool_name} because you don't have {tool_name} in your inventory. You need to take {tool_name} first.", False, False
        
        # If all checks pass, try the slicing action
        obs, reward, done, info = env.step([agent_action])
        result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
        
        # Provide clearer feedback if the action still fails
        if "Nothing happens" in result_obs:
            return f"You attempt to slice {object_name} with {tool_name}, but nothing happens. This may not be the right approach for this task, or the object may not be sliceable.", result_reward, result_done
        
        # Provide clear success feedback
        if "You " not in result_obs:
            return f"You successfully slice {object_name} with {tool_name}.", result_reward, result_done
        
        return result_obs, result_reward, result_done
    
    # For all other cases, proceed with the normal environment step
    obs, reward, done, info = env.step([agent_action])
    result_obs, result_reward, result_done = obs[0], info['won'][0], done[0]
    
    # Enhanced for Analysis Results 15 & 16 - provide better feedback for generic responses
    if "Nothing happens" in result_obs:
        logger.debug(f"Generic 'Nothing happens' response for action: {agent_action}")
        
        # Check if action involves a task-relevant object
        action_involves_task_object = any(obj_word in task.lower() for obj_word in action_parts[1:])
        
        if action_involves_task_object:
            return f"You tried '{agent_action}', but nothing happens. Since this action involves an object mentioned in your task, try exploring the environment more thoroughly. The object might be located in a different receptacle or require a different approach.", result_reward, result_done
        else:
            return f"You tried '{agent_action}', but nothing happens. This might not be the right approach, or you might need to be in a different location or have different objects in your inventory.", result_reward, result_done
    
    return result_obs, result_reward, result_done