def InferRules(init_obs, task):
    """
    Provides the rules for environment and task execute logic for different task types.
    
    Args:
        init_obs: Initial observation string containing information about the environment
        task: The specific task description
        
    Returns:
        A string describing the environment rules
    """
    return """
    1. Navigation and Location Rules:
       - You must go to a receptacle before you can examine it, open it, close it, or interact with objects in/on it.
       - You can only interact with objects and receptacles that are at your current location.
       - If you try to interact with a receptacle or object that is not at your current location, you will be informed that you need to go to that location first.
       - After successfully going to a location, you are at that location until you go somewhere else.

    2. Object Interaction Rules:
       - To take an object, it must be present at your current location and visible (not inside a closed receptacle).
       - Once you take an object, it goes into your inventory and is no longer at its original location.
       - To move an object to a receptacle, you must have the object in your inventory and be at the target receptacle.
       - To use, heat, clean, cool, or slice objects, you must have the required objects in your inventory or be at their location.
       - You cannot take an object that is already in your inventory.

    3. Container Rules:
       - Some receptacles can be opened and closed (like refrigerators, microwaves, cabinets, etc.).
       - You must open a closed container before you can take objects from it or put objects into it.
       - Objects inside closed containers are not visible or accessible until the container is opened.

    4. Action Sequence Requirements:
       - Some tasks require a specific sequence of actions - for example, to heat food, you need to:
         a) Go to the microwave
         b) Open the microwave
         c) Place the food inside
         d) Close the microwave
         e) Use the microwave
       - The environment will guide you if you're missing a prerequisite step for an action.

    5. Feedback Interpretation:
       - If an action cannot be performed, the environment will explain why and what prerequisites are needed.
       - The environment will inform you if you try to take an object that's already in your inventory.
       - The environment will inform you if you try to move an object that's not in your inventory.
       - Pay attention to the feedback to understand the current state of the environment and what actions are possible next.
       - When you successfully go to a location, the environment will describe what's there.
    """

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action and return the next observation, reward, and done status.
    
    Args:
        env: The environment object
        init_obs: Initial observation string containing information about the environment
        task: The specific task description
        agent_action: The action string from the agent
        logger: Logger object for debugging information
        
    Returns:
        obs: Observation string after the action
        reward: Boolean indicating if a reward was received
        done: Boolean indicating if the task is complete
    """
    # Track the agent's current location using an attribute on the env object
    if not hasattr(env, '_current_location'):
        env._current_location = None
    
    # Track container states (open/closed) using an attribute on the env object
    if not hasattr(env, '_container_states'):
        env._container_states = {}
    
    action_item = {
        'matched': False,
        'action': None,
        'object': None,
        'receptacle': None,
        'object2': None
    }

    # Parse the agent action
    # Simple actions without parameters
    if agent_action.lower() == 'look' or agent_action.lower() == 'inventory':
        action_item['matched'] = True
        action_item['action'] = agent_action.lower()
    
    # Pattern: go to (receptacle)
    elif agent_action.lower().startswith('go to '):
        receptacle = agent_action[6:].strip()
        action_item['matched'] = True
        action_item['action'] = 'go to'
        action_item['receptacle'] = receptacle
    
    # Pattern: open/close (receptacle)
    elif agent_action.lower().startswith('open ') or agent_action.lower().startswith('close '):
        action = 'open' if agent_action.lower().startswith('open ') else 'close'
        receptacle = agent_action[len(action)+1:].strip()
        action_item['matched'] = True
        action_item['action'] = action
        action_item['receptacle'] = receptacle
    
    # Pattern: take (object) from (receptacle)
    elif 'take ' in agent_action.lower() and ' from ' in agent_action.lower():
        parts = agent_action.split(' from ')
        if len(parts) == 2:
            obj = parts[0][5:].strip()  # Remove 'take ' prefix
            receptacle = parts[1].strip()
            action_item['matched'] = True
            action_item['action'] = 'take from'
            action_item['object'] = obj
            action_item['receptacle'] = receptacle
    
    # Pattern: move (object) to (receptacle)
    elif 'move ' in agent_action.lower() and ' to ' in agent_action.lower():
        parts = agent_action.split(' to ')
        if len(parts) == 2:
            obj = parts[0][5:].strip()  # Remove 'move ' prefix
            receptacle = parts[1].strip()
            action_item['matched'] = True
            action_item['action'] = 'move to'
            action_item['object'] = obj
            action_item['receptacle'] = receptacle
    
    # Pattern: examine (something)
    elif agent_action.lower().startswith('examine '):
        something = agent_action[8:].strip()
        action_item['matched'] = True
        action_item['action'] = 'examine'
        
        # Determine if it's a receptacle or object by checking if it appears in the initial observation
        if something.lower() in init_obs.lower():
            action_item['receptacle'] = something
        else:
            action_item['object'] = something
    
    # Pattern: use (object)
    elif agent_action.lower().startswith('use '):
        obj = agent_action[4:].strip()
        action_item['matched'] = True
        action_item['action'] = 'use'
        action_item['object'] = obj
    
    # Pattern: heat/clean/cool (object) with (receptacle)
    elif any(agent_action.lower().startswith(action) for action in ['heat ', 'clean ', 'cool ']) and ' with ' in agent_action.lower():
        for action in ['heat ', 'clean ', 'cool ']:
            if agent_action.lower().startswith(action):
                parts = agent_action.split(' with ')
                if len(parts) == 2:
                    obj = parts[0][len(action):].strip()
                    receptacle = parts[1].strip()
                    action_item['matched'] = True
                    action_item['action'] = action.strip()
                    action_item['object'] = obj
                    action_item['receptacle'] = receptacle
                break
    
    # Pattern: slice (object) with (object)
    elif agent_action.lower().startswith('slice ') and ' with ' in agent_action.lower():
        parts = agent_action.split(' with ')
        if len(parts) == 2:
            obj = parts[0][6:].strip()  # Remove 'slice ' prefix
            obj2 = parts[1].strip()
            action_item['matched'] = True
            action_item['action'] = 'slice'
            action_item['object'] = obj
            action_item['object2'] = obj2  # Using object2 for the tool used for slicing
    
    # If action wasn't matched, provide feedback
    if not action_item['matched']:
        return f"I don't understand the action '{agent_action}'. Please use one of the allowed actions from the action space.", False, False
    
    logger.debug(f"Parsed action: {action_item}")
    
    # Get the current observation to check location
    test_obs, _, _, _ = env.step(['look'])
    test_obs = test_obs[0]
    logger.debug(f"Current observation after 'look': {test_obs}")
    
    # Get inventory to check what objects the agent has
    inventory_obs, _, _, _ = env.step(['inventory'])
    inventory_obs = inventory_obs[0]
    logger.debug(f"Current inventory observation: {inventory_obs}")
    
    # Improved function to check if an object is in inventory
    def is_in_inventory(object_name):
        object_name_lower = object_name.lower()
        logger.debug(f"Checking if '{object_name_lower}' is in inventory")
        
        # Extract inventory items from the observation
        inventory_items = []
        
        # Check for common inventory patterns
        if "carrying:" in inventory_obs.lower():
            carrying_section = inventory_obs.lower().split("carrying:")[1].strip()
            inventory_items = [item.strip() for item in carrying_section.split(',')]
        elif "inventory:" in inventory_obs.lower():
            inventory_section = inventory_obs.lower().split("inventory:")[1].strip()
            inventory_items = [item.strip() for item in inventory_section.split(',')]
        elif "you are carrying:" in inventory_obs.lower():
            carrying_section = inventory_obs.lower().split("you are carrying:")[1].strip()
            inventory_items = [item.strip() for item in carrying_section.split(',')]
        
        # Also check line by line for inventory items
        inventory_lines = inventory_obs.lower().split('\n')
        for line in inventory_lines:
            line = line.strip()
            if line and not line.startswith(("you are", "carrying:", "inventory:")):
                inventory_items.append(line)
        
        logger.debug(f"Extracted inventory items: {inventory_items}")
        
        # Check if object_name or its base name (without numbers) is in inventory
        base_name = ''.join([c for c in object_name_lower if not c.isdigit()]).strip()
        
        for item in inventory_items:
            # Check for exact match
            if object_name_lower == item or f"{object_name_lower} (in your inventory)" == item:
                logger.debug(f"Found exact match '{item}' in inventory")
                return True
            
            # Check for base name match (without numbers)
            if base_name != object_name_lower and (base_name == item or f"{base_name} (in your inventory)" == item):
                logger.debug(f"Found base name match '{item}' in inventory")
                return True
            
            # Check if item contains the object name
            if object_name_lower in item:
                logger.debug(f"Found partial match '{item}' containing '{object_name_lower}' in inventory")
                return True
            
            # Check if item contains the base name
            if base_name != object_name_lower and base_name in item:
                logger.debug(f"Found partial match '{item}' containing base name '{base_name}' in inventory")
                return True
        
        # Direct check for common patterns in the full inventory text
        patterns = [
            f"carrying: {object_name_lower}",
            f"{object_name_lower} (in your inventory)",
            f"you are carrying: {object_name_lower}",
            f"inventory: {object_name_lower}"
        ]
        
        if base_name != object_name_lower:
            patterns.extend([
                f"carrying: {base_name}",
                f"{base_name} (in your inventory)",
                f"you are carrying: {base_name}",
                f"inventory: {base_name}"
            ])
        
        for pattern in patterns:
            if pattern in inventory_obs.lower():
                logger.debug(f"Found pattern '{pattern}' in inventory text")
                return True
        
        logger.debug(f"'{object_name_lower}' not found in inventory")
        return False
    
    # Helper function to check if we're at a location
    def is_at_location(location_name):
        location_name_lower = location_name.lower()
        
        # If we've already tracked this location, use the tracked value
        if env._current_location and location_name_lower in env._current_location.lower():
            logger.debug(f"Using tracked location: '{env._current_location}'")
            return True
        
        # Check if location is mentioned in current observation after "You are in"
        if "you are in" in test_obs.lower() and location_name_lower in test_obs.lower():
            logger.debug(f"Location '{location_name_lower}' mentioned in observation after 'You are in'")
            return True
        
        # Check if the location is in the first line of the observation (common format)
        first_line = test_obs.split('\n')[0].lower()
        if location_name_lower in first_line:
            logger.debug(f"Location '{location_name_lower}' found in first line of observation")
            return True
        
        # Check if the observation mentions items at/on the location
        location_patterns = [
            f"on the {location_name_lower}",
            f"in the {location_name_lower}",
            f"at the {location_name_lower}"
        ]
        
        for pattern in location_patterns:
            if pattern in test_obs.lower():
                logger.debug(f"Found pattern '{pattern}' in observation")
                return True
        
        logger.debug(f"Not at location '{location_name_lower}'")
        return False
    
    # Handle go to action
    if action_item['action'] == 'go to':
        receptacle = action_item['receptacle']
        receptacle_lower = receptacle.lower()
        
        # Check if we're already at this location
        if is_at_location(receptacle_lower):
            env._current_location = receptacle
            return f"You are already at the {receptacle}. You can interact with it directly.", False, False
        
        # Execute the go to action
        obs, reward, done, info = env.step([agent_action])
        obs, reward, done = obs[0], info['won'][0], done[0]
        
        # Update the current location if the action was successful
        if obs and "nothing happens" not in obs.lower():
            env._current_location = receptacle
            
            # If the observation doesn't clearly indicate arrival, enhance it
            if not any(phrase in obs.lower() for phrase in [f"you arrive at", f"you are at", f"you see {receptacle_lower}"]):
                obs = f"You arrive at the {receptacle}. {obs}"
        else:
            # Provide more informative feedback
            obs = f"Cannot go to {receptacle}. It might not be a valid location or not accessible from here."
        
        return obs, reward, done
    
    # Handle examine, open, close, take from, move to actions that require being at location
    if action_item['action'] in ['examine', 'open', 'close', 'take from', 'move to']:
        receptacle = action_item['receptacle'].lower() if action_item['receptacle'] else ""
        logger.debug(f"Action: {action_item['action']} with receptacle: {receptacle}")
        
        # Skip location check for examining objects in inventory
        if action_item['action'] == 'examine' and action_item['object'] and is_in_inventory(action_item['object']):
            # Execute the examine action directly
            obs, reward, done, info = env.step([agent_action])
            obs, reward, done = obs[0], info['won'][0], done[0]
            return obs, reward, done
        
        # Check if we need to be at a receptacle and if we're there
        if receptacle and not is_at_location(receptacle):
            action_name = action_item['action']
            if action_name == 'examine':
                return f"You must go to the {action_item['receptacle']} first before examining it.", False, False
            elif action_name == 'take from':
                return f"You need to go to the {action_item['receptacle']} first before taking objects from it.", False, False
            elif action_name == 'move to':
                return f"You need to go to the {action_item['receptacle']} first before placing objects on/in it.", False, False
            else:  # open or close
                return f"You need to go to the {action_item['receptacle']} first before you can {action_name} it.", False, False
    
    # Handle open and close actions to track container states
    if action_item['action'] in ['open', 'close']:
        receptacle = action_item['receptacle']
        
        # Execute the action
        obs, reward, done, info = env.step([agent_action])
        obs, reward, done = obs[0], info['won'][0], done[0]
        
        # Check for "Nothing happens" and provide more informative feedback
        if obs.strip() == "Nothing happens.":
            if action_item['action'] == 'open':
                return f"Unable to open {receptacle}. It might already be open or not be openable.", reward, done
            else:  # close
                return f"Unable to close {receptacle}. It might already be closed or not be closable.", reward, done
        
        # Update container state tracking
        if "successfully" in obs.lower() or "already" in obs.lower():
            env._container_states[receptacle.lower()] = 'open' if action_item['action'] == 'open' else 'closed'
        
        return obs, reward, done
    
    # Check if taking an object that's already in inventory
    if action_item['action'] == 'take from':
        object_name = action_item['object']
        if is_in_inventory(object_name):
            return f"You already have the {object_name} in your inventory. No need to take it again.", False, False
    
    # Check if moving an object that's not in inventory
    if action_item['action'] == 'move to':
        object_name = action_item['object']
        if not is_in_inventory(object_name):
            return f"You don't have the {object_name} in your inventory. You need to take it first.", False, False
    
    # Execute the action in the environment
    logger.debug(f"Executing action in environment: {agent_action}")
    obs, reward, done, info = env.step([agent_action])
    obs, reward, done = obs[0], info['won'][0], done[0]
    logger.debug(f"Environment response: {obs}")
    
    # Handle special case for "Nothing happens" response
    if obs.strip() == "Nothing happens." and action_item['action'] == 'take from':
        object_name = action_item['object']
        receptacle_name = action_item['receptacle']
        
        # Check if it might be because the object is already in inventory
        if is_in_inventory(object_name):
            return f"You already have the {object_name} in your inventory. No need to take it again.", reward, done
        
        # Check if it might be because the container is closed
        receptacle_state = env._container_states.get(receptacle_name.lower())
        if receptacle_state == 'closed':
            return f"You need to open the {receptacle_name} first before taking objects from it.", reward, done
        
        # Otherwise, the object might not be there
        return f"There is no {object_name} at the {receptacle_name} to take. It might be elsewhere or already taken.", reward, done
    
    # Handle special case for "Nothing happens" response for move action
    if obs.strip() == "Nothing happens." and action_item['action'] == 'move to':
        object_name = action_item['object']
        receptacle_name = action_item['receptacle']
        
        # Double-check if the object is in inventory
        if is_in_inventory(object_name):
            # If object is in inventory but move fails, check if receptacle is closed
            receptacle_state = env._container_states.get(receptacle_name.lower())
            if receptacle_state == 'closed':
                return f"You need to open the {receptacle_name} first before placing objects in it.", reward, done
            else:
                return f"Unable to move {object_name} to {receptacle_name}. Make sure the receptacle is open if it's a container.", reward, done
        else:
            # If object is not in inventory, provide clear feedback
            return f"You don't have the {object_name} in your inventory. You need to take it first before moving it.", reward, done
    
    # Handle other "Nothing happens" cases with more informative feedback
    if obs.strip() == "Nothing happens.":
        if action_item['action'] == 'open':
            return f"Unable to open {action_item['receptacle']}. It might already be open or not be openable.", reward, done
        elif action_item['action'] == 'close':
            return f"Unable to close {action_item['receptacle']}. It might already be closed or not be closable.", reward, done
        elif action_item['action'] == 'examine':
            if action_item['object']:
                return f"Unable to examine {action_item['object']}. Make sure you have it in your inventory or it's visible at your location.", reward, done
            else:
                return f"Unable to examine {action_item['receptacle']}. Make sure you're at the right location and it's visible.", reward, done
        elif action_item['action'] == 'use':
            return f"Unable to use {action_item['object']}. Make sure you have it in your inventory or it's at your current location and usable.", reward, done
        elif action_item['action'] in ['heat', 'clean', 'cool', 'slice']:
            return f"Unable to {action_item['action']} {action_item['object']}. Make sure you have all required objects and are at the right location.", reward, done
        elif action_item['action'] == 'go to':
            # This case should be handled earlier, but as a fallback
            return f"Cannot go to {action_item['receptacle']}. It might not be a valid location in this environment.", reward, done
        else:
            # Generic clarification for other actions
            return f"Action '{agent_action}' resulted in no effect. Check if you have all prerequisites or if the action is valid in this context.", reward, done
    
    # For successful move actions, verify the object was actually in inventory
    if "successfully" in obs.lower() and "place" in obs.lower() and action_item['action'] == 'move to':
        object_name = action_item['object']
        # If the environment says the move was successful, we should trust that and not override
        return obs, reward, done
    
    return obs, reward, done