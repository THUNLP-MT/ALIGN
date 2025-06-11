def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic for different task types.
    Returns a more detailed set of rules to help the agent understand the environment better.
    """
    rules = """
    Environment Rules:
    1. You must go to a location before you can examine objects or receptacles at that location.
    2. To interact with objects or receptacles, you must first be at their location.
    3. To examine an object with a tool, you must follow these sequence of steps:
       - First take the object
       - Then go to the tool's location
       - Finally use the tool
    4. You can only take objects that exist in their specified locations.
    5. To take an object from a closed receptacle, you must first open the receptacle.
    6. Compound actions like "examine [object] with [tool]" are not directly supported - you must break these down into the individual steps listed in rule 3.
    """
    return rules

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action and return the next observation, reward, and done status.
    Provides more informative feedback when actions cannot be completed.
    """
    action_item = {
        'matched': False,
        'action': None,
        'object': None,
        'receptacle': None,
        'object2': None
    }

    # Simple actions without parameters
    if agent_action == 'look' or agent_action == 'inventory':
        action_item['matched'] = True
        action_item['action'] = agent_action
    
    # Pattern: go to (receptacle)
    if agent_action.startswith('go to '):
        receptacle = agent_action[6:].strip()
        action_item['matched'] = True
        action_item['action'] = 'go to'
        action_item['receptacle'] = receptacle
    
    # Pattern: open/close (receptacle)
    for action in ['open ', 'close ']:
        if agent_action.startswith(action):
            receptacle = agent_action[len(action):].strip()
            action_item['matched'] = True
            action_item['action'] = action.strip()
            action_item['receptacle'] = receptacle
    
    # Pattern: take (object) from (receptacle)
    if 'take ' in agent_action and ' from ' in agent_action:
        parts = agent_action.split(' from ')
        if len(parts) == 2:
            obj = parts[0][5:].strip()  # Remove 'take ' prefix
            receptacle = parts[1].strip()
            action_item['matched'] = True
            action_item['action'] = 'take from'
            action_item['object'] = obj
            action_item['receptacle'] = receptacle
            
            # Check if agent is at the receptacle location
            current_obs, _, _, _ = env.step(['look'])
            current_obs = current_obs[0]
            logger.debug(f"Current observation before take from: {current_obs}")
            
            # If receptacle isn't in current location
            if receptacle not in current_obs:
                return f"You need to go to the location of the {receptacle} before you can take anything from it.", False, False
            
            # Try to look in the receptacle
            check_obs, _, _, _ = env.step([f'examine {receptacle}'])
            check_obs = check_obs[0]
            logger.debug(f"Check observation for {obj} in {receptacle}: {check_obs}")
            
            # If receptacle is closed
            if "closed" in check_obs.lower():
                return f"The {receptacle} is closed. You need to open it first before taking anything from it.", False, False
            
            # If object not found in receptacle contents
            if "no objects" in check_obs.lower() or obj not in check_obs:
                return f"There is no {obj} in the {receptacle}.", False, False
    
    # Pattern: move (object) to (receptacle)
    if 'move ' in agent_action and ' to ' in agent_action:
        parts = agent_action.split(' to ')
        if len(parts) == 2:
            obj = parts[0][5:].strip()  # Remove 'move ' prefix
            receptacle = parts[1].strip()
            action_item['matched'] = True
            action_item['action'] = 'move to'
            action_item['object'] = obj
            action_item['receptacle'] = receptacle
    
    # Handle invalid "examine with tool" action (Misalignment #2)
    if agent_action.startswith('examine ') and ' with ' in agent_action:
        parts = agent_action.split(' with ')
        if len(parts) == 2:
            obj = parts[0][8:].strip()  # Remove 'examine ' prefix
            tool = parts[1].strip()
            logger.debug(f"Detected invalid compound examine: {obj} with {tool}")
            return "You cannot examine an object with a tool directly. Try taking the object, moving to the tool's location, and using the tool.", False, False
    
    # Pattern: examine (something)
    if agent_action.startswith('examine '):
        something = agent_action[8:].strip()
        action_item['matched'] = True
        action_item['action'] = 'examine'
        if something in init_obs:
            action_item['receptacle'] = something
        else:
            action_item['object'] = something
            
        # Get the current observation to check agent's location
        current_obs, _, _, _ = env.step(['look'])
        current_obs = current_obs[0]
        logger.debug(f"Current observation before examine: {current_obs}")
        
        # Check if the target is at the current location (Misalignment #1)
        target = action_item['receptacle'] if action_item['receptacle'] else action_item['object']
        if target and target not in current_obs:
            logger.debug(f"Target {target} not in current location")
            return f"You need to go to the location of the {target} before you can examine it.", False, False
    
    # Pattern: use (object)
    if agent_action.startswith('use '):
        obj = agent_action[4:].strip()
        action_item['matched'] = True
        action_item['action'] = 'use'
        action_item['object'] = obj
    
    # Pattern: heat/clean/cool (object) with (receptacle)
    for action in ['heat ', 'clean ', 'cool ']:
        if agent_action.startswith(action) and ' with ' in agent_action:
            parts = agent_action.split(' with ')
            if len(parts) == 2:
                obj = parts[0][len(action):].strip()
                receptacle = parts[1].strip()
                action_item['matched'] = True
                action_item['action'] = action.strip()
                action_item['object'] = obj
                action_item['receptacle'] = receptacle
    
    # Pattern: slice (object) with (object)
    if agent_action.startswith('slice ') and ' with ' in agent_action:
        parts = agent_action.split(' with ')
        if len(parts) == 2:
            obj = parts[0][6:].strip()  # Remove 'slice ' prefix
            obj2 = parts[1].strip()
            action_item['matched'] = True
            action_item['action'] = 'slice'
            action_item['object'] = obj
            action_item['object2'] = obj2  # Using object2 for the tool used for slicing

    # Execute the action in the environment
    obs, reward, done, info = env.step([agent_action])
    obs, reward, done = obs[0], info['won'][0], done[0]
    
    # Check if "Nothing happens" is the response and add more informative feedback
    if obs == "Nothing happens.":
        logger.debug(f"Default 'Nothing happens' response for action: {agent_action}")
        
        # This covers cases where the action pattern was matched but didn't produce a result
        # Could be due to various reasons not caught by earlier checks
        if action_item['matched']:
            if action_item['action'] == 'take from':
                return f"Cannot take {action_item['object']} from {action_item['receptacle']}. Check if the object exists in this location or if the receptacle is open.", False, False
            elif action_item['action'] == 'examine':
                target = action_item['receptacle'] if action_item['receptacle'] else action_item['object']
                return f"Cannot examine {target}. Make sure you are at the correct location.", False, False
            elif action_item['action'] == 'go to':
                return f"Cannot go to {action_item['receptacle']}. This location might not exist in the current environment.", False, False
        
        # For cases where we don't have a specific handling but still want to be more informative
        return "Nothing happens. This action might not be valid in the current context. Check your current location and try a different action.", False, False
    
    return obs, reward, done