def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execution logic for different task types.
    Returns more comprehensive rules to better guide the agent.
    """
    return """
    Navigation Rule: You must go to a receptacle before you can interact with it. 
    For example, before opening a fridge, examining a countertop, or taking an object from a drawer, 
    you must first navigate to that location using the 'go to' action.
    
    After successfully going to a location, you can interact with that receptacle and objects in it
    without needing to navigate there again, unless you move to a different location.
    
    Interaction Rule: You can only interact with objects that exist in the environment.
    If you try to take an object from a receptacle and it doesn't exist there, the action will fail.
    Similarly, you can only place objects in receptacles that can contain them.
    
    Examination Rule: To examine objects, use "examine [receptacle]" to view contents of a receptacle,
    or "examine [object]" if the object is in your inventory or visible at your current location.
    Syntax like "examine [object] in/on [receptacle]" is not supported.
    
    To examine an object with another object (like examining a CD with a desklamp), 
    you must first have both objects in your possession. Spatial modifiers like "under", "near", or 
    "closely" are not supported for examine actions - you must examine specific objects directly.
    
    Object Manipulation Rule: You can only move objects that you've already taken and have in your 
    inventory. The target receptacle must be capable of containing the object for the action to succeed.
    
    Feedback Rule: The environment will provide specific feedback when actions fail, helping you 
    understand what went wrong and how to correct your approach. Pay close attention to these messages.
    
    Object Location Rule: Objects can be found in various places in the environment. Some objects may be 
    inside containers like drawers, cabinets, or fridges, while others might be on surfaces like countertops, 
    tables, stoveburners, or other appliances. When searching for an object, consider examining all types 
    of receptacles and surfaces in the environment, not just storage containers.
    
    Important: Kitchen items like pans, pots, plates, utensils, and food can be found both inside storage 
    containers (drawers, cabinets, fridges) AND on surfaces (countertops, stoveburners, tables). For a 
    complete search, you should check both storage locations and open surfaces.
    """

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action and return the next observation, reward, and done status.
    Enhanced to provide more informative feedback for common failure cases.
    """
    # Static variable to track agent's current location across function calls
    if not hasattr(WrapStep, "current_location"):
        WrapStep.current_location = None
    
    # Static variable to track agent's inventory
    if not hasattr(WrapStep, "inventory"):
        WrapStep.inventory = set()
    
    # Static variable to track the last observation
    if not hasattr(WrapStep, "last_observation"):
        WrapStep.last_observation = ""
    
    # Static variable to track examined receptacles and their contents
    if not hasattr(WrapStep, "examined_contents"):
        WrapStep.examined_contents = {}
    
    # Static variable to track searched storage locations
    if not hasattr(WrapStep, "searched_storage"):
        WrapStep.searched_storage = set()
    
    # Static variable to track searched surfaces
    if not hasattr(WrapStep, "searched_surfaces"):
        WrapStep.searched_surfaces = set()
    
    # Static variable to track failed storage searches
    if not hasattr(WrapStep, "failed_storage_searches"):
        WrapStep.failed_storage_searches = 0
    
    # Static variable to track if hint has been given
    if not hasattr(WrapStep, "hint_given"):
        WrapStep.hint_given = False
    
    # Static variable to track target objects from task
    if not hasattr(WrapStep, "target_objects"):
        WrapStep.target_objects = []
        # Extract potential target objects from task
        common_objects = ["pan", "pot", "knife", "fork", "spoon", "plate", "cup", "glass", "bowl", 
                         "apple", "banana", "lettuce", "tomato", "potato", "onion", "bread", "soap", 
                         "towel", "book", "cd", "laptop", "phone", "remote", "key"]
        for obj in common_objects:
            if obj in task.lower():
                WrapStep.target_objects.append(obj)
    
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
    elif agent_action.startswith('go to '):
        receptacle = agent_action[6:].strip()
        action_item['matched'] = True
        action_item['action'] = 'go to'
        action_item['receptacle'] = receptacle
    
    # Pattern: open/close (receptacle)
    elif any(agent_action.startswith(action) for action in ['open ', 'close ']):
        action_prefix = 'open ' if agent_action.startswith('open ') else 'close '
        receptacle = agent_action[len(action_prefix):].strip()
        action_item['matched'] = True
        action_item['action'] = action_prefix.strip()
        action_item['receptacle'] = receptacle
    
    # Pattern: take (object) from (receptacle)
    elif 'take ' in agent_action and ' from ' in agent_action:
        parts = agent_action.split(' from ')
        if len(parts) == 2:
            obj = parts[0][5:].strip()  # Remove 'take ' prefix
            receptacle = parts[1].strip()
            action_item['matched'] = True
            action_item['action'] = 'take from'
            action_item['object'] = obj
            action_item['receptacle'] = receptacle
    
    # Pattern: move (object) to (receptacle)
    elif 'move ' in agent_action and ' to ' in agent_action:
        parts = agent_action.split(' to ')
        if len(parts) == 2:
            obj = parts[0][5:].strip()  # Remove 'move ' prefix
            receptacle = parts[1].strip()
            action_item['matched'] = True
            action_item['action'] = 'move to'
            action_item['object'] = obj
            action_item['receptacle'] = receptacle
    
    # Pattern: examine (something)
    elif agent_action.startswith('examine '):
        # Handle "examine [object] in/on [receptacle]" pattern
        if ' in ' in agent_action or ' on ' in agent_action:
            separator = ' in ' if ' in ' in agent_action else ' on '
            parts = agent_action.split(separator)
            if len(parts) == 2:
                obj = parts[0][8:].strip()  # Remove 'examine ' prefix
                receptacle = parts[1].strip()
                action_item['matched'] = True
                action_item['action'] = 'examine in'
                action_item['object'] = obj
                action_item['receptacle'] = receptacle
                # Flag to handle this special case
                action_item['unsupported_syntax'] = True
        elif ' with ' in agent_action:
            # Handle "examine X with Y" pattern
            parts = agent_action.split(' with ')
            if len(parts) == 2:
                obj = parts[0][8:].strip()  # Remove 'examine ' prefix
                obj2 = parts[1].strip()
                action_item['matched'] = True
                action_item['action'] = 'examine with'
                action_item['object'] = obj
                action_item['object2'] = obj2
        else:
            something = agent_action[8:].strip()
            action_item['matched'] = True
            action_item['action'] = 'examine'
            
            # Check for spatial modifiers that are not supported
            spatial_modifiers = ['under', 'near', 'closely', 'above', 'beside', 'behind', 'area']
            has_spatial_modifier = any(modifier in something.lower() for modifier in spatial_modifiers)
            
            if has_spatial_modifier:
                action_item['has_spatial_modifier'] = True
                # Extract the potential objects from the examine command
                for modifier in spatial_modifiers:
                    if f" {modifier} " in f" {something} ":
                        parts = something.split(f" {modifier} ")
                        if len(parts) >= 2:
                            action_item['object'] = parts[0].strip()
                            action_item['modifier'] = modifier
                            action_item['modifier_target'] = parts[1].strip()
                            break
            elif something in init_obs:
                action_item['receptacle'] = something
            else:
                action_item['object'] = something
    
    # Pattern: use (object)
    elif agent_action.startswith('use '):
        obj = agent_action[4:].strip()
        action_item['matched'] = True
        action_item['action'] = 'use'
        action_item['object'] = obj
    
    # Pattern: heat/clean/cool (object) with (receptacle)
    elif any(agent_action.startswith(action) for action in ['heat ', 'clean ', 'cool ']) and ' with ' in agent_action:
        action_prefix = ''
        for prefix in ['heat ', 'clean ', 'cool ']:
            if agent_action.startswith(prefix):
                action_prefix = prefix
                break
                
        parts = agent_action.split(' with ')
        if len(parts) == 2:
            obj = parts[0][len(action_prefix):].strip()
            receptacle = parts[1].strip()
            action_item['matched'] = True
            action_item['action'] = action_prefix.strip()
            action_item['object'] = obj
            action_item['receptacle'] = receptacle
    
    # Pattern: slice (object) with (object)
    elif agent_action.startswith('slice ') and ' with ' in agent_action:
        parts = agent_action.split(' with ')
        if len(parts) == 2:
            obj = parts[0][6:].strip()  # Remove 'slice ' prefix
            obj2 = parts[1].strip()
            action_item['matched'] = True
            action_item['action'] = 'slice'
            action_item['object'] = obj
            action_item['object2'] = obj2  # Using object2 for the tool used for slicing
    
    # Handle alternative "put" syntax which is equivalent to "move to"
    elif agent_action.startswith('put ') and (' in ' in agent_action or ' on ' in agent_action):
        if ' in ' in agent_action:
            parts = agent_action.split(' in ')
        else:
            parts = agent_action.split(' on ')
        
        if len(parts) == 2:
            obj = parts[0][4:].strip()  # Remove 'put ' prefix
            receptacle = parts[1].strip()
            action_item['matched'] = True
            action_item['action'] = 'move to'  # Treat as equivalent to move to
            action_item['object'] = obj
            action_item['receptacle'] = receptacle
    
    # Handle unsupported or malformed actions
    if not action_item['matched']:
        logger.debug(f"Agent action {agent_action} did not match any known pattern")
        return "I don't understand that action. Please use one of the supported actions like 'look', 'go to', 'take from', etc.", False, False

    # Function to check if we should show a surface search hint
    def should_show_surface_hint():
        # Don't show hint if already given
        if WrapStep.hint_given:
            return False
        
        # Don't show hint if no target objects identified
        if not WrapStep.target_objects:
            return False
        
        # Show hint if agent has searched multiple storage locations but no surfaces
        if (len(WrapStep.searched_storage) >= 2 and 
            len(WrapStep.searched_surfaces) == 0):
            return True
        
        # Show hint after multiple failed storage searches
        if WrapStep.failed_storage_searches >= 2:
            return True
        
        return False
    
    # Function to generate surface hint text
    def generate_surface_hint():
        WrapStep.hint_given = True
        
        # Create more specific hint if we know what object the agent is searching for
        if WrapStep.target_objects:
            target_obj = WrapStep.target_objects[0]
            return f"\nHint: You've searched several storage containers but haven't found the {target_obj}. Remember that objects like {target_obj}s can be found not only inside storage containers (drawers, cabinets), but also on surfaces like countertops, tables, and stoveburners. Consider examining these areas as well."
        else:
            return "\nHint: Remember that objects can be found not only inside storage containers like drawers and cabinets, but also on surfaces like countertops, tables, and stoveburners. Consider examining these areas as well."

    # Check if we should add a hint to the current action
    should_add_hint = should_show_surface_hint()
    
    # Check for unsupported "examine [object] in/on [receptacle]" syntax
    if action_item['action'] == 'examine in' and action_item.get('unsupported_syntax', False):
        obj = action_item['object']
        receptacle = action_item['receptacle']
        preposition = "in" if " in " in agent_action else "on"
        logger.debug(f"Agent tried unsupported syntax: examine {obj} {preposition} {receptacle}")
        response = f"The action 'examine {obj} {preposition} {receptacle}' is not supported. To check the contents of {receptacle}, use 'examine {receptacle}'. To examine {obj}, you must have it in your inventory or be at its location."
        
        # Add hint if needed
        if should_add_hint:
            response += generate_surface_hint()
            
        return response, False, False

    # Check if action requires being at a specific location
    requires_proximity = action_item['action'] in ['examine', 'open', 'close', 'take from', 'move to', 'heat', 'clean', 'cool']
    target_receptacle = action_item['receptacle']
    
    # Special handling for "go to" action - update the current location
    if action_item['action'] == 'go to':
        # Check if already at the location
        if WrapStep.current_location == target_receptacle:
            logger.debug(f"Agent is already at {target_receptacle}")
            response = f"You are already at {target_receptacle}."
            
            # Add hint if needed
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, False, False
        
        # Execute the go to action normally
        obs, reward, done, info = env.step([agent_action])
        obs, reward, done = obs[0], info['won'][0], done[0]
        
        # Update the last observation
        WrapStep.last_observation = obs
        
        # If the action succeeded, update the tracked location
        if "nothing happens" in obs.lower():
            logger.debug(f"Agent tried to go to {target_receptacle} but it failed")
            response = f"You cannot go to {target_receptacle}. It might not exist or not be accessible from your current location."
            
            # Add hint if needed
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, reward, done
        
        if reward or ("You arrive at" in obs or "You are now at" in obs):
            logger.debug(f"Agent moved to {target_receptacle}")
            WrapStep.current_location = target_receptacle
            
            # Categorize the receptacle type for search tracking
            if any(storage_type in target_receptacle.lower() for storage_type in ['drawer', 'cabinet', 'fridge', 'refrigerator', 'cupboard', 'closet']):
                WrapStep.searched_storage.add(target_receptacle)
                logger.debug(f"Added {target_receptacle} to searched storage locations. Total: {len(WrapStep.searched_storage)}")
            elif any(surface_type in target_receptacle.lower() for surface_type in ['stoveburner', 'countertop', 'table', 'desk', 'shelf', 'stove', 'sink', 'toaster']):
                WrapStep.searched_surfaces.add(target_receptacle)
                logger.debug(f"Added {target_receptacle} to searched surfaces. Total: {len(WrapStep.searched_surfaces)}")
        
        # Add hint if needed
        if should_add_hint:
            obs += generate_surface_hint()
        
        return obs, reward, done
    
    # For actions requiring proximity, check if agent is at the correct location
    if requires_proximity and target_receptacle:
        if WrapStep.current_location != target_receptacle:
            logger.debug(f"Agent tried to {action_item['action']} {target_receptacle} but is not at that location")
            response = f"You must go to {target_receptacle} before you can {action_item['action']} it."
            
            # Add hint if needed
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, False, False
    
    # Special handling for "examine with" action
    if action_item['action'] == 'examine with':
        obj = action_item['object']
        tool = action_item['object2']
        
        # Check if agent has the object in inventory
        if obj not in WrapStep.inventory:
            logger.debug(f"Agent tried to examine {obj} with {tool} but doesn't have {obj}")
            response = f"You need to have {obj} in your inventory before you can examine it with {tool}."
            
            # Add hint if needed
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, False, False
        
        # Check if agent has the tool in inventory
        if tool not in WrapStep.inventory:
            logger.debug(f"Agent tried to examine {obj} with {tool} but doesn't have {tool}")
            response = f"You need to have {tool} in your inventory before you can use it to examine {obj}."
            
            # Add hint if needed
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, False, False
        
        # If all requirements met, execute the action
        obs, reward, done, info = env.step([agent_action])
        obs, reward, done = obs[0], info['won'][0], done[0]
        
        # Update the last observation
        WrapStep.last_observation = obs
        
        # Provide clearer feedback if the action failed
        if "nothing happens" in obs.lower():
            logger.debug(f"Agent tried to examine {obj} with {tool} but action failed")
            response = f"You cannot examine {obj} with {tool}. This might not be the correct way to use these objects together."
            
            # Add hint if needed
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, reward, done
        
        # Add hint if needed
        if should_add_hint:
            obs += generate_surface_hint()
        
        return obs, reward, done
    
    # Special handling for "examine" action with spatial modifiers
    if action_item['action'] == 'examine' and action_item.get('has_spatial_modifier', False):
        obj = action_item.get('object', '')
        modifier = action_item.get('modifier', '')
        target = action_item.get('modifier_target', '')
        
        logger.debug(f"Agent tried to examine with spatial modifier: {obj} {modifier} {target}")
        response = f"Spatial modifiers like '{modifier}' are not supported. Please examine specific objects directly, such as 'examine {target}' or 'examine {obj}'."
        
        # Add hint if needed
        if should_add_hint:
            response += generate_surface_hint()
            
        return response, False, False
    
    # Special handling for inventory updates
    if action_item['action'] == 'inventory':
        # Execute the step
        obs, reward, done, info = env.step([agent_action])
        obs, reward, done = obs[0], info['won'][0], done[0]
        
        # Update the last observation
        WrapStep.last_observation = obs
        
        # Try to parse inventory from observation
        if "You are carrying:" in obs:
            inventory_section = obs.split("You are carrying:")[1].strip()
            if "nothing" not in inventory_section.lower():
                items = [item.strip() for item in inventory_section.split('\n')]
                WrapStep.inventory = set(items)
            else:
                WrapStep.inventory = set()
        
        # Add hint if needed
        if should_add_hint:
            obs += generate_surface_hint()
        
        return obs, reward, done
    
    # Special handling for "take from" action to provide clearer feedback
    if action_item['action'] == 'take from':
        obj = action_item['object']
        receptacle = action_item['receptacle']
        
        # Check if this is a storage receptacle for tracking purposes
        is_storage = any(storage_type in receptacle.lower() for storage_type in ['drawer', 'cabinet', 'fridge', 'refrigerator', 'cupboard', 'closet'])
        
        # Check if the object is visible on the receptacle (addressing Analysis Result 11)
        object_on_receptacle = False
        if WrapStep.last_observation:
            # Look for patterns like "On the [receptacle], you see a/an [object]"
            on_pattern = f"On the {receptacle}, you see"
            if on_pattern.lower() in WrapStep.last_observation.lower():
                # Get the part after "you see"
                visible_objects_section = WrapStep.last_observation.lower().split(on_pattern.lower())[1]
                # Check if object is mentioned in this section
                if obj.lower() in visible_objects_section:
                    object_on_receptacle = True
                    logger.debug(f"Object {obj} is visible on {receptacle} from observation")
        
        # Record the examined contents if available
        if receptacle in WrapStep.examined_contents and not object_on_receptacle:
            receptacle_contents = WrapStep.examined_contents[receptacle]
            if obj not in receptacle_contents and "Nothing happens" in WrapStep.last_observation:
                logger.debug(f"Agent tried to take {obj} from {receptacle} but it's not there based on examined contents")
                
                # Increment failed storage search counter if applicable
                if is_storage:
                    WrapStep.failed_storage_searches += 1
                    logger.debug(f"Incremented failed storage searches to {WrapStep.failed_storage_searches}")
                
                response = f"There is no {obj} in the {receptacle}."
                
                # Add hint if needed or if we've had multiple failed storage searches
                if should_add_hint:
                    response += generate_surface_hint()
                
                return response, False, False
        
        # Execute the step to see what happens
        obs, reward, done, info = env.step([agent_action])
        obs, reward, done = obs[0], info['won'][0], done[0]
        
        # Update the last observation
        WrapStep.last_observation = obs
        
        # Update inventory if successful 
        if reward or "You pick up" in obs:
            if obj:
                WrapStep.inventory.add(obj)
                logger.debug(f"Agent successfully took {obj} from {receptacle}")
                
                # Reset failed search counter on success
                WrapStep.failed_storage_searches = 0
        
        # If the action failed with "Nothing happens" and the agent is at the right location,
        # provide more informative feedback
        if "nothing happens" in obs.lower() and WrapStep.current_location == receptacle:
            logger.debug(f"Agent tried to take {obj} from {receptacle} but action failed")
            
            # Increment failed storage search counter if applicable
            if is_storage:
                WrapStep.failed_storage_searches += 1
                logger.debug(f"Incremented failed storage searches to {WrapStep.failed_storage_searches}")
            
            # Check if object might be visible on the receptacle but take failed
            if object_on_receptacle:
                response = f"You see {obj} on the {receptacle}, but you can't pick it up for some reason. It might be fixed in place or too heavy."
            else:
                response = f"There is no {obj} in the {receptacle}."
            
            # Add hint if needed or if we've had multiple failed storage searches
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, reward, done
        
        # Add hint if needed
        if should_add_hint:
            obs += generate_surface_hint()
        
        return obs, reward, done
    
    # Special handling for "move to" action to provide clearer feedback
    if action_item['action'] == 'move to':
        # Check if agent has the object
        obj = action_item['object']
        receptacle = action_item['receptacle']
        
        if obj not in WrapStep.inventory:
            logger.debug(f"Agent tried to move {obj} but doesn't have it in inventory")
            response = f"You don't have {obj} in your inventory. You need to take it first."
            
            # Add hint if needed
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, False, False
        
        # Execute the action
        obs, reward, done, info = env.step([agent_action])
        obs, reward, done = obs[0], info['won'][0], done[0]
        
        # Update the last observation
        WrapStep.last_observation = obs
        
        # If successful, update inventory
        if reward or "You put" in obs:
            if obj in WrapStep.inventory:
                WrapStep.inventory.remove(obj)
                logger.debug(f"Agent successfully moved {obj} to {receptacle}")
        
        # Provide clearer feedback for failed move actions
        if "nothing happens" in obs.lower():
            logger.debug(f"Agent tried to move {obj} to {receptacle} but action failed")
            response = f"You cannot put {obj} in/on {receptacle}. This might be because the receptacle cannot hold this type of object."
            
            # Add hint if needed
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, reward, done
        
        # Add hint if needed
        if should_add_hint:
            obs += generate_surface_hint()
        
        return obs, reward, done
    
    # Special handling for "examine" action of objects or receptacles
    if action_item['action'] == 'examine':
        to_examine = action_item.get('object') or action_item.get('receptacle')
        
        if not to_examine:
            logger.debug(f"Agent tried to examine but no object/receptacle specified")
            response = "You need to specify what you want to examine. For example, 'examine fridge'."
            
            # Add hint if needed
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, False, False
        
        # Check if this is a storage receptacle for tracking purposes
        is_storage = any(storage_type in to_examine.lower() for storage_type in ['drawer', 'cabinet', 'fridge', 'refrigerator', 'cupboard', 'closet'])
        
        # Check if this is a surface for tracking purposes
        is_surface = any(surface_type in to_examine.lower() for surface_type in ['stoveburner', 'countertop', 'table', 'desk', 'shelf', 'stove', 'sink', 'toaster'])
        
        # Check if examining object in inventory
        if to_examine in WrapStep.inventory:
            # Allow examination of objects in inventory
            obs, reward, done, info = env.step([agent_action])
            obs, reward, done = obs[0], info['won'][0], done[0]
            
            # Update the last observation
            WrapStep.last_observation = obs
            
            # Add hint if needed
            if should_add_hint:
                obs += generate_surface_hint()
            
            return obs, reward, done
        
        # Check if examining the current receptacle or other objects at the location
        if WrapStep.current_location and (to_examine == WrapStep.current_location or
                                                     (WrapStep.last_observation and to_examine in WrapStep.last_observation)):
            obs, reward, done, info = env.step([agent_action])
            obs, reward, done = obs[0], info['won'][0], done[0]
            
            # Update the last observation
            WrapStep.last_observation = obs
            
            # Store the contents of examined receptacles
            if "contains:" in obs.lower():
                contents_section = obs.split("contains:")[1].strip()
                if "nothing" in contents_section.lower():
                    WrapStep.examined_contents[to_examine] = []
                    # If this is a storage receptacle and it's empty, increment failed search counter
                    if is_storage and WrapStep.target_objects:
                        WrapStep.failed_storage_searches += 1
                        logger.debug(f"Incremented failed storage searches to {WrapStep.failed_storage_searches} - empty {to_examine}")
                else:
                    contents = [item.strip() for item in contents_section.split('\n')]
                    WrapStep.examined_contents[to_examine] = contents
                    logger.debug(f"Stored contents for {to_examine}: {contents}")
                    
                    # Check if any target objects are in this receptacle
                    target_found = any(target in ' '.join(contents).lower() for target in WrapStep.target_objects)
                    if not target_found and is_storage and WrapStep.target_objects:
                        WrapStep.failed_storage_searches += 1
                        logger.debug(f"Incremented failed storage searches to {WrapStep.failed_storage_searches} - {to_examine} doesn't contain target")
            
            if "nothing happens" in obs.lower():
                logger.debug(f"Agent tried to examine {to_examine} but got 'Nothing happens'")
                response = f"You cannot examine {to_examine}. Make sure you are at the correct location and that {to_examine} exists."
                
                # Add hint if needed
                if should_add_hint:
                    response += generate_surface_hint()
                    
                return response, reward, done
            
            # Add hint if needed
            if should_add_hint:
                obs += generate_surface_hint()
            
            return obs, reward, done
        
        # If not at the right location or the object doesn't exist
        logger.debug(f"Agent tried to examine {to_examine} but is not at the right location or the object doesn't exist")
        
        if WrapStep.current_location:
            response = f"You cannot examine {to_examine} from your current location at {WrapStep.current_location}. You may need to go to a different location or check if {to_examine} exists."
            
            # Add hint if needed
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, False, False
        else:
            response = f"You need to go to the location of {to_examine} before you can examine it."
            
            # Add hint if needed
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, False, False
    
    # Special handling for "open" and "close" actions to provide clearer feedback
    if action_item['action'] in ['open', 'close']:
        receptacle = action_item['receptacle']
        
        # Check if this is a storage receptacle for tracking purposes
        is_storage = any(storage_type in receptacle.lower() for storage_type in ['drawer', 'cabinet', 'fridge', 'refrigerator', 'cupboard', 'closet'])
        
        # Execute the action
        obs, reward, done, info = env.step([agent_action])
        obs, reward, done = obs[0], info['won'][0], done[0]
        
        # Update the last observation
        WrapStep.last_observation = obs
        
        # Provide clearer feedback for failed actions
        if "nothing happens" in obs.lower():
            action_word = "open" if action_item['action'] == 'open' else "close"
            logger.debug(f"Agent tried to {action_word} {receptacle} but action failed")
            
            # If opening storage failed, might be worth tracking for hint purposes
            if action_item['action'] == 'open' and is_storage:
                WrapStep.failed_storage_searches += 1
                logger.debug(f"Incremented failed storage searches to {WrapStep.failed_storage_searches} - failed to open {receptacle}")
            
            # Check possible reasons and provide specific feedback
            if action_item['action'] == 'open':
                response = f"You cannot open {receptacle}. It might be already open, not openable, or doesn't exist at your current location."
            else:  # close
                response = f"You cannot close {receptacle}. It might be already closed, not closable, or doesn't exist at your current location."
            
            # Add hint if needed
            if should_add_hint:
                response += generate_surface_hint()
                
            return response, reward, done
        
        # Add hint if needed
        if should_add_hint:
            obs += generate_surface_hint()
        
        return obs, reward, done
    
    # Special handling for "look" action to provide hints about object locations
    if action_item['action'] == 'look':
        # Execute the step
        obs, reward, done, info = env.step([agent_action])
        obs, reward, done = obs[0], info['won'][0], done[0]
        
        # Update the last observation
        WrapStep.last_observation = obs
        
        # Add hint if needed
        if should_add_hint:
            obs += generate_surface_hint()
        
        return obs, reward, done
    
    # Attempt to execute other actions with the environment
    obs, reward, done, info = env.step([agent_action])
    obs, reward, done = obs[0], info['won'][0], done[0]
    
    # Update the last observation
    WrapStep.last_observation = obs
    
    # Provide more informative feedback for generic failures
    if "nothing happens" in obs.lower():
        logger.debug(f"Agent action {agent_action} resulted in 'Nothing happens'")
        
        # Try to construct a more informative error message based on the action
        if action_item['action'] in ['heat', 'clean', 'cool']:
            obj = action_item['object']
            receptacle = action_item['receptacle']
            response = f"You cannot {action_item['action']} {obj} with {receptacle}. Make sure both the object and receptacle exist, you possess {obj}, and this is a valid interaction."
        elif action_item['action'] == 'slice':
            obj = action_item['object']
            tool = action_item['object2']
            response = f"You cannot slice {obj} with {tool}. Make sure both objects exist, you possess them, and this is a valid interaction."
        elif action_item['action'] == 'use':
            obj = action_item['object']
            response = f"You cannot use {obj}. Make sure you possess it and it is usable in the current context."
        else:
            response = f"The action '{agent_action}' did not have any effect. Check that all objects and receptacles exist and that this is a valid interaction."
        
        # Add hint if needed
        if should_add_hint:
            response += generate_surface_hint()
            
        return response, reward, done
    
    # Add hint if needed
    if should_add_hint:
        obs += generate_surface_hint()
    
    return obs, reward, done