import re

def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic for different task types.
    """
    return """
    Navigation Rules:
    - You can only move to locations that are directly connected to your current room
    - Connected locations will be mentioned in the observation as doors or pathways (e.g., "You also see: A door to the kitchen (that is open)")
    - If you try to go to a location that isn't directly connected to your current room, the action will fail
    - To reach a location that isn't directly connected, you must first go to intermediate rooms

    Object Interaction Rules:
    - You can only interact with objects that are present in your current room
    - Different objects support different actions (e.g., containers can be opened but furniture might not)
    - If an object is mentioned in a previous room but not your current room, you must first go to that room
    - To interact with objects inside containers, you must first open the container
    - When referring to objects, use the simplest form that uniquely identifies them (e.g., "open furnace" instead of "open furnace door")
    - Do not include location specifiers in your actions (e.g., use "look at table" when in the kitchen, not "look at table in kitchen")
    - Items in your inventory can be put down in any room using "put down OBJ"
    
    Task Completion:
    - Some tasks may require exploring multiple rooms to find necessary objects
    - Pay attention to environment feedback which may guide you toward the right locations or actions
    - If you're having trouble finding specific objects (like containers for planting or water sources), try exploring different rooms
    - Remember that specialized objects (like flower pots or laboratory equipment) might be found in specialized rooms (like greenhouses or laboratories)
    """

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action and return the next observation, done status and score(returned by the environment).
    """
    original_action = agent_action
    agent_action = agent_action.strip().lower()
    action_parts = agent_action.split()
    
    logger.debug(f"Processing action: '{agent_action}'")
    
    # Check inventory first for put down actions
    is_put_down_action = agent_action.startswith("put down ")
    if is_put_down_action:
        inventory_obs, _, _, _ = env.step("inventory")
        logger.debug(f"Checking inventory for put down action: {inventory_obs}")
        
        # Extract object to put down
        object_to_put_down = agent_action[9:].strip()
        
        # Remove content in parentheses
        clean_obj_to_put_down = re.sub(r'\s*$$.*?$$', '', object_to_put_down).strip()
        
        # Remove articles for better matching
        if clean_obj_to_put_down.startswith("a "):
            clean_obj_to_put_down = clean_obj_to_put_down[2:]
        elif clean_obj_to_put_down.startswith("an "):
            clean_obj_to_put_down = clean_obj_to_put_down[3:]
        elif clean_obj_to_put_down.startswith("the "):
            clean_obj_to_put_down = clean_obj_to_put_down[4:]
        
        logger.debug(f"Cleaned object to put down: '{clean_obj_to_put_down}'")
        
        # Check if object is in inventory
        object_in_inventory = False
        matched_item = ""
        inventory_items = []
        
        # Support multiple inventory prefix formats
        inventory_patterns = [
            r"You're carrying:(.*?)(?=\n\n|$)",
            r"In your inventory, you see:(.*?)(?=\n\n|$)",
            r"Your inventory contains:(.*?)(?=\n\n|$)",
            r"You are carrying:(.*?)(?=\n\n|$)"
        ]
        
        for pattern in inventory_patterns:
            inventory_match = re.search(pattern, inventory_obs, re.DOTALL)
            if inventory_match:
                inventory_section = inventory_match.group(1).strip()
                raw_items = [item.strip() for item in inventory_section.split("\n") if item.strip()]
                
                # Process each item: remove leading symbols and clean
                for raw_item in raw_items:
                    # Remove leading symbols like tabs, -, *, etc.
                    clean_raw_item = re.sub(r'^\s*[-*\t ]+\s*', '', raw_item)
                    inventory_items.append(clean_raw_item)
                
                logger.debug(f"Inventory items detected using pattern '{pattern}': {inventory_items}")
                break
        
        if not inventory_items and "nothing" not in inventory_obs.lower():
            # Fallback: just use the whole observation if we couldn't parse it
            inventory_items = [line.strip() for line in inventory_obs.split("\n") 
                              if line.strip() and "inventory" not in line.lower()]
            logger.debug(f"Using fallback inventory parsing: {inventory_items}")
        
        for item in inventory_items:
            item_lower = item.lower()
            
            # Clean inventory item - remove content in parentheses
            clean_item = re.sub(r'\s*$$.*?$$', '', item_lower).strip()
            
            # Remove articles from inventory item
            if clean_item.startswith("a "):
                clean_item = clean_item[2:]
            elif clean_item.startswith("an "):
                clean_item = clean_item[3:]
            elif clean_item.startswith("the "):
                clean_item = clean_item[4:]
            
            logger.debug(f"Comparing: '{clean_obj_to_put_down}' with cleaned inventory item '{clean_item}'")
            
            # Multiple matching strategies with more permissive matching
            if (clean_obj_to_put_down == clean_item or  # Exact match
                clean_obj_to_put_down in clean_item or  # Object is part of inventory item
                clean_item in clean_obj_to_put_down or  # Inventory item is part of object
                clean_item.startswith(clean_obj_to_put_down) or  # Object is prefix of inventory item
                clean_obj_to_put_down.startswith(clean_item) or  # Inventory item is prefix of object
                clean_item.endswith(clean_obj_to_put_down) or  # Object is suffix of inventory item
                clean_obj_to_put_down.endswith(clean_item)):  # Inventory item is suffix of object
                
                object_in_inventory = True
                matched_item = item
                logger.debug(f"Match found! '{clean_obj_to_put_down}' matches with inventory item '{clean_item}'")
                break
        
        # If object is in inventory, proceed with action
        if object_in_inventory:
            logger.debug(f"Object '{object_to_put_down}' is in inventory as '{matched_item}', proceeding with put down action")
            obs, _, done, info = env.step(original_action)
            return obs, done, info["score"]
        else:
            # If object is not in inventory, provide informative feedback
            custom_obs = f"You cannot put down '{object_to_put_down}' because it is not in your inventory. Check your inventory with the 'inventory' command to see what you're carrying."
            logger.debug(f"Object '{object_to_put_down}' is not in inventory, returning custom feedback")
            return custom_obs, False, 0.0
    
    # Get current room information for object availability checks
    room_info_obs, _, _, _ = env.step("look around")
    logger.debug(f"Room info observation received")
    
    # Improved extraction of current room name with multiple patterns
    current_room = "unknown room"  # Default fallback
    # Pattern 1: "You are in the X"
    room_match = re.search(r"You are in (?:the )?([^\.]+)", room_info_obs)
    if room_match:
        current_room = room_match.group(1).strip()
        logger.debug(f"Current room identified (pattern 1): '{current_room}'")
    
    # Pattern 2: "This room is called the X"
    if current_room == "unknown room":
        room_match = re.search(r"This room is called (?:the )?([^\.]+)", room_info_obs)
        if room_match:
            current_room = room_match.group(1).strip()
            logger.debug(f"Current room identified (pattern 2): '{current_room}'")
    
    # Pattern 3: "You're in the X"
    if current_room == "unknown room":
        room_match = re.search(r"You're in (?:the )?([^\.]+)", room_info_obs)
        if room_match:
            current_room = room_match.group(1).strip()
            logger.debug(f"Current room identified (pattern 3): '{current_room}'")

    # Pattern 4: First line analysis
    if current_room == "unknown room":
        first_line = room_info_obs.split("\n") if "\n" in room_info_obs else room_info_obs
        if "." in first_line:
            first_sentence = first_line.split(".")
            if "You are" in first_sentence:
                room_parts = first_sentence.split("You are")
                if len(room_parts) > 1:
                    room_text = room_parts[1].strip()
                    if room_text.startswith("in the "):
                        current_room = room_text[7:].strip()
                    elif room_text.startswith("in "):
                        current_room = room_text[3:].strip()
                    logger.debug(f"Current room identified (pattern 4): '{current_room}'")
    
    # Function to recursively extract objects from text with improved parsing
    def extract_objects_recursive(text):
        objects = []
        
        # Extract top-level objects using multiple patterns
        # Pattern 1: "You see: X"
        top_level_pattern1 = r"You (?:see|also see):(.*?)(?=\n\n|$|You also see:|In it, you see:)"
        top_matches1 = re.findall(top_level_pattern1, text, re.DOTALL)
        
        # Pattern 2: "In it, you see: X"
        top_level_pattern2 = r"In it, you see:(.*?)(?=\n\n|$|You also see:)"
        top_matches2 = re.findall(top_level_pattern2, text, re.DOTALL)
        
        # Combine matches from both patterns
        top_matches = top_matches1 + top_matches2
        
        for match in top_matches:
            lines = match.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and "door to" not in line.lower():
                    # Clean up the object text
                    clean_obj = re.sub(r'\s*$$.*?$$', '', line).strip()
                    if clean_obj:
                        objects.append(clean_obj.lower())
        
        # Extract objects in/on containers with improved pattern
        container_pattern = r"(?:In|On) (?:the )?([^:]+)(?:,| you| is| are| that).*?(?:You see|There is|There are|you see):(.*?)(?=(?:In|On) (?:the )|$|\n\n)"
        container_matches = re.findall(container_pattern, text, re.DOTALL)
        
        for container, contents in container_matches:
            container = container.strip().lower()
            objects.append(container)  # Add the container itself
            
            # Add contents of the container
            lines = contents.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line:
                    # Clean up the object text
                    clean_obj = re.sub(r'\s*$$.*?$$', '', line).strip()
                    if clean_obj:
                        # Add the full object with container context
                        full_obj = f"{clean_obj.lower()} in {container}"
                        objects.append(full_obj)
                        # Also add the object itself
                        objects.append(clean_obj.lower())
        
        # Add variations without articles
        result = []
        for obj in objects:
            result.append(obj)
            if obj.startswith("a "):
                result.append(obj[2:].strip())
            elif obj.startswith("an "):
                result.append(obj[3:].strip())
            elif obj.startswith("the "):
                result.append(obj[4:].strip())
            
            # Add variations for compound objects
            parts = obj.split()
            if len(parts) > 1:
                # Add last word (e.g., "drawer" from "desk drawer")
                result.append(parts[-1])
                # Add combinations of adjacent words
                for i in range(len(parts) - 1):
                    result.append(f"{parts[i]} {parts[i+1]}")
        
        return list(set(result))  # Remove duplicates
    
    # Extract available objects in the current room with improved recursive parsing
    available_objects = extract_objects_recursive(room_info_obs)
    logger.debug(f"Available objects identified: {available_objects}")
    
    # Extract valid exits from the current room
    valid_exits = []
    for line in room_info_obs.split("\n"):
        if "door to" in line.lower():
            # Extract location names from text like "A door to the kitchen (that is open)"
            location_match = re.search(r'door to (?:the )?([^(]+)', line.lower())
            if location_match:
                location = location_match.group(1).strip()
                valid_exits.append(location)
            else:
                # Fallback method
                location_start = line.lower().find("door to") + 8
                location_end = line.lower().find("(", location_start) if "(" in line[location_start:] else len(line)
                location = line[location_start:location_end].strip()
                if location.startswith("the "):
                    location = location[4:]  # Remove "the " prefix
                valid_exits.append(location)
    
    logger.debug(f"Valid exits identified: {valid_exits}")
    
    # Create a list of known locations (valid exits + current room)
    known_locations = valid_exits.copy()
    if current_room != "unknown room":
        known_locations.append(current_room.lower())
    
    # Common room names to help with location specifier detection
    common_rooms = ["kitchen", "bathroom", "bedroom", "living room", "dining room", 
                   "office", "hallway", "corridor", "garden", "garage", "basement", 
                   "attic", "study", "patio", "balcony", "porch", "foyer", "entrance",
                   "greenhouse", "laboratory", "lab", "workshop", "library", "backyard"]
    
    # Add common rooms to known locations for better detection
    for room in common_rooms:
        if room not in known_locations:
            known_locations.append(room)
    
    # Check for water-related interactions in the kitchen when task involves water
    is_water_related_task = "water" in task.lower() or "liquid" in task.lower() or "pour" in task.lower()
    is_in_kitchen = "kitchen" in current_room.lower()
    is_water_interaction = "water" in agent_action.lower()
    is_sink_interaction = "sink" in agent_action.lower() and "look" in agent_action.lower()
    
    # Check if water is actually present in the room
    water_present = False
    for obj in available_objects:
        if "water" in obj.lower():
            water_present = True
            break
    
    # Provide specific guidance for water-related interactions in the kitchen
    if is_water_related_task and is_in_kitchen and (is_water_interaction or is_sink_interaction) and not water_present:
        custom_obs = "There does not appear to be any water in the kitchen. You may need to explore other rooms, such as the greenhouse, to find or generate water. Available exits from here: " + ", ".join(valid_exits)
        logger.debug(f"Providing specific water-related guidance for kitchen")
        return custom_obs, False, 0.0
    
    # Handle navigation commands ("go to LOC")
    if len(action_parts) >= 3 and action_parts == "go" and action_parts[1] == "to":
        target_location = " ".join(action_parts[2:])
        logger.debug(f"Navigation command detected: go to {target_location}")
        
        # Check if target location is in valid exits
        location_found = False
        for exit in valid_exits:
            if target_location in exit or exit in target_location:
                location_found = True
                logger.debug(f"Target location '{target_location}' matched with valid exit '{exit}'")
                break
        
        if not location_found and valid_exits:
            # Provide informative feedback about available locations
            available_locations = ", ".join(valid_exits)
            
            # Check if destination exists but is not directly connected
            possible_path = False
            for exit in valid_exits:
                # Check for at least one exit to explore further
                if exit not in [current_room.lower()]:
                    possible_path = True
                    break
            
            if possible_path:
                custom_obs = f"You cannot go directly to '{target_location}' from here. Available exits: {available_locations}. Try going to one of these locations first."
                logger.debug(f"Target location '{target_location}' is not directly connected, suggesting intermediate steps")
                return custom_obs, False, 0.0
            else:
                custom_obs = f"You are currently in '{current_room}' and there are no exits from this room. This may be a bug or a limitation of the environment."
                logger.debug(f"No exits found from current room, suggesting environment limitation")
                return custom_obs, False, 0.0
    
    # Special case for "open X door" -> try "open X" if it fails
    if agent_action.startswith("open ") and agent_action.endswith(" door"):
        base_object = agent_action[5:-5].strip()  # Remove "open " and " door"
        logger.debug(f"Special case: 'open X door' detected, will try 'open {base_object}' as fallback")
        
        # First try the original "open X door"
        obs, _, done, info = env.step(original_action)
        
        # If it fails with "No known action", try "open X" instead
        if "No known action matches that input" in obs:
            logger.debug(f"Original action failed, trying alternative: 'open {base_object}'")
            alt_action = f"open {base_object}"
            alt_obs, _, alt_done, alt_info = env.step(alt_action)
            
            # If the alternative succeeds, provide helpful feedback
            if "No known action matches that input" not in alt_obs:
                enhanced_obs = f"I opened the {base_object}. Note: In this environment, use 'open {base_object}' rather than 'open {base_object} door'.\n\n{alt_obs}"
                logger.debug(f"Alternative action succeeded, returning enhanced obs")
                return enhanced_obs, alt_done, alt_info["score"]
            
            # If both fail, provide enhanced feedback with all available objects
            object_list = ", ".join(available_objects) if available_objects else "none"
            enhanced_obs = f"That action couldn't be performed. This could be because:\n- The object doesn't support this action\n- The action format is incorrect\n- The object exists but with a slightly different name\n\nAvailable objects in this room: {object_list}\n\nTry a different action or examine the room again with 'look around'."
            logger.debug(f"Both original and alternative actions failed, returning enhanced feedback")
            return enhanced_obs, False, info["score"]
    
    # Process the action through the environment
    logger.debug(f"Executing action through environment: '{original_action}'")
    obs, _, done, info = env.step(original_action)
    
    # Check if the action failed with a generic message and provide better feedback
    if "No known action matches that input" in obs:
        logger.debug(f"Generic error message received: '{obs}'")
        
        # Improve feedback for navigation errors
        if agent_action.startswith("go "):
            direction = agent_action[3:].strip()
            custom_obs = f"You can't go {direction} from here. Available exits: {', '.join(valid_exits)}. Try using 'go to [location]' with one of these locations."
            return custom_obs, False, info["score"]
        
        # Improve feedback for object interaction errors
        if len(action_parts) >= 2:
            action_type = action_parts
            object_name = " ".join(action_parts[1:])
            
            if action_type in ["take", "get", "pick", "look", "examine", "open", "close", "use", "push", "pull"]:
                # Format the available objects for better readability
                formatted_objects = []
                for obj in sorted(set(available_objects)):
                    # Skip very long object descriptions and duplicates
                    if len(obj) < 50 and obj not in formatted_objects:
                        formatted_objects.append(obj)
                
                object_list = ", ".join(formatted_objects) if formatted_objects else "none"
                custom_obs = f"You cannot {action_type} '{object_name}' because it doesn't appear to be present in this room ({current_room}). Available objects here: {object_list}. You may need to explore other rooms to find this object, or the object might be inside a container that needs to be opened first."
                return custom_obs, False, info["score"]
    
    return obs, done, info["score"]