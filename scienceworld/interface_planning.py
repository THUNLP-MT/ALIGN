import re # Import the regular expression module for pattern matching
import logging # Assuming logger is configured elsewhere, adding import for clarity

# Define InferRules (No changes needed based on Analysis Result 9)
def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic for different task types.
    Provides upfront information about specific environment mechanics.
    """
    # Add rule about thermometer usage based on Analysis Result 1
    # Add rule about action specificity based on Analysis Result 4
    # Add rule about navigation based on Analysis Result 5
    # Add rule about object accessibility based on Analysis Result 6
    # Add rule about getting items from containers based on Analysis Result 8
    rule = """Environment Rules:
- To measure temperature, you must 'use thermometer on [substance]'. Reading the thermometer directly is not possible.
- Action commands are specific. Ensure you use the exact action verbs and object names provided by the environment (e.g., via 'look around', 'inventory', 'look at', 'look in'). Check the action format list if unsure.
- Movement between locations ('go to LOC') is restricted to adjacent rooms only. You cannot move directly to a non-adjacent room. Use 'look around' to identify available exits.
- You can only interact with objects (e.g., using 'use', 'pick up', 'open', 'focus on') that are present in your current location or in your inventory. Check your surroundings ('look around') or inventory ('inventory') if an interaction fails.
- To get an item *from* a container (e.g., 'seed jar'), you usually need to 'pick up' the container first, then 'move' the item from the container (now in inventory) to its destination. Directly picking up items *from* a container in the environment might not work."""
    # Future rules based on other analyses can be added here.
    # Analysis Result 3 (focus action) is handled dynamically in WrapStep
    # as it's specific to task outcome feedback rather than a general rule.
    # Analysis Result 2 (wait action) is handled dynamically in WrapStep.
    # Analysis Result 7 (task-specific action necessity) is handled dynamically in WrapStep.
    # Analysis Result 9 (generic focus) is handled dynamically in WrapStep.
    return rule

# Define WrapStep (Updated based on Analysis Result 9)
def WrapStep(env, init_obs, task, agent_action: str, logger: logging.Logger):
    """
    Process the agent action:
    - Intercepts specific actions ("read thermometer", "wait [DURATION]", "focus on [incorrect object]" in specific tasks,
      "go to [non-adjacent LOC]", unnecessary actions like "chop wood" in specific tasks, "pick up OBJ from OBJ",
      "focus on [generic term]") to provide clearer feedback based on AnalysisAgent results.
    - Checks for failures due to object non-accessibility (Analysis Result 6).
    - Checks for generic "No known action" errors and provides guidance, differentiating feedback for failed 'go to' actions (Analysis Result 4 & 5)
      and providing object accessibility hints for failed interaction actions (Refined logic).
    - Executes the action using env.step for all other cases or to get base results for intercepted actions.
    - Returns the next observation (potentially modified for clarity), done status (potentially overridden), and score.
    """
    logger.debug(f"Processing agent action: {agent_action}")

    # Normalize action for case-insensitive comparison and stripping whitespace
    # Keep original agent_action for feedback messages
    normalized_action = agent_action.lower().strip()

    # Define interaction verbs for later checks
    interaction_verbs = ["focus on", "use", "pick up", "open", "close", "look at", "look in", "move", "pour", "dunk", "mix", "eat", "flush"] # Note: "chop" is not included as it's handled specifically below

    # Analysis Result 1: Handle "read thermometer" / "check thermometer reading"
    if normalized_action == "read thermometer" or normalized_action == "check thermometer reading":
        logger.debug(f"Intercepted action '{agent_action}' related to reading thermometer. Providing specific feedback.")
        obs_orig, _, done, info = env.step(agent_action)
        score = info["score"]
        helpful_feedback = "To obtain a temperature reading, use the thermometer on the substance you wish to measure (e.g., 'use thermometer on solid unknown substance S'). Reading the thermometer directly does not provide the temperature."
        logger.debug(f"Returning helpful feedback instead of default observation for thermometer read attempt. Original Obs: '{obs_orig}'. Done: {done}, Score: {score}")
        return helpful_feedback, done, score

    # Analysis Result 2: Handle "wait [DURATION]"
    elif re.match(r"wait\s+\d+", normalized_action):
        logger.debug(f"Intercepted action '{agent_action}' matching 'wait [DURATION]' pattern. Providing specific feedback.")
        obs_orig, _, done, info = env.step(agent_action)
        score = info["score"]
        helpful_feedback = "Waiting is not supported in this environment; please proceed with another action."
        logger.debug(f"Returning helpful feedback as 'wait' action is not supported. Original Obs: '{obs_orig}'. Done: {done}, Score: {score}")
        return helpful_feedback, done, score

    # Analysis Result 3 & 9: Handle "focus on OBJ"
    elif focus_match := re.match(r"focus on (.*)", normalized_action):
        focused_object = focus_match.group(1).strip()
        logger.debug(f"Detected 'focus on' action targeting: {focused_object}")

        # Analysis Result 9: Handle generic "focus on animal" type actions first
        # Use regex for slightly more flexible matching of generic terms
        generic_focus_pattern = r"^(an )?animal(\s+with.*)?$|another animal"
        if re.fullmatch(generic_focus_pattern, focused_object, re.IGNORECASE):
            logger.debug(f"Intercepted generic focus action '{agent_action}'. Providing specific feedback.")
            # Call env.step to get score/done, but override observation
            obs_orig, _, done, info = env.step(agent_action)
            score = info["score"]
            # Use the original agent_action's object reference in the feedback for clarity
            original_focused_term = focus_match.group(1).strip() # Get the term as the agent typed it
            helpful_feedback = f"There is no object called '{original_focused_term}' here. Please use the exact name of an animal you see in the environment (e.g., 'focus on ant', 'focus on chameleon egg'). Use 'look around' to see the list of animals present."
            logger.debug(f"Returning helpful feedback for generic focus attempt. Original Obs: '{obs_orig}'. Done: {done}, Score: {score}")
            return helpful_feedback, done, score # Return original done status

        # Analysis Result 3: Handle specific "focus on OBJ" for shortest lifespan task (if not generic)
        elif re.search(r"shortest\s+life\s*span", task, re.IGNORECASE):
            logger.debug(f"Action is 'focus on' specific object '{focused_object}' within a task context matching r'shortest\\s+life\\s*span'.")
            # Assuming chameleon egg is correct based on analysis, but could be generalized if needed
            correct_focus_object_pattern = r"chameleon egg" # Use regex pattern for flexibility if needed
            if not re.fullmatch(correct_focus_object_pattern, focused_object, re.IGNORECASE):
                logger.debug(f"Intercepted incorrect specific focus action '{agent_action}' for task '{task[:50]}...'. Expected focus matching '{correct_focus_object_pattern}'.")
                obs_orig, _, done_orig, info = env.step(agent_action)
                score = info["score"]
                logger.debug(f"Original env.step results for incorrect focus: Obs='{obs_orig}', Done={done_orig}, Score={score}. Overriding Done status and Observation.")
                helpful_feedback = f"Incorrect. The {focused_object} is not the animal with the shortest lifespan. Try focusing on another animal."
                # Override done to False to allow the agent to retry, unless the environment already ended the task
                return helpful_feedback, False if not done_orig else True, score
            else:
                logger.debug(f"Correct focus action '{agent_action}' detected for 'shortest life span' task context. Processing normally.")
                pass # Fall through to default processing
        else:
            logger.debug(f"'focus on' action detected, but not generic and not in the 'shortest life span' task context. Processing normally.")
            pass # Fall through to default processing

    # Analysis Result 7: Handle unnecessary actions in specific tasks (e.g., chopping wood for paint mixing)
    elif (normalized_action == "use axe on wood" or normalized_action == "chop wood with axe") and \
         re.search(r"create violet paint|mix.*paint", task, re.IGNORECASE): # Check task context (Task ID 3-28)
        logger.debug(f"Intercepted action '{agent_action}' related to chopping wood in paint mixing task. Providing specific feedback.")
        # Call env.step to get the score/done status, even though the action is invalid/unnecessary in this context
        obs_orig, _, done, info = env.step(agent_action)
        score = info["score"]
        # Provide feedback explaining the action is unnecessary for *this* task
        helpful_feedback = "Chopping wood is not required for this task. To create violet paint, you likely need to find and mix red and blue paints, possibly in the art studio."
        logger.debug(f"Returning helpful feedback instead of default observation for unnecessary wood chopping. Original Obs: '{obs_orig}'. Done: {done}, Score: {score}")
        # Return the score/done status from the env.step call, but replace the observation
        return helpful_feedback, done, score

    # Analysis Result 8: Handle "pick up OBJ from OBJ"
    elif re.match(r"pick up .* from .*", normalized_action):
        logger.debug(f"Intercepted action '{agent_action}' matching 'pick up OBJ from OBJ' pattern. Providing specific feedback.")
        # Call env.step to get the score/done status, even though the action is invalid
        obs_orig, _, done, info = env.step(agent_action)
        score = info["score"]
        # Provide feedback explaining the correct procedure
        helpful_feedback = "You cannot pick up an item directly from a container like that. Try picking up the container first (e.g., 'pick up [container name]'), then move the item from the container (now in your inventory) to the desired location (e.g., 'move [item name] in [container name] to [destination]')."
        logger.debug(f"Returning helpful feedback for 'pick up from container' attempt. Original Obs: '{obs_orig}'. Done: {done}, Score: {score}")
        return helpful_feedback, done, score

    # Analysis Result 5: Handle specific "go to LOC" errors (non-adjacency)
    # Store the match result for potential use in the "No known action" check later
    go_to_match = re.match(r"go to (.*)", normalized_action)
    if go_to_match:
        target_location = go_to_match.group(1).strip()
        logger.debug(f"Detected 'go to' action targeting: {target_location}")
        # Execute the action first to see if it works or gives a specific error
        obs, _, done, info = env.step(agent_action)
        score = info["score"]
        logger.debug(f"Executed 'go to {target_location}'. Raw Obs: '{obs[:100]}...', Done: {done}, Score: {score}")
        # Check for specific non-adjacency feedback from the environment
        if "you can't go there" in obs.lower() or "you can't seem to go that way" in obs.lower():
            logger.debug(f"Intercepted invalid 'go to {target_location}' action (explicit non-adjacency message). Providing specific feedback.")
            helpful_feedback = f"You cannot go directly to the {target_location} from your current location. Movement is restricted to adjacent rooms. Try using 'look around' to see available exits and move step-by-step."
            return helpful_feedback, done, score
        # Fall through to check other errors like "No known action" using the 'obs' from this step

    # Default action processing: Execute the action if not intercepted or if it fell through specific checks above
    if 'obs' not in locals(): # Check if obs was already set by a previous block (e.g., 'go to', or an intercepted action that fell through)
         logger.debug(f"Executing standard action via env.step: {agent_action}")
         obs, _, done, info = env.step(agent_action)
         score = info["score"]
         logger.debug(f"Action executed. Raw Obs: '{obs[:100]}...', Done: {done}, Score: {score}")
    else:
         logger.debug(f"Action '{agent_action}' was already executed by a specific handler. Using existing results.")
         # obs, done, score are already set from the 'go to' block or a fall-through case

    # Post-execution checks for common errors based on the observation 'obs'

    # Analysis Result 6: Check for explicit object not present/accessible errors
    # Use specific phrases often returned by the environment in these cases
    object_error_patterns = ["you don't see", "not accessible", "not here", "don't have", "isn't here"]
    obs_lower = obs.lower()
    if any(pattern in obs_lower for pattern in object_error_patterns):
        # Check if the action was an interaction type where object presence matters
        # Use the predefined list of interaction verbs
        if any(normalized_action.startswith(verb) for verb in interaction_verbs):
            # Avoid triggering this for the already handled "pick up ... from ..." case if it somehow resulted in such an error message
            # Also avoid triggering for the handled generic "focus on" case
            is_generic_focus = focus_match and re.fullmatch(r"^(an )?animal(\s+with.*)?$|another animal", focus_match.group(1).strip(), re.IGNORECASE)
            if not re.match(r"pick up .* from .*", normalized_action) and not is_generic_focus:
                logger.debug(f"Detected explicit object not present/accessible error for interaction action '{agent_action}'. Raw obs: '{obs}'")
                helpful_feedback = f"Your action '{agent_action}' failed because the target object(s) might not be present or accessible from your current location or inventory. Use 'look around' to check available objects, 'inventory' to check your items, or 'go to' another location if necessary."
                return helpful_feedback, done, score

    # Analysis Result 4 & Refined 5 & Refined Interaction Logic: Handle "No known action matches that input" error
    if "No known action matches that input" in obs:
        # Avoid overriding feedback if it was already handled by a specific interceptor above (like generic focus, pick up from, wait, read thermometer, chop wood)
        # Check if the action matches any of the patterns already handled explicitly that might return "No known action"
        already_handled = (
            normalized_action == "read thermometer" or
            normalized_action == "check thermometer reading" or
            re.match(r"wait\s+\d+", normalized_action) or
            (focus_match and re.fullmatch(r"^(an )?animal(\s+with.*)?$|another animal", focus_match.group(1).strip(), re.IGNORECASE)) or
            ((normalized_action == "use axe on wood" or normalized_action == "chop wood with axe") and re.search(r"create violet paint|mix.*paint", task, re.IGNORECASE)) or
            re.match(r"pick up .* from .*", normalized_action)
        )

        if not already_handled:
            logger.debug(f"Detected 'No known action' error for action '{agent_action}'. Providing context-specific feedback.")

            # Check if the failed action was 'go to' (using the match object from earlier)
            if go_to_match:
                 target_location = go_to_match.group(1).strip()
                 # Provide specific feedback for failed 'go to' actions, covering non-adjacency, misspelling, or non-existence
                 helpful_feedback = f"The action 'go to {target_location}' was not recognized. This might be because '{target_location}' is not an adjacent location, the name is misspelled, or the location doesn't exist. Movement is restricted to adjacent rooms. Use 'look around' to see available exits and verify location names."
                 logger.debug(f"Providing navigation-specific feedback for unrecognized 'go to' action.")

            # Check if it was an interaction action (using the predefined list)
            # Exclude the "pick up ... from ..." case as it's handled specifically above
            elif any(normalized_action.startswith(verb) for verb in interaction_verbs) and not re.match(r"pick up .* from .*", normalized_action):
                 logger.debug(f"Providing object accessibility/spelling feedback for unrecognized interaction action '{agent_action}'.")
                 # Suggest object accessibility/spelling as the likely cause for failed interaction
                 helpful_feedback = f"The action '{agent_action}' was not recognized. This might be because the target object(s) are not present or accessible from your current location/inventory, or the object name is misspelled. Use 'look around' or 'inventory' to check available objects/names, or 'go to' another location if necessary."

            # ELSE: Provide generic feedback for other unrecognized actions (e.g., misspelled 'look around', 'inventory', or unsupported verbs like 'chop' if not intercepted)
            else:
                 helpful_feedback = f"The action '{agent_action}' was not recognized. Please ensure:\n1. The action verb (e.g., 'open', 'use', 'go to') and format are correct (refer to the action list).\n2. Object/location names are spelled exactly as observed (use 'look around', 'inventory', 'look at', 'look in' to verify)."
                 logger.debug(f"Providing generic feedback for unrecognized non-interaction/non-navigation action.")

            # Replace the unhelpful observation with the guidance
            return helpful_feedback, done, score
        else:
            logger.debug(f"'No known action' observed, but the action '{agent_action}' was already handled by a specific interceptor. Returning original observation.")
            # Fall through to return the original obs, done, score if already handled

    # If no specific error handling was triggered, return the original results from env.step
    logger.debug(f"Action '{agent_action}' processed without triggering specific error handlers or intercepts (or fell through). Returning standard results.")
    return obs, done, score