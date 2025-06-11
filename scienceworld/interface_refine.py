import re
import logging
# Assume logger is configured elsewhere and passed to WrapStep
# Example configuration (not part of the required output):
# import sys
# import io
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logger = logging.getLogger(__name__)
# log_stream = io.StringIO() # Example stream for logger if needed by template

def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic for different task types.
    Refined based on Analysis Result 1, 2, 3, 4, 5, and 6.
    """
    # Rule 3 updated based on Analysis Result 6
    rule = """General Environment Rules:
1. Object Accessibility: Most actions require the target object(s) to be present in your current location or inventory. If an object is not accessible, you may need to 'go to' its location or 'pick up' the object first. Actions like 'focus on', 'look in', or 'open' specifically require the object to be accessible.
2. Action Validity: Ensure your actions match the available commands and target valid objects or locations. Check spelling carefully.
3. Navigation ('go to'): To move between locations, use 'go to [location name]'. Ensure the location name is correct and that it is directly accessible (e.g., through an open door) from your current location. Check the room description for available exits. 'go to' only works for navigating between distinct locations listed as exits; it cannot be used to navigate to specific objects or areas within your current location (e.g., 'go to table', 'go to plants'). Use 'look around' to see objects in your current location.
4. Object Referencing: When referring to objects in actions (e.g., 'pick up', 'use', 'look at'), use the object's canonical name as listed in the environment descriptions (e.g., 'red bottle', 'metal pot'). Do not include location specifiers like 'on the table' or 'in the cupboard' as part of the object name in your command."""
    return rule

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action, check for specific failures like 'focus on' non-present object,
    'go to' failures with generic or ambiguous messages (including attempts to navigate to non-locations),
    'pick up' failures potentially due to location specifiers, or 'look in'/'open' failures due to object absence,
    and return clearer feedback if needed, based on Analysis Results 1, 2, 3, 4, 5, and 6.

    Args:
        env: The environment object.
        init_obs: The initial observation (provided by the template).
        task: The task description string.
        agent_action: The action string from the agent.
        logger: A logger object for debugging.

    Returns:
        tuple: (obs: str, done: bool, score: float)
               obs: The observation string for the agent.
               done: Boolean indicating if the task is finished.
               score: The score achieved after the action.
    """
    logger.debug(f"Processing agent action: {agent_action}")

    # Execute the action in the environment
    obs, _, done, info = env.step(agent_action)
    score = info["score"]
    logger.debug(f"Raw env.step results - obs: '{obs}', done: {done}, score: {score}")

    # Define common failure messages
    generic_failure_message = "No known action matches that input."
    ambiguous_goto_failure_message = "It's not clear how to get there from here."

    # Analysis Result 5: Refine feedback for 'go to' when already at the location.
    goto_match_5 = re.match(r"go to (.*)", agent_action, re.IGNORECASE)
    if goto_match_5 and obs.strip() == ambiguous_goto_failure_message:
        target_loc = goto_match_5.group(1).strip()
        # Assume this specific ambiguous message for 'go to' implies the agent is already there.
        custom_obs = f"You are already in the {target_loc}."
        logger.debug(f"Detected 'go to' action to potential current location '{target_loc}'. Replacing ambiguous feedback ('{obs.strip()}') with custom feedback: '{custom_obs}'")
        return custom_obs, done, score # Return immediately with clearer feedback

    # Check if the observation is the generic failure message AFTER checking the specific 'go to' failure
    is_generic_failure = obs.strip() == generic_failure_message

    # Analysis Result 1: Refine feedback for 'focus on' failure due to object absence.
    focus_match = re.match(r"focus on (.*)", agent_action, re.IGNORECASE)
    if focus_match and is_generic_failure:
        target_obj = focus_match.group(1).strip()
        # Provide specific feedback if 'focus on' failed generically, assuming it might be due to absence.
        custom_obs = f"You cannot focus on the {target_obj} because it is not present in your current location or inventory. Move to the location containing the {target_obj} or pick it up first."
        logger.debug(f"Detected 'focus on' failure for potentially non-present object '{target_obj}'. Replacing generic feedback ('{obs.strip()}') with custom feedback.")
        return custom_obs, done, score

    # Analysis Result 2 & 6: Refine feedback for 'go to' failure with generic message.
    # This handles generic failures for 'go to', covering inaccessible/misspelled/non-existent locations (Result 2)
    # AND attempts to navigate to non-locations like object groups or sub-areas (Result 6).
    goto_match_2_6 = re.match(r"go to (.*)", agent_action, re.IGNORECASE)
    if goto_match_2_6 and is_generic_failure:
        target = goto_match_2_6.group(1).strip()
        # Provide combined feedback covering both possibilities based on Analysis Results 2 and 6.
        custom_obs = f"You cannot go to '{target}'. This might be because:\n1. It's not a valid location name, is misspelled, or is not directly accessible from here. Check available exits in the room description.\n2. '{target}' refers to an object or area within your current location, not a separate navigable location. Use 'look around' to see objects here and interact with them directly."
        logger.debug(f"Detected 'go to' failure for target '{target}'. Replacing generic feedback ('{obs.strip()}') with combined custom feedback for Results 2 & 6.")
        return custom_obs, done, score

    # Analysis Result 3: Refine feedback for 'pick up' failure potentially due to location specifier.
    pickup_match = re.match(r"pick up (.*)", agent_action, re.IGNORECASE)
    if pickup_match and is_generic_failure:
        full_target = pickup_match.group(1).strip()
        logger.debug(f"Detected 'pick up' failure for target '{full_target}'. Checking for potential location specifiers.")

        # Heuristic to detect location specifiers (split by common prepositions indicating location)
        # Using word boundaries (\b) to avoid matching prepositions within words.
        prepositions = [r"\bon\b", r"\bin\b", r"\bat\b", r"\binside\b", r"\bunder\b", r"\bbehind\b", r"\bnear\b"]
        potential_object_name = full_target
        extra_details = ""

        for prep_pattern in prepositions:
            # Search for the preposition pattern in the target string
            match = re.search(prep_pattern, full_target, re.IGNORECASE)
            if match:
                # Split based on the match position
                split_index = match.start()
                potential_object_name = full_target[:split_index].strip()
                extra_details = full_target[split_index:].strip()
                # Basic check to ensure we didn't split off nothing or everything
                if potential_object_name and extra_details:
                    logger.debug(f"Potential object name: '{potential_object_name}', Extra details: '{extra_details}' based on preposition '{match.group(0)}'")
                    # Provide specific feedback suggesting the agent included extra details.
                    custom_obs = f"You tried to 'pick up {full_target}'. It seems you might have included extra location details like '{extra_details}'. Please refer to objects using only their exact names (e.g., 'pick up {potential_object_name}'). Check the room description or use 'look around'/'look at' to find the correct object names."
                    logger.debug(f"Replacing generic feedback with custom feedback for 'pick up' with suspected location specifier.")
                    return custom_obs, done, score
                else:
                    # Reset if split was not meaningful
                    potential_object_name = full_target
                    extra_details = ""
                    logger.debug(f"Split based on '{match.group(0)}' resulted in empty parts, ignoring.")

        # If the loop completes without returning, it means either no preposition was found,
        # or the split was not considered meaningful. In this case, the generic failure might be
        # due to other reasons (e.g., object truly doesn't exist, misspelling).
        # We return the original generic failure message.
        logger.debug(f"Generic failure for 'pick up {full_target}', but no common location preposition pattern detected or split was not meaningful. Returning original generic error.")
        # Fall through to default return (handled below)

    # Analysis Result 4: Refine feedback for 'look in'/'open' failure due to object absence.
    lookin_match = re.match(r"look in (.*)", agent_action, re.IGNORECASE)
    open_match = re.match(r"open (.*)", agent_action, re.IGNORECASE)

    if (lookin_match or open_match) and is_generic_failure:
        action_verb = "look in" if lookin_match else "open"
        target_obj = lookin_match.group(1).strip() if lookin_match else open_match.group(1).strip()

        # Provide specific feedback if 'look in' or 'open' failed generically, assuming it might be due to absence.
        custom_obs = f"You cannot {action_verb} the {target_obj} because it is not present in your current location or inventory. Move to the location containing the {target_obj} or pick it up first."
        logger.debug(f"Detected '{action_verb}' failure for potentially non-present object '{target_obj}'. Replacing generic feedback ('{obs.strip()}') with custom feedback.")
        return custom_obs, done, score

    # If it wasn't a handled failure case, return the original results from env.step.
    logger.debug(f"Returning standard results for action '{agent_action}'.")
    return obs, done, score