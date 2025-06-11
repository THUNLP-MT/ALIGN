import re
import logging

# Assuming logger is configured elsewhere in the main script
# Example configuration:
# import sys
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic for different task types,
    derived from common misalignments.
    """
    rules = """
# Environment Rules:

1.  **Navigation:** To move between locations using 'go to LOC', you must usually be in an adjacent location. Doors connecting locations might need to be opened first using 'open OBJ'. Check your surroundings ('look around') to see available locations and doors.
2.  **Action Syntax:** Use the specific action formats provided in the action space description (e.g., 'pour OBJ into OBJ', 'use OBJ on OBJ'). Natural language variations (like 'fill X with Y') might not be understood. Refer to the action list if unsure.
3.  **Object Naming:** Refer to objects using the names provided in the environment descriptions ('look around', 'look at OBJ', 'look in OBJ'). Sometimes, compound names (like 'bee hive door') might not work; try referring to the main object (e.g., 'open bee hive'). Ensure the object is present in your current location or inventory.
4.  **Prerequisites:** Some actions require specific conditions (e.g., needing an object in inventory, a device being activated, a container being open). Pay attention to feedback messages which might indicate missing prerequisites.
"""
    return rules

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action:
    1. Execute the action using env.step.
    2. Check for specific failure messages indicating potential world model misalignment.
    3. If a known misalignment pattern is detected, provide enhanced feedback in the observation.
    4. Return the observation (potentially enhanced), done status, and score.
    """
    logger.debug(f"Processing agent action: {agent_action}")
    original_obs, _, original_done, original_info = env.step(agent_action)
    original_score = original_info["score"]
    logger.debug(f"Original env.step obs: {original_obs}")
    logger.debug(f"Original env.step score: {original_score}, done: {original_done}")


    # Define common failure messages that might indicate misalignment
    failure_message_nomatch = "No known action matches that input."
    # Add other potential failure messages if needed, e.g., "You can't do that."

    # Check if the action failed with the specific "no match" message
    if failure_message_nomatch in original_obs:
        logger.debug(f"Action '{agent_action}' failed with '{failure_message_nomatch}'. Analyzing for misalignment.")
        enhanced_feedback = ""

        # Analysis Result 1: Navigation (go to LOC)
        match_go = re.match(r"go to (.*)", agent_action, re.IGNORECASE)
        if match_go:
            location = match_go.group(1).strip()
            enhanced_feedback = (f"Hint: Failed to '{agent_action}'. To move to '{location}', "
                                 f"ensure you are in an adjacent location and that any connecting doors are open. "
                                 f"Use 'look around' to check exits.")
            logger.debug(f"Navigation Misalignment Detected for '{agent_action}'. Providing hint.")

        # Analysis Result 2: Action Phrasing (Filling)
        # Use word boundaries (\b) to avoid matching substrings within words
        elif re.search(r"\bfill\b.*\bwith\b", agent_action, re.IGNORECASE) or \
             re.search(r"\bfill\b.*\bfrom\b", agent_action, re.IGNORECASE):
            enhanced_feedback = (f"Hint: The action '{agent_action}' seems like trying to fill something, but the syntax might be incorrect. "
                                 f"Try using 'pour LIQUID into CONTAINER' if you have the liquid, or 'use CONTAINER on SOURCE' (like 'use jug on sink').")
            logger.debug(f"Filling Phrasing Misalignment Detected for '{agent_action}'. Providing hint.")

        # Analysis Result 3: Object Naming (Opening)
        elif agent_action.lower().startswith("open "):
             match_open = re.match(r"open (.*)", agent_action, re.IGNORECASE)
             if match_open:
                obj_name = match_open.group(1).strip()
                parts = obj_name.split()
                # Check for common suffixes that might indicate referring to a part instead of the whole
                if len(parts) > 1 and parts[-1].lower() in ["door", "lid", "cover", "top", "cap"]:
                    base_obj_name = " ".join(parts[:-1])
                    enhanced_feedback = (f"Hint: Could not perform '{agent_action}' on '{obj_name}'. "
                                         f"Did you mean to 'open {base_obj_name}' instead?")
                    logger.debug(f"Object Naming Misalignment (Suffix) Detected for '{agent_action}'. Providing hint.")
                else:
                    # Generic hint for "open" failures if no specific pattern matched
                     enhanced_feedback = (f"Hint: Failed to '{agent_action}'. Make sure '{obj_name}' is an object "
                                         f"that can be opened and is present in the current location.")
                     logger.debug(f"Generic Open Failure Hint for '{agent_action}'.")


        # If a specific misalignment pattern was detected, append the hint
        if enhanced_feedback:
            # Combine original observation (which contains the failure message) with the hint
            enhanced_obs = f"{original_obs} {enhanced_feedback}"
            return enhanced_obs, original_done, original_score
        else:
            # If it failed with "No known action..." but didn't match specific patterns, provide a generic hint
            generic_hint = (f"Hint: The action '{agent_action}' was not recognized or is not possible right now. "
                            f"Please check the Action Space list and ensure object names are correct and objects are accessible.")
            enhanced_obs = f"{original_obs} {generic_hint}"
            logger.debug(f"Generic Failure '{failure_message_nomatch}' for '{agent_action}'. Providing generic hint.")
            return enhanced_obs, original_done, original_score


    # If the action succeeded, or failed with a different message, return the original results
    logger.debug(f"Action '{agent_action}' processed without specific misalignment detection or intervention.")
    return original_obs, original_done, original_score