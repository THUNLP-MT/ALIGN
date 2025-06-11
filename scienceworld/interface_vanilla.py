# Imports
import re
import logging
import io
import sys # Keep for potential future use, though stream preferred

# Assuming logger is configured elsewhere
# log_stream = io.StringIO() # Example for testing capture
# logger = logging.getLogger("AgentEnvInteraction") # ... setup ...

# --- InferRules function (Refined based on Analysis 12) ---
def InferRules(init_obs: str, task: str) -> str:
    """
    Generates environment rules based on the initial observation and task description.
    Includes rules for 'focus on', container interactions, common syntax issues, and movement.
    """
    rules = []
    rules.append("General Environment Rules:")
    rules.append("- Only one action can be performed per turn.")
    rules.append("- Actions must be chosen from the available action space provided in the system prompt.")
    rules.append("- Ensure objects exist and are accessible before interacting with them (e.g., check 'look around', 'look in CONTAINER', check your current location).")
    rules.append("- If the environment presents multiple objects with the same name (ambiguity), it will ask you to clarify by choosing a number (e.g., 'Which X do you mean? 0: X 1: X'). Respond with ONLY the number (e.g., '0') to select the corresponding item.")

    # --- Container Interaction Rules (Existing - Analysis 9) ---
    rules.append("\nInteracting with Containers:")
    rules.append("- The action 'take OBJ from CONTAINER' is generally not valid.")
    rules.append("- To get an item from a container (like a jar, box, freezer):")
    rules.append("  1. You often need to 'pick up CONTAINER' first to hold it.")
    rules.append("  2. Then, you might need to 'move OBJ to inventory' or 'put down OBJ' somewhere else.")
    rules.append("- The action 'pick up OBJ from CONTAINER' might also not work for all containers. If it fails, try picking up the container itself.")
    rules.append("- Use 'look in CONTAINER' to see contents.")

    # --- Focus Rules (Existing + Refinement for Analysis 8, 12) ---
    # Use refined regex to find required focus objects based on "focus on the ..." pattern
    required_focus_objects_raw = re.findall(r"focus on the (.*?)(?: you created|\.|$)", task, re.IGNORECASE)
    required_focus_objects = [obj.strip() for obj in required_focus_objects_raw if obj.strip()]

    if required_focus_objects:
        rules.append("\nTask-Specific Rules for 'focus on':")
        required_objects_str = " or ".join([f"'{obj}'" for obj in required_focus_objects])
        rules.append(f"- The 'focus on OBJ' action has a special meaning in this task and is used to signal progress or completion.")
        rules.append(f"- Use 'focus on' ONLY for the required task items related to: {required_objects_str}.")
        # Refinement for Analysis 12: Add note about conceptual focus
        rules.append(f"- Sometimes, the task might ask you to focus on an item with a specific property (e.g., 'the animal with the longest lifespan'). In such cases, you might need to identify the specific item (e.g., 'crocodile') and use 'focus on [specific item name]' to fulfill the requirement.")
        rules.append(f"- You must use this command on the specified items when they are ready (e.g., created, planted, in the correct location), as per the task instructions.")
        rules.append(f"- Using 'focus on' for any other object (e.g., 'focus on blast furnace', 'focus on beaker') is considered an incorrect action for this task and will not advance your progress. You will receive feedback if you attempt this.")
        rules.append(f"- If you need to disambiguate one of the required focus items (e.g., 'focus on {required_focus_objects[0]}'), respond with the number only, not 'focus on {required_focus_objects[0]} [number]'.")
        rules.append(f"- If 'focus on' fails for a required item, ensure it exists, is ready (prerequisites met), you are in the correct location, and you are using the exact correct name (sometimes this might be 'OBJ in RECEPTACLE').") # Refinement for Analysis 8 & 5
    else:
         # General note about 'focus on' if not specifically required by the "focus on the..." pattern
         rules.append("\nNote on 'focus on OBJ':")
         rules.append("- The 'focus on OBJ' action typically signals intent towards a task-critical object.")
         rules.append("- Its specific function and validity may vary depending on the task. Check the task description for any specific instructions regarding 'focus on'.")
         # Refinement for Analysis 12: Add note about conceptual focus
         rules.append(f"- Sometimes, the task might ask you to focus on an item with a specific property. In such cases, you might need to identify the specific item and use 'focus on [specific item name]' to fulfill the requirement.")
         rules.append("- This action might require the object to be in a specific state (e.g., created, planted) or require a specific name format (e.g., 'OBJ in RECEPTACLE').") # Refinement for Analysis 8
         rules.append("- If 'focus on' fails, check if the object exists, if prerequisites are met, if you are in the correct location, and if you are using the correct name.") # Refinement for Analysis 8 & 5

    # --- Movement Rules (Existing - Analysis 10, 11) ---
    rules.append("\nMovement Rules:")
    rules.append("- Use 'go to LOC' to move between locations (e.g., 'go to kitchen').")
    rules.append("- You can only 'go to' locations that are directly connected to your current location.")
    rules.append("- If 'go to LOC' fails, it might be because the location is not directly connected or you are already there. Use 'look around' to see available exits and connected locations.") # Added note about being already there for Analysis 13
    rules.append("- The syntax 'go to DESTINATION from SOURCE' is not valid. Use only 'go to DESTINATION'.") # Added based on Analysis 11 feedback logic
    rules.append("- The 'teleport to LOC' action allows direct travel but might not be available in all tasks.")

    # --- General Syntax Notes (Existing - Analysis 6 & 7) ---
    rules.append("\nGeneral Action Syntax Notes:")
    rules.append("- For actions like 'open OBJ', use the object's base name (e.g., 'open freezer', not 'open freezer door').")
    rules.append("- The 'wait' command only accepts 'wait1' (no space) to pass a single time step. Other durations (e.g., 'wait 10') are not supported.")

    return "\n".join(rules)


# --- _get_current_state helper function (Unchanged) ---
def _get_current_state(env, logger, previous_score=0.0):
    """Helper function to get current observation and score using 'look around'."""
    try:
        # Use 'look around' as it's less likely to change game state significantly
        # than repeating the failed action or doing nothing.
        current_obs, _, _, current_info = env.step("look around")
        current_score = current_info["score"]
        logger.debug(f"Performed 'look around' to get current state. Score: {current_score}")
        # Extract location using the updated regex
        location_match = re.search(r"(?:You are in|This room is called) the (.*?)\.", current_obs) # MODIFIED REGEX
        current_location = f"the {location_match.group(1)}" if location_match else "your current location"
        return current_obs, current_score, current_location
    except Exception as e:
        logger.error(f"Error performing 'look around' to get current state: {e}")
        try:
            # Fallback if 'look around' fails (less likely but possible)
            current_obs = env.look()
            current_score = previous_score # Assume score didn't change if look around failed
            logger.warning(f"Fallback to env.look(). Score assumed {current_score}")
        except AttributeError:
            logger.error("env.look() not available as fallback.")
            current_obs = "Error: Could not retrieve current environment state."
            current_score = previous_score # Keep score from before the failed action
        # Try to extract location from fallback obs if possible using the updated regex
        location_match = re.search(r"(?:You are in|This room is called) the (.*?)\.", current_obs) # MODIFIED REGEX
        current_location = f"the {location_match.group(1)}" if location_match else "your current location"
        return current_obs, current_score, current_location

# --- WrapStep function (Refined based on Analysis 12, 13) ---
def WrapStep(env, init_obs: str, task: str, agent_action: str, logger: logging.Logger):
    """
    Processes the agent's action, providing specific feedback for incorrect 'focus on' usage
    (including conceptual vs specific targets), invalid 'take from', 'pick up from', 'open door',
    'wait' syntax/usage, and invalid 'go to' attempts (including syntax errors, non-adjacency,
    and attempting to go to the current location), without causing immediate failure where possible.
    Uses substring matching for target identification and checks observation strings/exceptions
    for environment-reported errors.

    Args:
        env: The environment instance.
        init_obs: The initial observation string.
        task: The task description string.
        agent_action: The action string from the agent.
        logger: The logger instance.

    Returns:
        A tuple containing:
        - obs (str): The observation string after the action (or feedback).
        - done (bool): Whether the task is done.
        - score (float): The score after the action.
    """
    logger.debug(f"Processing agent action: {agent_action}")
    action_normalized = agent_action.lower().strip()
    current_score = 0.0 # Placeholder, will be updated

    # --- Get initial score before attempting action (needed for fallback in _get_current_state) ---
    # (Placeholder - score is retrieved reliably after action attempt or failure)
    pass

    # --- Check for invalid 'wait' command syntax (Analysis 7) ---
    wait_match = re.match(r"wait\s*(\d+)", action_normalized)
    if wait_match and action_normalized != "wait1":
        wait_duration = wait_match.group(1)
        logger.warning(f"Intercepted invalid wait command: '{agent_action}'. Only 'wait1' is supported.")
        current_obs, current_score, current_location = _get_current_state(env, logger)
        custom_feedback = (
            f"\n\n[Environment Feedback]: Your action '{agent_action}' uses an invalid format or duration.\n"
            f"Reason: The environment only supports waiting for a single time step using the command 'wait1' (no space between 'wait' and '1'). Waiting for {wait_duration} steps is not supported.\n"
            f"Your action was not executed. Please use 'wait1' if you intend to wait."
        )
        final_obs = current_obs + custom_feedback
        return final_obs, False, current_score

    # --- Check for invalid 'take ... from ...' syntax (Analysis 9) ---
    take_from_match = re.match(r"take\s+(.*)\s+from\s+(.*)", action_normalized)
    if take_from_match:
        taken_object = take_from_match.group(1).strip()
        container = take_from_match.group(2).strip()
        logger.warning(f"Intercepted invalid action syntax: '{agent_action}'. 'take ... from ...' is not supported.")
        current_obs, current_score, current_location = _get_current_state(env, logger)
        custom_feedback = (
            f"\n\n[Environment Feedback]: Your action '{agent_action}' uses invalid syntax.\n"
            f"Reason: The action 'take {taken_object} from {container}' is not supported in this environment.\n"
            f"To get items from containers like '{container}', you usually need to 'pick up {container}' first, or check if you can 'move {taken_object} to inventory'.\n"
            f"Your action was not executed."
        )
        final_obs = current_obs + custom_feedback
        return final_obs, False, current_score

    # --- Check 'focus on' action (Analysis 1, 2, 4, 5, 8, 12) ---
    focus_match = re.match(r"focus on (.*)", action_normalized)
    if focus_match:
        focused_object_raw = focus_match.group(1).strip()
        normalized_focused_object = focused_object_raw.lower()

        # Get required focus objects based on "focus on the ..." pattern in the task
        required_focus_objects_raw = re.findall(r"focus on the (.*?)(?: you created|\.|$)", task, re.IGNORECASE)
        normalized_required_objects = [obj.strip().lower() for obj in required_focus_objects_raw if obj.strip()]
        required_objects_str_display = " or ".join([f"'{obj.strip()}'" for obj in required_focus_objects_raw if obj.strip()])

        # --- Check for Ambiguity Resolution Syntax Error FIRST (Analysis 2) ---
        # Matches 'focus on base_object number'
        ambiguity_match = re.match(r"^(.*)\s+(\d+)$", focused_object_raw)
        if ambiguity_match:
            base_object = ambiguity_match.group(1).strip()
            number_str = ambiguity_match.group(2)
            normalized_base_object = base_object.lower()

            # Check if the base object is related to *any* required focus object
            is_required_base_object = False
            if normalized_required_objects:
                 for req_obj in normalized_required_objects:
                     # Use substring matching for robustness
                     if req_obj in normalized_base_object or normalized_base_object in req_obj:
                         is_required_base_object = True
                         logger.debug(f"Ambiguity syntax check: Base object '{normalized_base_object}' potentially matches required '{req_obj}'.")
                         break
            # Also consider if the task generally mentions focusing on this object type, even if not in "focus on the..."
            elif normalized_base_object in task.lower():
                 is_required_base_object = True # Assume relevant if mentioned in task and ambiguity arises
                 logger.debug(f"Ambiguity syntax check: Base object '{normalized_base_object}' appears in task description.")


            if is_required_base_object:
                logger.warning(f"Intercepted incorrect ambiguity resolution syntax: '{agent_action}'. Agent should use only the number '{number_str}'.")
                current_obs, current_score, current_location = _get_current_state(env, logger)
                custom_feedback = (
                    f"\n\n[Environment Feedback]: Your action '{agent_action}' uses an invalid format for selecting an option.\n"
                    f"Reason: It seems you are trying to select option {number_str} for '{base_object}'. To select this option, please respond with just the number: '{number_str}'.\n"
                    f"Your action was not executed."
                )
                final_obs = current_obs + custom_feedback
                return final_obs, False, current_score
        # --- End Ambiguity Syntax Check ---

        # --- Determine if the focus target is potentially correct (Analysis 1, 12) ---
        is_potentially_correct_target = False
        matched_req_obj = None
        is_conceptual_focus_task = False # Flag for Analysis 12

        if normalized_required_objects:
            # Check if any required object looks conceptual (Analysis 12)
            conceptual_keywords = ["with the", "longest", "shortest", "heaviest", "lightest", "smallest", "largest"]
            for req_obj_raw in required_focus_objects_raw:
                if any(keyword in req_obj_raw.lower() for keyword in conceptual_keywords):
                    is_conceptual_focus_task = True
                    logger.debug(f"Detected conceptual focus task based on required object: '{req_obj_raw}'")
                    break

            # Check if agent's target matches any required object
            for req_obj in normalized_required_objects:
                # Use substring matching: is required obj part of agent target, or agent target part of required obj?
                if req_obj in normalized_focused_object or (normalized_focused_object and normalized_focused_object in req_obj):
                    is_potentially_correct_target = True
                    matched_req_obj = req_obj
                    logger.debug(f"Focus target potentially matches required '{req_obj}': Agent specified '{normalized_focused_object}'.")
                    break

            # Analysis 12 Relaxation: If it's a conceptual task, don't block focusing on a specific instance yet.
            # Allow it to proceed to env.step, even if it didn't literally match the conceptual phrase.
            if is_conceptual_focus_task and not is_potentially_correct_target:
                logger.info(f"Conceptual focus task detected. Allowing action '{agent_action}' targeting specific instance '{focused_object_raw}' to proceed, bypassing literal match check against '{required_objects_str_display}'.")
                is_potentially_correct_target = True # Override: Let the environment check the instance

            # Intercept BEFORE execution ONLY IF:
            # 1. There are specific required objects AND
            # 2. It's NOT a conceptual focus task where the agent might be trying a specific instance AND
            # 3. The agent's target did not match any required object.
            if not is_potentially_correct_target: # This now correctly handles the conceptual case due to the override above
                logger.warning(f"Intercepted incorrect focus target action on '{focused_object_raw}'. Task requires focus on items related to: {required_objects_str_display}. Providing feedback.")
                current_obs, current_score, current_location = _get_current_state(env, logger)
                custom_feedback = (
                    f"\n\n[Environment Feedback]: Your action '{agent_action}' was not executed as intended.\n"
                    f"Reason: The 'focus on' action has a specific purpose in this task. It should only be used for items related to: {required_objects_str_display}.\n"
                    f"Using 'focus on {focused_object_raw}' is not the correct procedure here. Please choose another action or use 'focus on' with the correct item when it is ready."
                )
                # Add hint for conceptual tasks if applicable (even if interception happens for other reasons)
                if is_conceptual_focus_task:
                     custom_feedback += f"\nNote: For tasks requiring focus based on a property (like '{required_objects_str_display}'), you usually need to identify the specific item that has that property and focus on its name."

                final_obs = current_obs + custom_feedback
                return final_obs, False, current_score
        else:
            # Task does not have "focus on the ..." requirement. Assume potentially correct.
            is_potentially_correct_target = True
            logger.debug(f"Proceeding with focus action '{agent_action}'. No specific required objects.")


        # --- Try executing the focus action ---
        if is_potentially_correct_target:
            try:
                obs, _, done, info = env.step(agent_action)
                score = info["score"]
                logger.debug(f"Executed '{agent_action}'. Obs received: '{obs[:100]}...', Done: {done}, Score: {score}")

                # Check observation for failure messages
                error_detected_in_obs = False
                error_phrases = ["no known action", "unknown action", "could not find object", "object not found", "is not here", "nothing happens", "don't know how to"]
                obs_lower = obs.lower()
                failure_phrase_found = None
                for phrase in error_phrases:
                    # Avoid matching harmless phrases like "You are not focusing on anything"
                    if phrase in obs_lower and not obs_lower.startswith("you are not"):
                        error_detected_in_obs = True
                        failure_phrase_found = phrase
                        logger.warning(f"Detected potential error phrase '{phrase}' in observation string for focus action '{agent_action}'.")
                        break

                if error_detected_in_obs:
                    # Focus action failed based on observation content
                    logger.warning(f"Handling focus failure based on observation content for target '{focused_object_raw}'. Failure phrase: '{failure_phrase_found}'.")
                    current_obs, current_score, current_location = _get_current_state(env, logger, score) # Pass score from failed step

                    # --- Provide Enhanced Feedback (Analysis 4, 5, 8) ---
                    feedback_parts = [
                        f"\n\n[Environment Feedback]: Your action '{agent_action}' did not succeed (Observation: \"{obs.strip()}\")."
                    ]
                    reasons = []
                    # Reason 1: Existence/Name/Readiness (Analysis 4, 8)
                    object_in_task = focused_object_raw.lower() in task.lower()
                    reason_existence = f"The object '{focused_object_raw}' might not exist yet, might not be ready (e.g., needs planting, mixing), or you might need to use its exact name."
                    if object_in_task:
                         reason_existence += " Check the task steps and ensure all prerequisites are met."
                    # Suggest specific naming if applicable (Analysis 8)
                    if "seed" in focused_object_raw.lower() and "plant" in task.lower():
                         reason_existence += " For planted items, the name might be like 'orange seed in flower pot'."
                    elif matched_req_obj: # If it matched a "focus on the..." object originally
                         original_matched_req_obj_display = f"'{matched_req_obj}'" # Default to normalized
                         for raw_obj in required_focus_objects_raw:
                             if raw_obj.strip().lower() == matched_req_obj:
                                 original_matched_req_obj_display = f"'{raw_obj.strip()}'" # Use original casing if found
                                 break
                         reason_existence += f" Ensure you are using the correct name, perhaps '{original_matched_req_obj_display}' if that is the expected item."
                    # Add hint for conceptual tasks (Analysis 12)
                    elif is_conceptual_focus_task:
                         reason_existence += f" For tasks requiring focus based on a property (like '{required_objects_str_display}'), ensure you have identified the correct specific item that has that property and are using its exact name."


                    reasons.append(reason_existence)

                    # Reason 2: Location (Analysis 5)
                    # Basic location check - more sophisticated checks might need external knowledge
                    if failure_phrase_found in ["could not find object", "object not found", "is not here"]:
                         reasons.append(f"The object might exist but not be accessible or interactable from your current location ({current_location}).")
                         # Example Task-Specific Location Hint (can be generalized if needed)
                         if "greenhouse" in task.lower() and ("red box" in focused_object_raw or "green box" in focused_object_raw) and "greenhouse" not in current_location:
                             reasons.append("Remember, the red and green boxes are expected to be in the greenhouse.")

                    feedback_parts.append("Possible Reasons:")
                    for i, r in enumerate(reasons):
                        feedback_parts.append(f"- {r}")
                    feedback_parts.append("Suggestion: Please check the environment state, your location, ensure the object is ready, and verify you are using the correct name and syntax.")

                    custom_feedback = "\n".join(feedback_parts)
                    final_obs = current_obs + custom_feedback
                    return final_obs, False, current_score # Return feedback, keep task running

                else:
                    # Focus action seemed successful based on observation
                    if done and score < 0: # Check for unexpected failure on success (Analysis 1 edge case)
                        logger.warning(f"Focus action '{agent_action}' resulted in task completion with score {score}. Prerequisites might have been missed.")
                        obs += (
                            f"\n\n[Environment Note]: The task finished after focusing on '{focused_object_raw}', but the score ({score}) indicates potential issues. "
                            f"Ensure all necessary steps and conditions were met before using the 'focus on' command."
                        )
                    return obs, done, score # Return original results

            except Exception as e:
                 # Focus action failed with an exception
                 logger.error(f"Exception occurred executing focus action '{agent_action}': {e}")
                 error_msg_str = str(e)
                 current_obs, current_score, current_location = _get_current_state(env, logger) # Get state after exception

                 # --- Provide Enhanced Feedback (Analysis 4, 5, 8) ---
                 feedback_parts = [
                     f"\n\n[Environment Feedback]: Your action '{agent_action}' failed with an error: \"{error_msg_str}\"."
                 ]
                 reasons = []
                 error_phrases_exception = ["no known action", "unknown action", "could not find object", "object not found", "is not here"]
                 exception_indicates_issue = any(phrase in error_msg_str.lower() for phrase in error_phrases_exception)

                 if exception_indicates_issue:
                     # Reason 1: Existence/Name/Readiness (Analysis 4, 8)
                     object_in_task = focused_object_raw.lower() in task.lower()
                     reason_existence = f"The object '{focused_object_raw}' might not exist yet, might not be ready (e.g., needs planting, mixing), or you might need to use its exact name."
                     if object_in_task:
                          reason_existence += " Check the task steps and ensure all prerequisites are met."
                     # Suggest specific naming if applicable (Analysis 8)
                     if "seed" in focused_object_raw.lower() and "plant" in task.lower():
                          reason_existence += " For planted items, the name might be like 'orange seed in flower pot'."
                     elif matched_req_obj: # If it matched a "focus on the..." object originally
                          original_matched_req_obj_display = f"'{matched_req_obj}'" # Default to normalized
                          for raw_obj in required_focus_objects_raw:
                              if raw_obj.strip().lower() == matched_req_obj:
                                  original_matched_req_obj_display = f"'{raw_obj.strip()}'" # Use original casing if found
                                  break
                          reason_existence += f" Ensure you are using the correct name, perhaps '{original_matched_req_obj_display}' if that is the expected item."
                     # Add hint for conceptual tasks (Analysis 12)
                     elif is_conceptual_focus_task:
                          reason_existence += f" For tasks requiring focus based on a property (like '{required_objects_str_display}'), ensure you have identified the correct specific item that has that property and are using its exact name."

                     reasons.append(reason_existence)

                     # Reason 2: Location (Analysis 5)
                     reasons.append(f"The object might exist but not be accessible or interactable from your current location ({current_location}).")
                     # Example Task-Specific Location Hint
                     if "greenhouse" in task.lower() and ("red box" in focused_object_raw or "green box" in focused_object_raw) and "greenhouse" not in current_location:
                         reasons.append("Remember, the red and green boxes are expected to be in the greenhouse.")

                 else: # General error
                     reasons.append(f"An unexpected error occurred: {error_msg_str}")

                 feedback_parts.append("Possible Reasons:")
                 for i, r in enumerate(reasons):
                     feedback_parts.append(f"- {r}")
                 feedback_parts.append("Suggestion: Please check the environment state, your location, ensure the object is ready, and verify you are using the correct name and syntax.")

                 custom_feedback = "\n".join(feedback_parts)
                 final_obs = current_obs + custom_feedback
                 return final_obs, False, current_score # Return feedback, keep task running
        # --- End of 'focus on' specific logic ---

    else:
        # --- Handle standard actions (including checks for 'open ... door', 'pick up ... from ...', 'go to ...') ---

        # --- Check for invalid 'go to ... from ...' syntax FIRST (Analysis 11) ---
        go_to_from_match = re.match(r"go to\s+(.+)\s+from\s+(.+)", action_normalized, re.IGNORECASE)
        if go_to_from_match:
            destination = go_to_from_match.group(1).strip()
            source = go_to_from_match.group(2).strip()
            logger.warning(f"Intercepted invalid 'go to ... from ...' syntax: '{agent_action}'.")
            current_obs, current_score, current_location = _get_current_state(env, logger)
            custom_feedback = (
                f"\n\n[Environment Feedback]: Your action '{agent_action}' uses an invalid command format.\n"
                f"Reason: The 'go to' action only accepts the destination location name (e.g., 'go to {destination}' or 'go to hallway'). Specifying the source location using 'from {source}' is not supported.\n"
                f"Suggestion: Please use 'look around' to see valid exits from your current location ({current_location}) and then use 'go to [valid exit]'."
            )
            final_obs = current_obs + custom_feedback
            return final_obs, False, current_score
        # --- End 'go to ... from ...' check ---

        # --- If not the invalid 'go to from' syntax, proceed with standard execution ---
        logger.debug(f"Executing standard action: {agent_action}")
        try:
            obs, _, done, info = env.step(agent_action)
            score = info["score"]
            logger.debug(f"Executed '{agent_action}'. Obs: '{obs[:100]}...', Done: {done}, Score: {score}")

            # --- Check for specific failure cases based on observation AFTER successful execution ---

            # Check for 'go to current location' failure (Analysis 13)
            go_to_match = re.match(r"go to (.*)", action_normalized)
            # Use the specific, potentially ambiguous feedback string as the trigger
            go_to_current_loc_feedback = "It's not clear how to get there from here."
            if go_to_match and go_to_current_loc_feedback in obs:
                target_location = go_to_match.group(1).strip()
                logger.warning(f"Detected ambiguous feedback '{go_to_current_loc_feedback}' after 'go to {target_location}'. Assuming agent tried to go to current location.")
                # Get current state to ensure obs is fresh before adding feedback
                current_obs, current_score, current_location = _get_current_state(env, logger, score) # Pass score from failed step
                custom_feedback = (
                    f"\n\n[Environment Feedback]: Your action '{agent_action}' failed.\n"
                    f"Reason: You cannot use 'go to {target_location}' because you are already in that location ({current_location}).\n"
                    f"Suggestion: Use 'look around' to see available exits to other locations."
                )
                final_obs = current_obs + custom_feedback
                # Since the original step technically executed (but resulted in this feedback),
                # we return done=False and the score from that step.
                return final_obs, False, current_score

            # Check observation for other general failure messages
            error_detected_in_obs = False
            # Added more potential failure phrases, especially for movement
            error_phrases = ["no known action", "unknown action", "could not find object", "object not found", "is not here", "nothing happens", "cannot", "can't go that way", "not a valid exit", "don't know how to go there"]
            obs_lower = obs.lower()
            failure_phrase_found = None
            for phrase in error_phrases:
                 # Avoid matching harmless phrases like "You cannot see that" if it's just descriptive
                 # Also avoid matching the specific 'go to current loc' feedback handled above
                 if phrase in obs_lower and not obs_lower.startswith("you are carrying") and not obs_lower.startswith("you are in") and go_to_current_loc_feedback not in obs:
                     error_detected_in_obs = True
                     failure_phrase_found = phrase
                     logger.warning(f"Detected potential failure phrase '{phrase}' in observation string for standard action '{agent_action}'.")
                     break

            if error_detected_in_obs:
                 # Action failed based on observation content. Provide specific feedback.
                 logger.warning(f"Handling failure based on observation content for standard action '{agent_action}'. Failure phrase: '{failure_phrase_found}'.")
                 current_obs, current_score, current_location = _get_current_state(env, logger, score) # Pass score from failed step

                 custom_feedback = None
                 # Check for 'go to LOC' failure due to non-adjacency (Analysis 10)
                 # This check should only trigger for the valid 'go to LOC' syntax,
                 # as the invalid 'go to ... from ...' syntax is caught above.
                 # Also ensure it's not the 'go to current loc' case handled above.
                 if go_to_match and failure_phrase_found in ["no known action", "unknown action", "cannot", "can't go that way", "not a valid exit", "don't know how to go there"]:
                     target_location = go_to_match.group(1).strip()
                     logger.info(f"Detected failed 'go to {target_location}' action, likely due to non-adjacency from {current_location}.")
                     custom_feedback = (
                         f"\n\n[Environment Feedback]: Your action '{agent_action}' failed (Observation: \"{obs.strip()}\").\n"
                         f"Reason: You cannot go directly to '{target_location}' from your current location ({current_location}). Movement is only possible between directly connected locations.\n"
                         f"Suggestion: Use 'look around' to see the available exits and connected locations from here."
                     )

                 # Check for 'open ... door' syntax error (Analysis 6)
                 open_door_match = re.match(r"open (.*) door", action_normalized)
                 if not custom_feedback and open_door_match and failure_phrase_found in ["no known action", "unknown action", "could not find object", "object not found", "cannot"]:
                     target_object = open_door_match.group(1).strip()
                     logger.info(f"Detected failed 'open ... door' syntax for '{agent_action}'. Suggesting 'open {target_object}'.")
                     custom_feedback = (
                         f"\n\n[Environment Feedback]: Your action '{agent_action}' failed (Observation: \"{obs.strip()}\").\n"
                         f"Reason: The syntax might be incorrect. To open objects like '{target_object}', try using the command 'open {target_object}' instead of specifying 'door'.\n"
                         f"Suggestion: Please check the object name and try the suggested syntax."
                     )

                 # Check for 'pick up ... from ...' failure (Analysis 9)
                 pickup_from_match = re.match(r"pick up\s+(.*)\s+from\s+(.*)", action_normalized)
                 if not custom_feedback and pickup_from_match and failure_phrase_found in ["no known action", "unknown action", "cannot"]:
                     picked_object = pickup_from_match.group(1).strip()
                     container = pickup_from_match.group(2).strip()
                     logger.info(f"Detected failed 'pick up ... from ...' action for '{agent_action}'. Suggesting 'pick up {container}'.")
                     custom_feedback = (
                         f"\n\n[Environment Feedback]: Your action '{agent_action}' failed (Observation: \"{obs.strip()}\").\n"
                         f"Reason: The action 'pick up {picked_object} from {container}' might not be supported for this container.\n"
                         f"Suggestion: Try picking up the container itself first using 'pick up {container}'. You might then be able to access its contents."
                     )

                 # Default failure feedback
                 if not custom_feedback:
                     custom_feedback = (
                         f"\n\n[Environment Feedback]: Your action '{agent_action}' did not succeed as expected in {current_location} (Observation: \"{obs.strip()}\").\n"
                         f"Reason: This could be due to an incorrect command, a non-existent or inaccessible object, or the action not being applicable in the current situation.\n"
                         f"Suggestion: Please check the command syntax, object names, your location, and the environment state."
                     )

                 final_obs = current_obs + custom_feedback
                 return final_obs, False, current_score # Return corrected obs, keep task running

            else:
                 # Action executed successfully without known error phrases in obs
                 return obs, done, score # Return original results

        except Exception as e:
            # Standard action failed with an exception
            logger.error(f"Error executing standard action '{agent_action}': {e}")
            error_msg_str = str(e)
            current_obs, current_score, current_location = _get_current_state(env, logger) # Get state after exception

            custom_feedback = None
            # Added more potential failure phrases, especially for movement
            error_phrases_exception = ["no known action", "unknown action", "could not find object", "object not found", "is not here", "cannot", "can't go that way", "not a valid exit", "don't know how to go there"]
            exception_indicates_issue = any(phrase in error_msg_str.lower() for phrase in error_phrases_exception)

            # Check for 'go to LOC' failure due to non-adjacency based on exception (Analysis 10)
            # Ensure it wasn't the 'go to ... from ...' pattern caught earlier
            go_to_match = re.match(r"go to (.*)", action_normalized)
            if not go_to_from_match and go_to_match and exception_indicates_issue:
                 target_location = go_to_match.group(1).strip()
                 logger.info(f"Detected failed 'go to {target_location}' action based on exception, likely due to non-adjacency from {current_location}.")
                 custom_feedback = (
                     f"\n\n[Environment Feedback]: Your action '{agent_action}' failed with an error: \"{error_msg_str}\".\n"
                     f"Reason: You might not be able to go directly to '{target_location}' from your current location ({current_location}). Movement is only possible between directly connected locations.\n"
                     f"Suggestion: Use 'look around' to see the available exits and connected locations from here."
                 )

            # Check for 'open ... door' syntax error based on exception (Analysis 6)
            open_door_match = re.match(r"open (.*) door", action_normalized)
            if not custom_feedback and open_door_match and exception_indicates_issue:
                target_object = open_door_match.group(1).strip()
                logger.info(f"Detected failed 'open ... door' syntax for '{agent_action}' based on exception. Suggesting 'open {target_object}'.")
                custom_feedback = (
                    f"\n\n[Environment Feedback]: Your action '{agent_action}' failed with an error: \"{error_msg_str}\".\n"
                    f"Reason: The syntax might be incorrect. To open objects like '{target_object}', try using the command 'open {target_object}' instead of specifying 'door'.\n"
                    f"Suggestion: Please check the object name and try the suggested syntax."
                )

            # Check for 'pick up ... from ...' failure based on exception (Analysis 9)
            pickup_from_match = re.match(r"pick up\s+(.*)\s+from\s+(.*)", action_normalized)
            if not custom_feedback and pickup_from_match and exception_indicates_issue:
                picked_object = pickup_from_match.group(1).strip()
                container = pickup_from_match.group(2).strip()
                logger.info(f"Detected failed 'pick up ... from ...' action for '{agent_action}' based on exception. Suggesting 'pick up {container}'.")
                custom_feedback = (
                    f"\n\n[Environment Feedback]: Your action '{agent_action}' failed with an error: \"{error_msg_str}\".\n"
                    f"Reason: The action 'pick up {picked_object} from {container}' might not be supported for this container.\n"
                    f"Suggestion: Try picking up the container itself first using 'pick up {container}'. You might then be able to access its contents."
                )

            # Default error feedback based on exception
            if not custom_feedback:
                custom_feedback = (
                    f"\n\n[Environment Feedback]: Error executing action '{agent_action}' in {current_location}.\n"
                    f"Reason: {e}\n"
                    f"Suggestion: Please check the command syntax, object names, your location, and the environment state."
                )

            final_obs = current_obs + custom_feedback
            return final_obs, False, current_score # Return error message, keep task running