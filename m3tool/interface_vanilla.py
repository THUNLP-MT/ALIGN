import re
import logging
from typing import Any, Tuple # Assuming Task is defined elsewhere, added Any for env type hint clarity

# Define task type mapping for clarity if needed elsewhere
TASK_TYPE_MAP = {
    0: 'message_decoder',
    1: 'cryptobotanists_plant_dna_sequencer',
    2: 'trade_calculator',
    3: 'travel_itinerary_planning',
    4: 'web_browsing',
}

# Assume env object has methods like step() and attributes like name, instruction
# Assume logger is a configured logging.Logger instance

def InferRules(task_name: str, task_type_idx: int) -> str:
    """
    Contains the rules for environment and task execute logic for different task types.
    """
    if task_type_idx == 1: # cryptobotanists_plant_dna_sequencer
        # Add rule based on Analysis Result 4
        return "When providing the final answer for this task, please output only the single longest valid DNA sequence found. Do not output a list of all valid sequences."
    # Keep the previous logic for other tasks (no specific rules defined here previously)
    # Based on the analysis (Results 1, 2, 3), no specific rules needed to be defined here for other tasks,
    # as the feedback was handled during action processing.
    return "There is no specific rule for this environment beyond the standard tool usage format. Follow instructions carefully."

def WrapStep(env: Any, task_name: str, instruction: str, agent_action: str, logger: logging.Logger) -> Tuple[str, float, bool]:
    """
    Process the agent action:
    1. Check for common invocation errors based on Analysis Results 1 and 2:
        - Using func() instead of func.
        - Using func(arg) instead of func, arg.
    2. If no known format errors are detected, pass the action to the environment's step function.
    3. Check for specific scenarios based on Analysis Result 3:
        - If the task involves finding Allison Hill's email and the agent provides an incorrect final answer,
          modify the feedback to acknowledge the potential non-discoverability.
    4. Check for specific scenarios based on Analysis Result 4:
        - If the task is cryptobotanists_plant_dna_sequencer (task_type_idx=1) and the agent provides an incorrect answer formatted as a list,
          modify the feedback to clarify that only the single longest sequence is required.
    Return the next observation, reward, and done status.
    """
    obs, reward, done = "", 0.0, False
    # Log the task name and type for debugging purposes
    task_type_idx = -1
    for idx, name in TASK_TYPE_MAP.items():
        # A simple heuristic to find task_type_idx based on task_name or env type if available
        # This might need refinement depending on how task_type_idx is actually determined in the full system
        # Assuming env might have a type attribute or task_name implies type
        if name in task_name.lower(): # Basic check, might need improvement
             task_type_idx = idx
             break
        # Or if env has a type attribute: if env.type == name: task_type_idx = idx; break
    logger.debug(f"Processing action for task: '{task_name}' (Deduced Type Index: {task_type_idx})")


    # Combined check for tool_name(...) format based on Analysis Results 1 & 2
    parenthesis_args_pattern = r"^\s*Action:\s*([a-zA-Z0-9_]+)\((.*)\)\s*End Action\s*$"
    match = re.match(parenthesis_args_pattern, agent_action)

    if match:
        tool_name = match.group(1)
        args_inside = match.group(2).strip() # Remove leading/trailing whitespace from args

        if not args_inside: # Case: tool_name() - Analysis Result 1
            obs = f"Error: Found tool invocation with empty parentheses '{tool_name}()'. Tool names should be invoked without parentheses, e.g., 'Action: {tool_name} End Action'."
            reward = 0.0
            done = False
            logger.debug(f"Detected incorrect tool format: {agent_action} (empty parentheses). Provided specific feedback.")
            return obs, reward, done
        else: # Case: tool_name(arg) or tool_name(arg1, arg2) etc. - Analysis Result 2
            suggested_format = f"Action: {tool_name}, {args_inside} End Action"
            obs = f"Error: Found tool invocation with arguments inside parentheses like '{tool_name}({args_inside})'. Tool arguments should be provided after the tool name, separated by a comma, e.g., '{suggested_format}'."
            reward = 0.0
            done = False
            logger.debug(f"Detected incorrect tool format: {agent_action} (arguments in parentheses). Provided specific feedback.")
            return obs, reward, done
    else:
        # If the format doesn't match the specific error patterns, proceed as normal
        logger.debug(f"Action format '{agent_action}' doesn't match the tool_name(...) pattern, proceeding with env.step.")
        try:
            # Pass the original agent_action to env.step
            obs, reward, done = env.step(agent_action)
            logger.debug(f"env.step executed successfully for action: {agent_action}. Obs: {obs}, Reward: {reward}, Done: {done}")

            # --- Add specific handling for Analysis Result 3 ---
            # Refined check: Identify the task by checking for keywords "allison", "hill", and "email"
            # in the lowercased task name for robustness.
            task_name_lower = task_name.strip().lower()
            is_allison_hill_email_task = (
                "allison" in task_name_lower and
                "hill" in task_name_lower and
                "email" in task_name_lower
            )
            logger.debug(f"Checking for Allison Hill email task: Name='{task_name}', Lower='{task_name_lower}', Keywords found={is_allison_hill_email_task}")

            # Check if it's the target task, the agent submitted an answer, the answer was wrong (reward=0), and the task is marked as done.
            if is_allison_hill_email_task and agent_action.startswith("Answer:") and reward == 0.0 and done:
                logger.debug(f"Handling incorrect answer for Allison Hill email task ({task_name}). Original Obs: {obs}")
                # Modify the observation to be more informative about potential unsolvability
                original_feedback = obs # Keep the original feedback from env.step
                # Append a note about potential non-discoverability.
                modified_obs = f"{original_feedback} Note: The expected information ('allison.hill@taylor.net') might not be discoverable with the provided tools and website structure in this specific scenario."
                obs = modified_obs
                logger.info(f"Modified Obs for Allison Hill email task due to potential non-discoverability: {obs}")
            # --- End of specific handling for Analysis Result 3 ---


            # --- Add specific handling for Analysis Result 4 ---
            # Check if it's the DNA sequence task (task_type_idx=1), the agent submitted an answer,
            # the answer was wrong (reward=0), and the task is done.
            if task_type_idx == 1 and agent_action.startswith("Answer:") and reward == 0.0 and done:
                 # Extract the answer part
                 answer_content = agent_action.split("Answer:", 1)[1].strip()
                 # Check if the answer looks like a list
                 if answer_content.startswith('[') and answer_content.endswith(']'):
                     logger.debug(f"Handling incorrect answer format for DNA sequence task ({task_name}). Agent provided a list: {answer_content}. Original Obs: {obs}")
                     # Modify the observation to provide specific feedback
                     modified_obs = "Incorrect. Please output only the single longest valid DNA sequence, not a list of all valid sequences."
                     obs = modified_obs
                     logger.info(f"Modified Obs for DNA sequence task due to incorrect list format: {obs}")
            # --- End of specific handling for Analysis Result 4 ---


        except Exception as e:
            # Catch potential errors during env.step
            logger.error(f"Error during env.step for action '{agent_action}': {e}", exc_info=True)
            obs = f"Error executing action '{agent_action}': {e}"
            reward = 0.0
            # Assume error means task is not successfully completed, but allow agent to retry
            done = False

        # Return the final observation, reward, and done status
        return obs, reward, done