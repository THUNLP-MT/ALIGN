import re
import logging
from typing import Tuple, Dict, List, Any, Optional, Set

# Assuming TOOL_DESC is accessible globally or passed appropriately in a real scenario.
# Example TOOL_DESC structure (needed for context, but not modified here)
# TOOL_DESC = {
#     0: "Tool descriptions for message_decoder",
#     1: "Tool descriptions for cryptobotanists_plant_dna_sequencer",
#     2: "Tool descriptions for trade_calculator",
#     3: "Tool descriptions for travel_itinerary_planning",
#     4: "Available tools: click_url(url: str), go_to_previous_page(), scroll_down(), scroll_up(), view(), calculator(expression: str)"
# }

# Mapping from task type index to name
TASK_TYPE_MAP: Dict[int, str] = {
    0: 'message_decoder',
    1: 'cryptobotanists_plant_dna_sequencer',
    2: 'trade_calculator',
    3: 'travel_itinerary_planning',
    4: 'web_browsing',
}

# Known valid tool names for specific task types based on analysis
# Analysis Result 1 specifically mentioned tools for web_browsing (task_type_idx=4)
KNOWN_TOOLS: Dict[int, List[str]] = {
    4: ['click_url', 'go_to_previous_page', 'scroll_down', 'scroll_up', 'view', 'calculator']
    # Add lists for other task types here if future analyses identify similar issues
}

# Known tools that take NO arguments for specific task types (for Analysis Result 4)
NO_ARG_TOOLS: Dict[int, Set[str]] = {
    4: {'go_to_previous_page', 'scroll_down', 'scroll_up', 'view'}
    # Add other task types and their no-argument tools if needed
}


def InferRules(task_name: str, task_type_idx: int) -> str:
    """
    Contains the rules for environment and task execute logic for different task types.
    Refined based on Analysis Results 2, 3, and 4.
    """
    # Enhanced rule based on Analysis Results 2, 3, and 4
    return (
        "You must invoke only one tool action per turn. "
        "Use the exact format 'Action: tool_name, arguments End Action' if the tool takes arguments. "
        "Note the comma separating the tool name and arguments. "
        "If a tool takes *no* arguments, use the format 'Action: tool_name End Action' (without a comma or arguments). "
        "Do not use function call syntax like 'tool_name(arguments)'. "
        "Example of correct format with arguments: 'Action: click_url, /about End Action'. "
        "Example of correct format without arguments: 'Action: view End Action'."
    )


def WrapStep(env: Any, task_name: str, instruction: str, agent_action: str, logger: logging.Logger) -> Tuple[str, float, bool]:
    """
    Process the agent action:
    1. Check for multiple actions in the response. If found, return an error (Analysis Result 2).
    2. Check if the response is an answer. If yes, pass to env.step.
    3. Check if the response is a single valid action format ('Action: ... End Action').
    4. Parse the single action string.
    5. Check for incorrect function-call syntax like 'tool_name(arguments)' (Analysis Result 3). If found, return a specific format error.
    6. If format is comma-separated or no-arg format, extract tool name and arguments.
       - `arguments` is None if no comma is present after the tool name.
       - `arguments` is a string (possibly empty) if a comma is present.
    7. Clean the tool name (remove trailing '()' - Analysis Result 1).
    8. Determine task type by checking if task_name starts with known base types.
    9. Check if a known no-argument tool is incorrectly invoked with a comma/arguments (Analysis Result 4). If so, return specific error.
    10. Validate the cleaned tool name against known tools for the task type (Analysis Result 1).
    11. Provide informative feedback if validation fails.
    12. Reconstruct the action string using the cleaned tool name and appropriate format (with or without comma/arguments).
    13. Execute the reconstructed action using env.step.
    14. Handle potential errors during env.step execution.
    """
    logger.debug(f"Processing agent action: {agent_action} for task: {task_name}")

    # --- Check for Multiple Actions (Analysis Result 2) ---
    action_blocks = re.findall(r"Action:.*?End Action", agent_action, re.DOTALL | re.IGNORECASE) # Use findall to detect multiple actions, ignore case for Action:
    num_actions = len(action_blocks)

    if num_actions > 1:
        logger.warning(f"Agent provided multiple ({num_actions}) actions in a single response: {agent_action}")
        obs = "Error: Only one tool invocation is allowed per response. Please invoke only one tool at a time, following the format 'Action: tool_name, arguments End Action' or 'Action: tool_name End Action' for tools without arguments. Your request contained multiple actions and none were executed."
        # Return obs with error, 0 reward, and done=False to allow agent correction
        return obs, 0.0, False
    elif num_actions == 0:
        # Handle cases where no 'Action: ... End Action' block is found
        if agent_action.strip().startswith("Answer:"):
             logger.debug("Agent provided an answer, passing directly to env.step.")
             # Pass it to env.step to get the final result.
             try:
                 obs, reward, done = env.step(agent_action)
                 logger.debug(f"env.step result for Answer - obs: {obs}, reward: {reward}, done: {done}")
                 return obs, reward, done
             except Exception as e:
                 logger.error(f"Error during env.step execution for Answer '{agent_action}': {e}", exc_info=True)
                 obs = f"An error occurred while processing your answer: {e}."
                 return obs, 0.0, False # Allow retry if answer processing failed
        else:
            logger.warning(f"Action string '{agent_action}' does not contain a valid 'Action: ... End Action' block or start with 'Answer:'.")
            obs = f"Error: Invalid format. Please use 'Action: tool_name, argument_1 End Action' for tool calls with arguments, 'Action: tool_name End Action' for tool calls without arguments, or 'Answer: <your answer>' to provide the final answer."
            # Returning done=False allows the agent to retry
            return obs, 0.0, False
    else: # num_actions == 1
        # Proceed with processing the single action block
        single_action_str = action_blocks[0]
        logger.debug(f"Detected single action block: {single_action_str}")
        action_match = re.match(r"^\s*Action:\s*(.*?)\s*End Action\s*$", single_action_str, re.DOTALL | re.IGNORECASE) # Match content of the single block

        if not action_match:
             # This case should theoretically not be reached if findall found one block,
             # but added as a safeguard against weird formatting within the block.
             logger.error(f"Could not parse the content of the detected action block: {single_action_str}")
             obs = f"Error: Invalid action format within the action block. Please use 'Action: tool_name, argument_1 End Action' or 'Action: tool_name End Action'."
             return obs, 0.0, False

        action_content = action_match.group(1).strip()
        logger.debug(f"Extracted action content: {action_content}")

        # --- Check for Function-Call Syntax (Analysis Result 3) ---
        func_call_match = re.match(r"^\s*(\w+)\s*\((.*)\)\s*$", action_content)
        if func_call_match:
            tool_name_from_func = func_call_match.group(1)
            args_from_func = func_call_match.group(2)
            logger.warning(f"Detected incorrect function-call syntax: '{action_content}'. Tool: '{tool_name_from_func}', Args: '{args_from_func}'")
            # Provide specific feedback as requested in Analysis Result 3
            # Determine if the tool expects args to provide a better example
            task_type_idx_for_example: Optional[int] = None
            for idx, base_task_name in TASK_TYPE_MAP.items():
                 if task_name.startswith(base_task_name):
                     task_type_idx_for_example = idx
                     break

            is_no_arg_example = False
            if task_type_idx_for_example is not None and task_type_idx_for_example in NO_ARG_TOOLS:
                if tool_name_from_func in NO_ARG_TOOLS[task_type_idx_for_example]:
                    is_no_arg_example = True

            if is_no_arg_example:
                 correct_format_example = f"Action: {tool_name_from_func} End Action"
            elif args_from_func: # If args were provided in the incorrect syntax, show them in the correct one
                 correct_format_example = f"Action: {tool_name_from_func}, {args_from_func} End Action"
            else: # If no args were provided, show generic example or just the tool name
                 correct_format_example = f"Action: {tool_name_from_func}, <arguments> End Action" # Or potentially Action: tool_name End Action if it's known no-arg

            obs = (f"Error: Tool invocation format incorrect. Detected function-call syntax '{action_content}'. "
                   f"Please use the comma-separated format 'Action: tool_name, arguments End Action' or 'Action: tool_name End Action' for tools without arguments. "
                   f"For example, use '{correct_format_example}' instead.")
            return obs, 0.0, False

        # --- If not function-call syntax, proceed with parsing ---
        logger.debug("Action content does not match function-call syntax. Proceeding with parsing.")
        parts = action_content.split(',', 1)
        potential_tool_name = parts[0].strip()
        # arguments is None if no comma exists, otherwise it's the stripped string after the comma (can be empty)
        arguments = parts[1].strip() if len(parts) > 1 else None

        # Clean the tool name (remove potential trailing parentheses added by the agent - Analysis Result 1)
        cleaned_tool_name = potential_tool_name.replace('()', '')
        logger.debug(f"Potential tool name: '{potential_tool_name}', Cleaned tool name: '{cleaned_tool_name}', Arguments: {repr(arguments)}") # Use repr to show None vs ""

        # --- Refined Task Type Identification ---
        task_type_idx: Optional[int] = None
        matched_base_task_name: Optional[str] = None
        for idx, base_task_name in TASK_TYPE_MAP.items():
            if task_name.startswith(base_task_name):
                task_type_idx = idx
                matched_base_task_name = base_task_name
                logger.debug(f"Task name '{task_name}' matched base type '{base_task_name}' (index: {task_type_idx}).")
                break

        # --- Check for Incorrect Format for No-Argument Tools (Analysis Result 4) ---
        is_no_arg_tool = False
        if task_type_idx is not None and task_type_idx in NO_ARG_TOOLS:
            if cleaned_tool_name in NO_ARG_TOOLS[task_type_idx]:
                is_no_arg_tool = True
                logger.debug(f"Tool '{cleaned_tool_name}' identified as a no-argument tool for task type {task_type_idx}.")

        # If it's a known no-argument tool, but arguments is not None (meaning a comma was present), it's an error.
        if is_no_arg_tool and arguments is not None:
            logger.warning(f"Incorrect format for no-argument tool '{cleaned_tool_name}'. A comma or arguments were provided: '{action_content}'")
            obs = (f"Error: The tool '{cleaned_tool_name}' does not take any arguments. "
                   f"Please use the format 'Action: {cleaned_tool_name} End Action' instead of including a comma or arguments.")
            return obs, 0.0, False

        # --- Tool Validation based on identified task type using the CLEANED tool name (Analysis Result 1) ---
        if task_type_idx is not None and task_type_idx in KNOWN_TOOLS:
            valid_tools = KNOWN_TOOLS[task_type_idx]
            logger.debug(f"Valid tools for task type {task_type_idx} ('{matched_base_task_name}'): {valid_tools}")
            if cleaned_tool_name not in valid_tools:
                logger.warning(f"Invalid tool name '{cleaned_tool_name}' (derived from '{potential_tool_name}') used for task '{task_name}'. Valid tools for base type '{matched_base_task_name}': {valid_tools}")
                obs = f"Error: Tool '{cleaned_tool_name}' not recognized. Available tools for this task type ('{matched_base_task_name}') are: {', '.join(valid_tools)}. Please use the format 'Action: tool_name, arguments End Action' or 'Action: tool_name End Action'."
                return obs, 0.0, False
            else:
                logger.debug(f"Cleaned tool name '{cleaned_tool_name}' is valid for task '{task_name}' (base type '{matched_base_task_name}').")
        elif task_type_idx is not None:
             logger.debug(f"Task type {task_type_idx} ('{matched_base_task_name}') identified, but no specific tool validation rules defined in KNOWN_TOOLS. Proceeding without validation.")
        else:
            logger.warning(f"Could not determine base task type for task_name: '{task_name}'. Skipping tool name validation.")

        # --- Reconstruct the action string using the cleaned tool name and appropriate format ---
        # If we are here, the format is considered valid for the tool (either args provided for arg-tool, or no args/comma for no-arg tool)
        if arguments is not None:
            # This case applies to tools that take arguments, and arguments were provided.
            # It also covers cases where a tool might optionally take arguments, and they were provided.
            reconstructed_action_content = f"{cleaned_tool_name}, {arguments}"
        else:
            # This case applies to:
            # 1. Tools that correctly take no arguments (e.g., "view").
            # 2. Tools that might optionally take arguments, but none were provided.
            reconstructed_action_content = cleaned_tool_name

        # Reconstruct using the original casing of "Action:" and "End Action" for consistency with env expectations
        action_prefix_match = re.match(r"^\s*(Action:)\s*", single_action_str, re.IGNORECASE)
        action_suffix_match = re.search(r"\s*(End Action)\s*$", single_action_str, re.IGNORECASE)
        action_prefix = action_prefix_match.group(1) if action_prefix_match else "Action:" # Default just in case
        action_suffix = action_suffix_match.group(1) if action_suffix_match else "End Action" # Default just in case

        reconstructed_agent_action = f"{action_prefix} {reconstructed_action_content} {action_suffix}"

        logger.info(f"Original agent action: '{agent_action}'") # Log the original full response
        logger.info(f"Reconstructed agent action for env.step: '{reconstructed_agent_action}'")


        # If format is correct and tool name (if validated) is valid, proceed with env.step using the RECONSTRUCTED action
        try:
            logger.debug(f"Executing env.step with reconstructed action: {reconstructed_agent_action}")
            obs, reward, done = env.step(reconstructed_agent_action)
            logger.debug(f"env.step result - obs: {obs}, reward: {reward}, done: {done}")
            return obs, reward, done
        except Exception as e:
            # Catch potential errors during env.step execution itself
            logger.error(f"Error during env.step execution for reconstructed action '{reconstructed_agent_action}': {e}", exc_info=True)
            error_message = str(e)
            # Check if the error message indicates the tool wasn't found (common case from original analysis)
            if "could not find tool with name" in error_message.lower():
                 if task_type_idx is not None and task_type_idx in KNOWN_TOOLS:
                     valid_tools = KNOWN_TOOLS[task_type_idx]
                     obs = f"Error executing action: {error_message}. Remember, the available tools for this task type ('{matched_base_task_name}') are: {', '.join(valid_tools)}."
                 else:
                     obs = f"Error executing action: {error_message}. Please ensure you are using a valid tool and format."
            # Check if the error message relates to arguments (might happen if env.step has its own arg parsing)
            elif "argument" in error_message.lower():
                 # Provide specific feedback based on whether it was a no-arg tool or not
                 if is_no_arg_tool:
                      obs = f"An error occurred while executing the action '{cleaned_tool_name}': {e}. This tool should not take arguments. Please use the format 'Action: {cleaned_tool_name} End Action'."
                 else:
                      obs = f"An error occurred while executing the action '{cleaned_tool_name}' with arguments '{arguments}': {e}. Please check your arguments and format ('Action: tool_name, arguments End Action')."
            else:
                 obs = f"An error occurred while executing the action '{cleaned_tool_name}': {e}. Please check your action and arguments."

            # Returning done=False allows the agent to potentially try a different approach
            return obs, 0.0, False