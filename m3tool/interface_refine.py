import re
import logging
from typing import Tuple, Any # Assuming Task is defined elsewhere, added Any for env type hint

# Assuming Task class and other necessary components like logger are defined elsewhere
# Define Task type hint placeholder if needed
class Task: # Placeholder definition
    def step(self, action: str) -> Tuple[str, float, bool]:
        # This is a placeholder implementation for env.step
        # In a real scenario, this would interact with the specific task environment
        if "Action:" in action and "End Action" in action and not action.strip().startswith("Action: Action:"):
             return f"Executed: {action}", 0.0, False
        elif action.strip().startswith("Answer:"):
             return "Final answer received.", 1.0, True
        else:
             # Default response for invalid format or non-action/answer strings
             # This might be where the original environment handled errors implicitly
             return "Invalid action format. Use 'Action: <tool_name>, <args> End Action' or 'Answer: <your answer>'.", 0.0, False

    def reset(self):
        pass

    @property
    def name(self) -> str:
        return "Sample Task"

    @property
    def instruction(self) -> str:
        return "This is a sample instruction."

# Placeholder for logger if not passed explicitly
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def InferRules(task_name: str, task_type_idx: int) -> str:
    """
    Contains the rules for environment and task execute logic for different task types.
    """
    # No changes needed based on the analysis
    return "There is no rule for this environment."

def WrapStep(env: Any, task_name: str, instruction: str, agent_action: str, logger: logging.Logger) -> Tuple[str, float, bool]:
    """
    Process the agent action: check for specific format errors identified by AnalysisAgent
    before potentially calling env.step, and return the next observation, reward, and done status.

    Args:
        env: The task environment instance.
        task_name: The name of the task.
        instruction: The task instruction.
        agent_action: The action string output by the agent.
        logger: Logger instance for debugging.

    Returns:
        A tuple containing the observation (str), reward (float), and done status (bool).
    """
    processed_action = agent_action.strip()

    # Check for the specific misalignment: double "Action:" prefix
    # This addresses Analysis Result 1 and 2 regarding "Action: Action:"
    if processed_action.startswith("Action: Action:"):
        logger.debug(f"Detected double 'Action:' prefix in agent action: {agent_action}")
        # Provide specific feedback as requested by the analysis
        obs = "Invalid format: double 'Action:' prefix detected. Please use only one 'Action:' prefix per tool call in the format 'Action: <tool_name>, <args> End Action'."
        reward = 0.0
        done = False
        return obs, reward, done
    # Potential check for multiple actions could be added here if needed,
    # but the current analysis focuses specifically on the double prefix.
    # Example: Check if "Action:" appears multiple times *not* at the start.
    # elif processed_action.count("Action:") > 1:
    #     logger.debug(f"Detected multiple 'Action:' instances in agent action: {agent_action}")
    #     obs = "Invalid format: multiple actions detected in a single response. Please invoke only one tool at a time using 'Action: <tool_name>, <args> End Action'."
    #     reward = 0.0
    #     done = False
    #     return obs, reward, done

    # If no specific format errors detected by this function, proceed with env.step
    logger.debug(f"Agent action format check passed. Calling env.step with: {agent_action}")
    obs, reward, done = env.step(agent_action)
    return obs, reward, done

# Example Usage (for demonstration purposes)
if __name__ == '__main__':
    # Mock environment and logger
    mock_env = Task()
    mock_logger = logging.getLogger("MockLogger")
    mock_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    mock_logger.addHandler(handler)


    # --- Test Case 1: Double Action Prefix ---
    agent_action_double = " Action: Action: find_maximum, 1, 2 End Action "
    print(f"--- Testing Double Action Prefix ---")
    print(f"Agent Action: '{agent_action_double}'")
    obs, reward, done = WrapStep(mock_env, "trade_calculator", "Find max", agent_action_double, mock_logger)
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print("-" * 20)

    # --- Test Case 2: Valid Action ---
    agent_action_valid = "Action: find_minimum, 5, 3 End Action"
    print(f"--- Testing Valid Action ---")
    print(f"Agent Action: '{agent_action_valid}'")
    obs, reward, done = WrapStep(mock_env, "trade_calculator", "Find min", agent_action_valid, mock_logger)
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print("-" * 20)

    # --- Test Case 3: Final Answer ---
    agent_action_answer = "Answer: 42"
    print(f"--- Testing Final Answer ---")
    print(f"Agent Action: '{agent_action_answer}'")
    obs, reward, done = WrapStep(mock_env, "trade_calculator", "Give answer", agent_action_answer, mock_logger)
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print("-" * 20)

    # --- Test Case 4: Invalid Format (handled by mock env.step) ---
    agent_action_invalid = "find_maximum, 1, 2"
    print(f"--- Testing Invalid Format (handled by env.step) ---")
    print(f"Agent Action: '{agent_action_invalid}'")
    obs, reward, done = WrapStep(mock_env, "trade_calculator", "Find max", agent_action_invalid, mock_logger)
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print("-" * 20)

    # --- Test Case 5: Get Environment Rule ---
    rule = InferRules("trade_calculator", 2)
    print(f"--- Testing Get Environment Rule ---")
    print(f"Rule: {rule}")
    print("-" * 20)