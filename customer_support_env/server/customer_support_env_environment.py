import os
import sys
from uuid import uuid4
from typing import Optional, Dict, Any, List
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Add parent directory to sys.path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from models import CustomerSupportAction, CustomerSupportObservation, CustomerSupportReward, Message
    from tasks import TASKS, TaskDefinition
    from grader import CustomerSupportGrader
except (ImportError, ModuleNotFoundError, ValueError):
    from ..models import CustomerSupportAction, CustomerSupportObservation, CustomerSupportReward, Message
    from ..tasks import TASKS, TaskDefinition
    from ..grader import CustomerSupportGrader


class CustomerSupportEnvironment(Environment):
    """
    A reinforcement learning environment for customer support workflows.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 5

    def __init__(self, task_id: str = "easy_refund"):
        """Initialize the customer support environment with a specific task."""
        self.task_id = task_id
        self.task: TaskDefinition = next((t for t in TASKS if t.id == task_id), TASKS[0])
        self.grader = CustomerSupportGrader(self.task)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.conversation_history: List[Message] = []
        self._is_done = False

    def reset(self, task_id: Optional[str] = None) -> CustomerSupportObservation:
        """
        Reset the environment for a new episode.
        """
        if task_id:
            self.task_id = task_id
            self.task = next((t for t in TASKS if t.id == task_id), TASKS[0])
            self.grader = CustomerSupportGrader(self.task)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.conversation_history = [
            Message(role="user", content=self.task.ticket_text)
        ]
        self._is_done = False

        return CustomerSupportObservation(
            ticket_text=self.task.ticket_text,
            conversation_history=self.conversation_history,
            ticket_type=self.task.ticket_type,
            step_count=0,
            done=False,
            reward=0.0
        )

    def step(self, action: CustomerSupportAction) -> CustomerSupportObservation:
        """
        Execute a step in the environment.
        """
        if self._is_done:
             raise RuntimeError("Episode already finished. Call reset() first.")

        self._state.step_count += 1

        # Add agent response to history
        self.conversation_history.append(
            Message(role="agent", content=action.agent_response)
        )

        # Evaluate the response
        reward_obj = self.grader.evaluate(action, self.conversation_history[:-1], action.is_final_answer)

        # Check termination conditions
        if action.is_final_answer or self._state.step_count >= self.MAX_STEPS:
            self._is_done = True

        # Construct the observation
        obs = CustomerSupportObservation(
            ticket_text=self.task.ticket_text,
            conversation_history=self.conversation_history,
            ticket_type=self.task.ticket_type,
            step_count=self._state.step_count,
            done=self._is_done,
            reward=reward_obj.value,
            metadata={
                "relevance": reward_obj.relevance,
                "correctness": reward_obj.correctness,
                "tone": reward_obj.tone,
                "completeness": reward_obj.completeness,
                "is_final": action.is_final_answer
            }
        )

        return obs

    @property
    def state(self) -> State:
        """
        Get the current environment state.
        """
        return self._state
