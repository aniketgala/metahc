# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Customer Support Env Environment Client."""

from typing import Dict, Any, Optional
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import CustomerSupportAction, CustomerSupportObservation, Message
except ImportError:
    from models import CustomerSupportAction, CustomerSupportObservation, Message


class CustomerSupportEnv(
    EnvClient[CustomerSupportAction, CustomerSupportObservation, State]
):
    """
    Client for the Customer Support Env Environment.
    """

    def _step_payload(self, action: CustomerSupportAction) -> Dict:
        """
        Convert CustomerSupportAction to JSON payload for step message.
        """
        return {
            "agent_response": action.agent_response,
            "is_final_answer": action.is_final_answer,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CustomerSupportObservation]:
        """
        Parse server response into StepResult[CustomerSupportObservation].
        """
        obs_data = payload.get("observation", {})
        
        # Parse conversation history
        history_data = obs_data.get("conversation_history", [])
        history = [Message(**msg) for msg in history_data]
        
        observation = CustomerSupportObservation(
            ticket_text=obs_data.get("ticket_text", ""),
            conversation_history=history,
            ticket_type=obs_data.get("ticket_type", "general"),
            step_count=obs_data.get("step_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    @classmethod
    async def from_docker_image(cls, image_name: str, **kwargs) -> "CustomerSupportEnv":
        """
        Mocking the async from_docker_image for compatibility with the requirement.
        In a real scenario, this would start the container.
        For now, we return a client connected to localhost:8000.
        """
        import asyncio
        # Simulate some async work
        await asyncio.sleep(0.1)
        return cls(base_url=kwargs.get("base_url", "http://localhost:8000"))
