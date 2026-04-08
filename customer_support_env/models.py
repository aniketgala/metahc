from openenv.core.env_server.types import Action, Observation, BaseModel
from pydantic import Field
from typing import List, Optional, Literal


class Message(BaseModel):
    """A message in the conversation history."""
    role: Literal["user", "agent"] = Field(..., description="Role of the sender")
    content: str = Field(..., description="Content of the message")


class CustomerSupportObservation(Observation):
    """Observation for the customer support environment."""
    ticket_text: str = Field(..., description="Initial customer issue")
    conversation_history: List[Message] = Field(default_factory=list, description="History of messages in the episode")
    ticket_type: Literal["billing", "technical", "refund", "general"] = Field(..., description="Type of the ticket")
    step_count: int = Field(default=0, description="Current step count in the episode")


class CustomerSupportAction(Action):
    """Action for the customer support environment."""
    agent_response: str = Field(..., description="The response from the AI agent")
    is_final_answer: bool = Field(default=False, description="Whether this is the final resolution")


class CustomerSupportReward(BaseModel):
    """Reward for the customer support environment."""
    relevance: float = Field(..., description="Relevance of the response (0.1 to 0.95)")
    correctness: float = Field(..., description="Correctness of the response (0.1 to 0.95)")
    tone: float = Field(..., description="Tone of the response (0.1 to 0.95)")
    completeness: float = Field(..., description="Completeness of the response (0.1 to 0.95)")
    value: float = Field(..., description="Final weighted and clamped reward (0.01 to 0.99)")
