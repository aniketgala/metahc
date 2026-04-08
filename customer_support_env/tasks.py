from pydantic import BaseModel
from typing import List, Literal

class TaskDefinition(BaseModel):
    id: str
    ticket_text: str
    ticket_type: Literal["billing", "technical", "refund", "general"]
    expected_resolution: str
    success_criteria: List[str]

TASKS = [
    TaskDefinition(
        id="easy_refund",
        ticket_text="I accidentally bought two subscriptions of the basic plan. I only need one. Can I get a refund for the second one?",
        ticket_type="refund",
        expected_resolution="The agent should offer a refund for the duplicate subscription.",
        success_criteria=["refund", "duplicate", "subscription"]
    ),
    TaskDefinition(
        id="medium_delayed_order",
        ticket_text="My order #12345 was supposed to arrive yesterday, but the tracking still says 'Processing'. I'm very frustrated as this was a gift.",
        ticket_type="billing",
        expected_resolution="The agent should apologize, explain the delay, and offer a partial refund or discount for the inconvenience.",
        success_criteria=["apology", "delay", "partial refund", "discount"]
    ),
    TaskDefinition(
        id="hard_technical_reasoning",
        ticket_text="I'm trying to set up the API integration, but I keep getting a '403 Forbidden' error even though I've confirmed my API key is correct. I've also checked my firewall settings and they seem fine.",
        ticket_type="technical",
        expected_resolution="The agent should ask if the user has enabled the specific API permissions in the dashboard or if they are using the correct endpoint (sandbox vs production).",
        success_criteria=["permissions", "dashboard", "endpoint", "403"]
    )
]
