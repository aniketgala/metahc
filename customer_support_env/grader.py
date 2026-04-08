from typing import List, Tuple
try:
    from .models import CustomerSupportAction, CustomerSupportReward, Message
    from .tasks import TaskDefinition
except (ImportError, ModuleNotFoundError, ValueError):
    from models import CustomerSupportAction, CustomerSupportReward, Message
    from tasks import TaskDefinition

class CustomerSupportGrader:
    def __init__(self, task: TaskDefinition):
        self.task = task

    def _calculate_component(self, keywords: List[str], text: str) -> float:
        """Calculate a component score between 0.1 and 0.95 based on keyword matching."""
        count = sum(1 for keyword in keywords if keyword.lower() in text.lower())
        if not keywords:
            return 0.5
        score = 0.1 + (count / len(keywords)) * 0.85
        return min(0.95, max(0.1, score))

    def evaluate(self, action: CustomerSupportAction, history: List[Message], is_final: bool) -> CustomerSupportReward:
        # 1. Relevance: Is the response related to the ticket_type and content?
        relevance_keywords = [self.task.ticket_type] + self.task.ticket_text.lower().split()[:3]
        relevance = self._calculate_component(relevance_keywords, action.agent_response)

        # 2. Correctness: Does it address the success criteria?
        correctness = self._calculate_component(self.task.success_criteria, action.agent_response)

        # 3. Tone: Is the language professional and helpful?
        tone_keywords = ["please", "thank you", "apologize", "sorry", "assist", "help", "regret", "understand"]
        tone = self._calculate_component(tone_keywords, action.agent_response)

        # 4. Completeness: Is it a final answer or a useful question?
        # If is_final is True, we check if it addresses all success criteria.
        # If not, we check if it asks a question (useful for multi-step).
        if is_final:
            completeness = correctness # Simplification
        else:
            # If it asks a question (contains '?')
            if "?" in action.agent_response:
                completeness = 0.7
            else:
                completeness = 0.3

        # Weighted sum
        # relevance (0.3), correctness (0.3), tone (0.2), completeness (0.2)
        total_reward = (
            relevance * 0.3 +
            correctness * 0.3 +
            tone * 0.2 +
            completeness * 0.2
        )

        # Intermediate penalty for repeated responses (if any)
        if len(history) > 2:
            last_agent_responses = [m.content for m in history if m.role == "agent"]
            if action.agent_response in last_agent_responses:
                total_reward -= 0.1

        # Strict clamping
        final_reward_value = min(0.99, max(0.01, total_reward))

        return CustomerSupportReward(
            relevance=relevance,
            correctness=correctness,
            tone=tone,
            completeness=completeness,
            value=final_reward_value
        )
