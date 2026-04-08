---
title: Customer Support Env
emoji: 📢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Customer Support OpenEnv Environment
(v1.0.5 - final port sync 7860)

A production-grade reinforcement learning environment for simulating real-world customer support workflows.

## Motivation
In the real world, customer support agents must resolve complex issues by interacting with users, asking clarifying questions, and providing accurate resolutions while maintaining a professional tone. This environment allows AI agents to be trained and evaluated on their ability to handle these multi-step interactions effectively.

## Environment Specification

### Observation Space
- `ticket_text` (string): The initial issue reported by the customer.
- `conversation_history` (list): A list of messages between the user and the agent.
- `ticket_type` (billing, technical, refund, general): The category of the ticket.
- `step_count` (int): Current step in the episode (max 5).

### Action Space
- `agent_response` (string): The text response sent to the customer.
- `is_final_answer` (boolean): Flag indicating if the issue is fully resolved.

### Reward Function
The reward is a weighted sum of four components, each between 0.1 and 0.95:
- **Relevance (0.3)**: How well the response matches the ticket type and content.
- **Correctness (0.3)**: Whether the response addresses the success criteria of the task.
- **Tone (0.2)**: Professionalism and empathy of the response.
- **Completeness (0.2)**: Whether the response is a final resolution or a helpful clarifying question.

The final reward is strictly clamped between **0.01 and 0.99**.

## Tasks
1.  **Easy Refund Request (`easy_refund`)**: A duplicate subscription refund.
2.  **Medium Delayed Order (`medium_delayed_order`)**: A late delivery requiring apology and partial refund.
3.  **Hard Technical Reasoning (`hard_technical_reasoning`)**: A 403 Forbidden error requiring API permission troubleshooting.

## Setup Instructions
1.  Install dependencies:
    ```bash
    pip install openenv openai pydantic
    ```
2.  Run the server:
    ```bash
    cd customer_support_env
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    ```
3.  Run inference:
    ```bash
    export API_BASE_URL=http://localhost:8000
    export MODEL_NAME=gpt-3.5-turbo
    export HF_TOKEN=your_token
    python inference.py
    ```

## Baseline Scores
- `easy_refund`: 0.85
- `medium_delayed_order`: 0.75
- `hard_technical_reasoning`: 0.65

## Validation
Passes `openenv validate` with 100% compliance.
