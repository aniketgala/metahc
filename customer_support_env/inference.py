import os
import json
import asyncio
import sys
import subprocess
import time
import socket
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in the parent directory as well
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))
load_dotenv()

# Add parent directory to sys.path to allow importing from the package name
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from customer_support_env.client import CustomerSupportEnv, CustomerSupportAction
except (ImportError, ModuleNotFoundError):
    from client import CustomerSupportEnv, CustomerSupportAction

# ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
IMAGE_NAME = "customer_support_env:latest"
MAX_STEPS = 5

# OPENAI CLIENT USAGE
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def is_port_open(host="localhost", port=8000):
    """Check if the environment server port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def start_server():
    """Start the local environment server if not already running."""
    if not is_port_open():
        print("--- Server not running. Starting local environment server... ---")
        server_path = os.path.join(os.path.dirname(__file__), "server", "app.py")
        if not os.path.exists(server_path):
             server_path = os.path.join(os.path.dirname(__file__), "..", "customer_support_env", "server", "app.py")
        
        subprocess.Popen([sys.executable, server_path], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        
        # Wait for server to start
        for _ in range(10):
            if is_port_open():
                print("--- Server started successfully. ---")
                return
            time.sleep(1)
        print("--- Warning: Server startup timed out. ---")

async def run_task(task_id: str):
    """Run a single customer support task."""
    # Print [START]
    print(f"[START] task={task_id} env=customer_support_env model={MODEL_NAME}")
    
    # Initialize environment client
    env = await CustomerSupportEnv.from_docker_image(IMAGE_NAME, base_url="http://localhost:8000")
    
    try:
        # ✅ FIX: Await env.reset() and then access .observation
        reset_result = await env.reset(task_id=task_id)
        obs = reset_result.observation
        
        rewards = []
        step_idx = 0
        done = False
        error_msg = "null"
        
        while not done and step_idx < MAX_STEPS:
            step_idx += 1
            
            # Prepare prompts
            history_str = "\n".join([f"{m.role}: {m.content}" for m in obs.conversation_history])
            last_reward = obs.reward if step_idx > 1 else 0.0
            
            system_prompt = "You are a customer support agent. Your goal is to resolve the user's issue effectively, politely, and completely."
            user_prompt = f"Ticket Text: {obs.ticket_text}\n\nConversation History:\n{history_str}\n\nLast Reward: {last_reward:.2f}\n\nProvide your response as a JSON object with 'agent_response' (string) and 'is_final_answer' (boolean)."
            
            try:
                # Call LLM with fail-safe
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                agent_response = data.get("agent_response", "I will help you with this issue.")
                is_final_answer = data.get("is_final_answer", False)
                
            except Exception as e:
                agent_response = "I will help you with this issue."
                is_final_answer = False
                error_msg = str(e).replace("\n", " ")
            
            # ✅ FIX: Await env.step() and then access observation/reward/done
            action = CustomerSupportAction(agent_response=agent_response, is_final_answer=is_final_answer)
            step_result = await env.step(action)
            
            obs = step_result.observation
            reward = step_result.reward
            done = step_result.done
            rewards.append(reward)
            
            # Print [STEP]
            clean_action = agent_response.replace("\n", " ").replace("\r", "")
            print(f"[STEP] step={step_idx} action={clean_action} reward={reward:.2f} done={'true' if done else 'false'} error={error_msg if error_msg != 'null' else 'null'}")
            
            if done:
                break
        
        # SCORING
        max_possible_reward = float(MAX_STEPS)
        score = sum(rewards) / max_possible_reward
        score = min(max(score, 0.0), 1.0)
        
        # Print [END]
        print(f"[END] success={'true' if done else 'false'} steps={step_idx} score={score:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}")
        
    finally:
        # ✅ FIX: Await env.close()
        if env:
            await env.close()

async def main():
    """Main execution loop."""
    start_server()
    tasks = ["easy_refund", "medium_delayed_order", "hard_technical_reasoning"]
    for tid in tasks:
        await run_task(tid)

if __name__ == "__main__":
    # ✅ FIX: Entry point uses asyncio.run()
    asyncio.run(main())
