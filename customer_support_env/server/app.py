# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import sys
from openenv.core.env_server.http_server import create_app

# Add parent directory to sys.path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from models import CustomerSupportAction, CustomerSupportObservation
    from server.customer_support_env_environment import CustomerSupportEnvironment
except (ImportError, ModuleNotFoundError, ValueError):
    from ..models import CustomerSupportAction, CustomerSupportObservation
    from .customer_support_env_environment import CustomerSupportEnvironment

app = create_app(
    CustomerSupportEnvironment,
    CustomerSupportAction,
    CustomerSupportObservation,
    env_name="customer_support_env",
)

# Health check to allow HF Space to transition to "Running" state
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Customer Support Env is active and running!"}

def main():
    import uvicorn
    # Use port 7860 as preferred by Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
