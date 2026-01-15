"""
Pytest fixtures and configuration for ArXiv Classifier tests.
"""
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BACKEND_DIR = PROJECT_ROOT / "backend"

# Test abstract for inference
TEST_ABSTRACT = """
We present a novel deep learning approach for quantum computing simulation. 
Our method combines transformer architectures with quantum circuit optimization 
to achieve state-of-the-art performance on variational quantum eigensolvers.
"""


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def artifacts_dir() -> Path:
    """Return the artifacts directory."""
    return ARTIFACTS_DIR


@pytest.fixture
def backend_dir() -> Path:
    """Return the backend directory."""
    return BACKEND_DIR


@pytest.fixture
def test_abstract() -> str:
    """Return a test abstract for inference."""
    return TEST_ABSTRACT


@pytest.fixture
def django_server(request):
    """
    Fixture to start/stop Django server with a model.
    
    Usage:
        @pytest.mark.parametrize("django_server", [("path", "model_type")], indirect=True)
        def test_api(django_server):
            ...
    """
    model_path, model_type = request.param
    
    env = os.environ.copy()
    env["MODEL_PATH"] = str(model_path)
    env["MODEL_TYPE"] = model_type
    env["DEBUG"] = "True"
    
    process = subprocess.Popen(
        [sys.executable, "manage.py", "runserver", "8000", "--noreload"],
        cwd=BACKEND_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Wait for server to be ready
    start_time = time.time()
    server_ready = False
    
    while time.time() - start_time < 180:  # 180 second timeout for transformer models
        if process.poll() is not None:
            break
        
        try:
            resp = requests.get("http://localhost:8000/api/v1/health/", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("model_loaded"):
                    server_ready = True
                    break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        
        time.sleep(2)
    
    if not server_ready:
        process.terminate()
        pytest.fail(f"Server did not start in time for {model_type}")
    
    yield process
    
    # Cleanup
    process.terminate()
    process.wait()
    time.sleep(2)
