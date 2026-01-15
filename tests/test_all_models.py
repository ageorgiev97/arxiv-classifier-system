#!/usr/bin/env python
"""
Test script to train all 3 model types and verify they work with the Django API.

This script:
1. Trains baseline, scibert, and specter models for quick testing (1 epoch, 100 samples)
2. Saves each model as .keras
3. Tests loading each model via the Django API

Usage:
    pytest tests/test_all_models.py -v
    pytest tests/test_all_models.py::TestTraining -v  # Training only
    pytest tests/test_all_models.py::TestAPIIntegration -v  # API only
"""
import os
import sys
import subprocess
import time
from pathlib import Path

import pytest
import requests


class TestTraining:
    """Tests for model training."""
    
    @pytest.fixture(autouse=True)
    def setup(self, project_root, artifacts_dir):
        """Setup test fixtures."""
        self.project_root = project_root
        self.artifacts_dir = artifacts_dir
    
    @pytest.mark.parametrize("model_type,run_name", [
        ("baseline", "test_baseline"),
        ("scibert", "test_scibert"),
        ("specter", "test_specter"),
    ])
    def test_training(self, model_type, run_name):
        """Train a model with minimal config for testing."""
        cmd = [
            sys.executable,
            str(self.project_root / "scripts" / "train.py"),
            "--model_type", model_type,
            "--run_name", run_name,
            "--epochs", "1",
            "--batch_size", "32",
            "--limit_samples", "100",
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        assert result.returncode == 0, f"Training failed for {model_type}"
        
        model_path = self.artifacts_dir / f"{run_name}.keras"
        assert model_path.exists(), f"Model not saved: {model_path}"


class TestAPIIntegration:
    """Integration tests for Django API with trained models."""
    
    @pytest.fixture
    def server_process(self, request, artifacts_dir, backend_dir):
        """Start Django server with specified model."""
        model_type, run_name = request.param
        model_path = artifacts_dir / f"{run_name}.keras"
        
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}. Run training tests first.")
        
        env = os.environ.copy()
        env["MODEL_PATH"] = str(model_path)
        env["MODEL_TYPE"] = model_type
        
        process = subprocess.Popen(
            [sys.executable, "manage.py", "runserver", "8000", "--noreload"],
            cwd=backend_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for server to start
        time.sleep(10)
        
        yield process, model_type
        
        process.terminate()
        process.wait()
        time.sleep(2)
    
    @pytest.mark.parametrize("server_process", [
        ("baseline", "test_baseline"),
        ("scibert", "test_scibert"),
        ("specter", "test_specter"),
    ], indirect=True)
    def test_health_check(self, server_process):
        """Test health check endpoint with model loaded."""
        process, model_type = server_process
        
        max_retries = 15
        for attempt in range(max_retries):
            try:
                resp = requests.get("http://localhost:8000/api/v1/health/", timeout=5)
                data = resp.json()
                
                if data.get("model_loaded"):
                    assert data.get("model_type") == model_type
                    return
                    
            except requests.ConnectionError:
                pass
            
            if attempt == max_retries - 1:
                pytest.fail(f"Model not loaded after {max_retries * 2}s")
            
            time.sleep(2)
    
    @pytest.mark.parametrize("server_process", [
        ("baseline", "test_baseline"),
        ("scibert", "test_scibert"),
        ("specter", "test_specter"),
    ], indirect=True)
    def test_prediction(self, server_process, test_abstract):
        """Test prediction endpoint."""
        process, model_type = server_process
        
        # Wait for model to load
        max_retries = 15
        for attempt in range(max_retries):
            try:
                resp = requests.get("http://localhost:8000/api/v1/health/", timeout=5)
                if resp.json().get("model_loaded"):
                    break
            except (requests.ConnectionError, requests.exceptions.Timeout):
                pass
            
            if attempt == max_retries - 1:
                pytest.fail("Model not loaded in time")
            
            time.sleep(2)
        
        # Test prediction
        resp = requests.post(
            "http://localhost:8000/api/v1/predict/",
            json={"abstracts": [test_abstract]},
            timeout=30
        )
        
        assert resp.status_code == 200
        
        data = resp.json()
        assert "results" in data
        
        results = data["results"][0]
        predictions = results.get("predictions", [])
        
        assert len(predictions) > 0
        
        # Verify prediction format
        for pred in predictions[:3]:
            assert "label" in pred
            assert "probability" in pred


class TestEndToEnd:
    """End-to-end tests: train and then test API."""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,run_name", [
        ("baseline", "test_baseline"),
        ("scibert", "test_scibert"),
        ("specter", "test_specter"),
    ])
    def test_train_and_serve(self, model_type, run_name, project_root, artifacts_dir, 
                              backend_dir, test_abstract):
        """Train a model and test it via the API."""
        # Train
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "train.py"),
            "--model_type", model_type,
            "--run_name", run_name,
            "--epochs", "1",
            "--batch_size", "32",
            "--limit_samples", "100",
        ]
        
        result = subprocess.run(cmd, cwd=project_root)
        assert result.returncode == 0, f"Training failed for {model_type}"
        
        model_path = artifacts_dir / f"{run_name}.keras"
        assert model_path.exists()
        
        # Start server
        env = os.environ.copy()
        env["MODEL_PATH"] = str(model_path)
        env["MODEL_TYPE"] = model_type
        
        server = subprocess.Popen(
            [sys.executable, "manage.py", "runserver", "8000", "--noreload"],
            cwd=backend_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        try:
            # Wait for server
            time.sleep(10)
            
            # Wait for model to load
            for _ in range(15):
                try:
                    resp = requests.get("http://localhost:8000/api/v1/health/", timeout=5)
                    if resp.json().get("model_loaded"):
                        break
                except:
                    pass
                time.sleep(2)
            
            # Test prediction
            resp = requests.post(
                "http://localhost:8000/api/v1/predict/",
                json={"abstracts": [test_abstract]},
                timeout=30
            )
            
            assert resp.status_code == 200
            assert "results" in resp.json()
            
        finally:
            server.terminate()
            server.wait()
