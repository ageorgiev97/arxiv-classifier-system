#!/usr/bin/env python
"""
Test script to verify Django API works with pre-trained models.

This script tests loading and inference for each model type.
Models must already exist in the artifacts directory.

Usage:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v --model-type scibert
"""
import os
import sys
import subprocess
import time
from pathlib import Path

import pytest
import requests


class TestModelLoading:
    """Tests for direct model loading without Django."""
    
    @pytest.fixture
    def model_configs(self, request, artifacts_dir):
        """Get model configurations based on command line option."""
        model_type = request.config.getoption("--model-type", default="all")
        
        if model_type == "all":
            models = [
                ("baseline", "test_baseline"),
                ("scibert", "test_scibert"),
                ("specter", "test_specter"),
            ]
        else:
            models = [(model_type, f"test_{model_type}")]
        
        return [(t, artifacts_dir / f"{n}.keras") for t, n in models]
    
    @pytest.mark.parametrize("model_type,run_name", [
        ("baseline", "test_baseline"),
        ("scibert", "test_scibert"),
        ("specter", "test_specter"),
    ])
    def test_model_loading_directly(self, model_type, run_name, artifacts_dir, test_abstract, project_root):
        """Test loading the model directly without Django."""
        model_path = artifacts_dir / f"{run_name}.keras"
        
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        
        from src.arxiv_classifier.inference import ArxivInferenceEngine
        
        engine = ArxivInferenceEngine(
            model_path=str(model_path),
            model_type=model_type,
        )
        
        assert engine.model_type == model_type
        
        # Test inference
        results = engine.predict([test_abstract])
        
        assert results is not None
        assert len(results) == 1
        assert "predictions" in results[0]
        assert len(results[0]["predictions"]) > 0
        
        # Verify prediction structure
        pred = results[0]["predictions"][0]
        assert "label" in pred
        assert "probability" in pred
        assert 0 <= pred["probability"] <= 1


class TestDjangoAPI:
    """Tests for Django API endpoints."""
    
    @pytest.fixture
    def server_process(self, request, artifacts_dir, backend_dir):
        """Start Django server for testing."""
        model_type = request.param
        run_name = f"test_{model_type}"
        model_path = artifacts_dir / f"{run_name}.keras"
        
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        
        env = os.environ.copy()
        env["MODEL_PATH"] = str(model_path)
        env["MODEL_TYPE"] = model_type
        env["DEBUG"] = "True"
        
        process = subprocess.Popen(
            [sys.executable, "manage.py", "runserver", "8000", "--noreload"],
            cwd=backend_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # Wait for server to be ready
        start_time = time.time()
        server_ready = False
        
        while time.time() - start_time < 180:
            if process.poll() is not None:
                break
            
            try:
                resp = requests.get("http://localhost:8000/api/v1/health/", timeout=5)
                if resp.status_code == 200 and resp.json().get("model_loaded"):
                    server_ready = True
                    break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass
            
            time.sleep(2)
        
        if not server_ready:
            process.terminate()
            pytest.fail(f"Server did not start for {model_type}")
        
        yield process, model_type
        
        process.terminate()
        process.wait()
        time.sleep(2)
    
    @pytest.mark.parametrize("server_process", ["baseline", "scibert", "specter"], indirect=True)
    def test_health_endpoint(self, server_process):
        """Test the health check endpoint."""
        process, model_type = server_process
        
        resp = requests.get("http://localhost:8000/api/v1/health/", timeout=10)
        
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("model_loaded") is True
        assert data.get("model_type") == model_type
    
    @pytest.mark.parametrize("server_process", ["baseline", "scibert", "specter"], indirect=True)
    def test_predict_endpoint(self, server_process, test_abstract):
        """Test the prediction endpoint."""
        process, model_type = server_process
        
        resp = requests.post(
            "http://localhost:8000/api/v1/predict/",
            json={"abstracts": [test_abstract]},
            timeout=60
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1
        
        results = data["results"][0]
        assert "predictions" in results
        
        predictions = results["predictions"]
        assert len(predictions) > 0
        
        # Verify prediction structure
        for pred in predictions[:3]:
            assert "label" in pred
            assert "probability" in pred
            assert 0 <= pred["probability"] <= 1


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--model-type",
        action="store",
        default="all",
        choices=["baseline", "scibert", "specter", "all"],
        help="Model type to test (default: all)"
    )
    parser.addoption(
        "--direct-only",
        action="store_true",
        help="Only test direct model loading, skip Django"
    )
    parser.addoption(
        "--api-only",
        action="store_true",
        help="Only test API endpoint, skip direct model loading"
    )
