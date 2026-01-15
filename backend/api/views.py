"""
API Views for ArXiv Classifier inference.

These are thin views that delegate actual inference to the engine loaded at startup.
"""
import logging
from django.apps import apps
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from .serializers import (
    PredictRequestSerializer,
    PredictResponseSerializer,
    HealthResponseSerializer,
)

logger = logging.getLogger(__name__)


class PredictView(APIView):
    """
    Endpoint for classifying research article abstracts.
    
    POST /api/v1/predict/
    
    Request body:
    {
        "abstracts": ["Abstract text 1...", "Abstract text 2..."]
    }
    
    Response:
    {
        "results": [
            {
                "abstract_preview": "Abstract text 1...",
                "predictions": [
                    {"label": "cs.LG", "probability": 0.9234},
                    {"label": "stat.ML", "probability": 0.7821}
                ]
            }
        ],
        "model_version": "scibert-baseline-v1"
    }
    """
    
    def post(self, request):
        """Handle prediction requests."""
        # Validate input
        serializer = PredictRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"error": "Invalid request", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get the inference engine
        engine = apps.get_app_config('api').inference_engine
        if engine is None:
            logger.error("Inference engine not available")
            return Response(
                {"error": "Model not loaded. Please try again later."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        # Run inference
        abstracts = serializer.validated_data['abstracts']
        try:
            raw_results = engine.predict(abstracts)
        except ValueError as e:
            # Input validation errors (bad input from user)
            logger.warning(f"Validation error: {e}")
            return Response(
                {"error": "Validation error", "details": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except RuntimeError as e:
            # Model/inference errors (internal issues)
            logger.exception("Inference runtime error")
            return Response(
                {"error": "Inference failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            # Unexpected errors
            logger.exception("Unexpected inference error")
            return Response(
                {"error": "Internal server error", "details": "An unexpected error occurred"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Format response
        results = []
        for abstract, result in zip(abstracts, raw_results):
            results.append({
                "abstract_preview": abstract[:200] + "..." if len(abstract) > 200 else abstract,
                "predictions": result.get("predictions", [])
            })
        
        # Include model type in response
        model_type = getattr(engine, 'model_type', 'unknown')
        response_data = {
            "results": results,
            "model_version": f"{model_type}-v1"
        }
        
        # Validate output format
        response_serializer = PredictResponseSerializer(data=response_data)
        response_serializer.is_valid(raise_exception=True)
        
        return Response(response_serializer.validated_data)


class HealthView(APIView):
    """
    Health check endpoint.
    
    GET /api/v1/health/
    
    Returns the status of the API and whether the model is loaded.
    """
    
    def get(self, request):
        """Return health status."""
        engine = apps.get_app_config('api').inference_engine
        
        model_type = getattr(engine, 'model_type', None) if engine else None
        response_data = {
            "status": "healthy" if engine is not None else "degraded",
            "model_loaded": engine is not None,
            "model_type": model_type,
            "version": "1.0.0"
        }
        
        serializer = HealthResponseSerializer(data=response_data)
        serializer.is_valid(raise_exception=True)
        
        status_code = status.HTTP_200_OK if engine else status.HTTP_503_SERVICE_UNAVAILABLE
        return Response(serializer.validated_data, status=status_code)
