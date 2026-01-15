"""
API Application Configuration.

This module handles model warming - loading the ML model once at Django startup
to avoid repeated loading overhead on each request.
"""
import logging
from django.apps import AppConfig
from django.conf import settings as django_settings

logger = logging.getLogger(__name__)


class ApiConfig(AppConfig):
    """
    Django App Configuration for the API.
    
    The inference engine is loaded as a class attribute (singleton pattern)
    when Django starts, making it available to all views without reloading.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    
    # Singleton reference to the inference engine
    inference_engine = None
    
    def ready(self):
        """
        Called when Django starts. Loads the model into memory.
        
        This runs once per process, so the model is shared across all requests
        in that process. For production with multiple workers, each worker
        will have its own copy of the model.
        """
        # Avoid double-loading during Django's auto-reload in development.
        # When using --noreload, RUN_MAIN is not set, so we should always load.
        # When using auto-reload, RUN_MAIN is 'true' only in the child process.
        import os
        run_main = os.environ.get('RUN_MAIN', None)
        
        # If RUN_MAIN is explicitly 'false', we're in the parent reloader process - skip
        # If RUN_MAIN is 'true', we're in the child process - load
        # If RUN_MAIN is None (--noreload), we should always load
        if run_main == 'false':
            return
        
        self._load_inference_engine()
    
    def _load_inference_engine(self):
        """Load the inference engine with error handling."""
        try:
            from src.arxiv_classifier.inference import ArxivInferenceEngine
            
            model_path = django_settings.MODEL_PATH
            model_type = getattr(django_settings, 'MODEL_TYPE', 'auto')
            tokenizer_name = getattr(django_settings, 'TOKENIZER_NAME', None)
            threshold = django_settings.PREDICTION_THRESHOLD
            
            logger.info(f"Loading inference engine from: {model_path}")
            logger.info(f"Model type: {model_type}, Tokenizer: {tokenizer_name or 'auto-detect'}")
            
            ApiConfig.inference_engine = ArxivInferenceEngine(
                model_path=model_path,
                model_type=model_type,
                threshold=threshold,
                tokenizer_name=tokenizer_name
            )
            
            logger.info(f"✓ Inference engine loaded successfully! (type: {ApiConfig.inference_engine.model_type})")
            
        except Exception as e:
            logger.error(f"✗ Failed to load inference engine: {e}")
            # Don't crash the server - allow it to start but log the error
            # Requests will fail gracefully if engine is None
            ApiConfig.inference_engine = None
