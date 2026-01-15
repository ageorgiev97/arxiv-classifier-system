import numpy as np
import re
from typing import List, Dict, Any, Union, Optional
import logging
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)

# Input validation constants
MAX_ABSTRACT_LENGTH = 10000  # Maximum characters per abstract
MIN_ABSTRACT_LENGTH = 20    # Minimum characters for meaningful classification
MAX_BATCH_SIZE = 100        # Maximum abstracts per batch


def validate_abstracts(texts: List[str]) -> List[str]:
    """
    Validate and sanitize input abstracts.
    
    Args:
        texts: List of abstract strings to validate.
        
    Returns:
        List of sanitized abstract strings.
        
    Raises:
        ValueError: If validation fails (empty input, too long, etc.)
    """
    if not texts:
        raise ValueError("Input list cannot be empty")
    
    if len(texts) > MAX_BATCH_SIZE:
        raise ValueError(
            f"Batch size {len(texts)} exceeds maximum of {MAX_BATCH_SIZE}. "
            "Please split your request into smaller batches."
        )
    
    sanitized = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"Abstract at index {i} must be a string, got {type(text).__name__}")
        
        # Strip whitespace
        text = text.strip()
        
        if not text:
            raise ValueError(f"Abstract at index {i} cannot be empty or whitespace only")
        
        if len(text) < MIN_ABSTRACT_LENGTH:
            raise ValueError(
                f"Abstract at index {i} is too short ({len(text)} chars). "
                f"Minimum length is {MIN_ABSTRACT_LENGTH} characters."
            )
        
        if len(text) > MAX_ABSTRACT_LENGTH:
            raise ValueError(
                f"Abstract at index {i} exceeds maximum length ({len(text)} chars). "
                f"Maximum allowed is {MAX_ABSTRACT_LENGTH} characters."
            )
        
        sanitized.append(text)
    
    return sanitized


class ArxivInferenceEngine:
    """
    The production inference engine for ArXiv article classification.
    
    This class encapsulates the model, the tokenizer, and the post-processing
    logic (thresholding). It is designed to be loaded once at application 
    startup (e.g., in Django's AppConfig).
    
    Supports multiple model types:
    - baseline: TF-IDF/TextVectorization model (raw text input)
    - scibert: SciBERT transformer model (tokenized input)
    - specter: SPECTER transformer model (tokenized input)
    """

    # Model type to tokenizer name mapping
    MODEL_TOKENIZERS = {
        "scibert": "giacomomiolo/scibert_reupload",
        "specter": "allenai/specter",
        "transformer": "giacomomiolo/scibert_reupload",  # Default transformer
    }

    def __init__(
        self, 
        model_path: str, 
        model_type: str = "auto",
        threshold: float = None,
        tokenizer_name: Optional[str] = None
    ):
        """
        Initializes the engine by loading the Keras model.
        
        Args:
            model_path: Path to the Keras model file (.keras or .h5)
            model_type: Type of model ('baseline', 'scibert', 'specter', or 'auto')
                       'auto' will try to detect from model name
            threshold: Prediction threshold for multi-label classification
            tokenizer_name: Optional HuggingFace tokenizer name (auto-detected if not provided)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        self.threshold = threshold or settings.model.threshold
        self.model_path = Path(model_path)
        self.model = None
        self._tokenizer = None
        
        # Validate model path exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Detect model type from filename if 'auto'
        self.model_type = self._detect_model_type(model_type)
        logger.info(f"Detected model type: {self.model_type}")
        
        # Initialize tokenizer for transformer models
        if self.model_type != "baseline":
            self._init_tokenizer(tokenizer_name)
        
        logger.info(f"Loading Keras model from {model_path}...")
        self._load_model(str(model_path))
        
        # Ensure category config is loaded to map IDs to Labels
        if not settings.id2label:
            settings.load_category_config()

    def _detect_model_type(self, model_type: str) -> str:
        """Detect model type from filename or explicit setting."""
        if model_type != "auto":
            return model_type
        
        # Try to detect from filename
        filename = self.model_path.stem.lower()
        if "baseline" in filename:
            return "baseline"
        elif "specter" in filename:
            return "specter"
        elif "scibert" in filename:
            return "scibert"
        else:
            # Default to scibert for unknown transformer models
            logger.warning(f"Could not detect model type from '{filename}', defaulting to 'scibert'")
            return "scibert"

    def _init_tokenizer(self, tokenizer_name: Optional[str] = None):
        """
        Initialize the HuggingFace tokenizer for transformer models.
        
        Raises:
            RuntimeError: If tokenizer loading fails
        """
        try:
            from transformers import AutoTokenizer
            
            # Use provided tokenizer name or detect from model type
            if tokenizer_name:
                name = tokenizer_name
            elif self.model_type in self.MODEL_TOKENIZERS:
                name = self.MODEL_TOKENIZERS[self.model_type]
            else:
                name = settings.model.model_name
            
            logger.info(f"Loading tokenizer: {name}")
            self._tokenizer = AutoTokenizer.from_pretrained(name)
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise RuntimeError(f"Failed to load tokenizer '{name}': {e}") from e

    def _load_model(self, model_path: str):
        """
        Load a Keras model using TensorFlow.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            import tensorflow as tf
            
            # Import custom objects for model loading
            from ..utils import MultiLabelFocalLoss, F1Score
            from ..models import BaselineClassifier, SciBertClassifier, SpecterClassifier
            
            custom_objects = {
                'MultiLabelFocalLoss': MultiLabelFocalLoss,
                'F1Score': F1Score,
                'BaselineClassifier': BaselineClassifier,
                'SciBertClassifier': SciBertClassifier,
                'SpecterClassifier': SpecterClassifier,
            }
            
            # Try loading with custom objects first, then without
            try:
                self.model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects=custom_objects,
                    compile=False
                )
            except Exception:
                self.model = tf.keras.models.load_model(model_path, compile=False)
            
            logger.info("Keras model loaded successfully.")

            # If baseline, load the external vectorizer vocab
            if self.model_type == "baseline":
                self._init_baseline_vectorizer()
            
        except Exception as e:
            logger.error(f"Failed to load Keras model: {e}")
            raise RuntimeError(f"Failed to load model from '{model_path}': {e}") from e

    def _init_baseline_vectorizer(self):
        """
        Load TextVectorization model for baseline.
        
        Raises:
            RuntimeError: If vectorizer loading fails and is required
        """
        import tensorflow as tf
        
        # Assume vectorizer model is named {run_name}_vectorizer_tf in the same dir as model
        vec_path = self.model_path.with_name(f"{self.model_path.stem}_vectorizer_tf")
        
        if not vec_path.exists():
            logger.warning(f"Vectorizer model not found at {vec_path}. Baseline model may fail if inputs are raw text.")
            return

        try:
            logger.info(f"Loading baseline vectorizer from {vec_path}")
            # Load using TFSMLayer for Keras 3 compatibility with SavedModel
            self._vectorizer_module = tf.keras.layers.TFSMLayer(
                str(vec_path),
                call_endpoint='serving_default'
            )
            logger.info("Baseline vectorizer initialized (TFSMLayer).")
            
        except Exception as e:
            logger.error(f"Failed to load baseline vectorizer: {e}")
            raise RuntimeError(f"Failed to load vectorizer from '{vec_path}': {e}") from e

    def predict(self, text: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Performs inference on one or more abstracts.
        
        Args:
            text: A single abstract string or a list of abstract strings.
            
        Returns:
            A list of dictionaries containing predicted categories and scores.
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If inference fails
        """
        if isinstance(text, str):
            text = [text]

        # Validate inputs
        text = validate_abstracts(text)

        try:
            # 1. Preprocessing based on model type
            if self.model_type == "baseline":
                inputs = self._preprocess_baseline(text)
                # 2. Forward Pass for baseline (positional arg)
                predictions = self.model(inputs, training=False)
            else:
                inputs = self._preprocess_transformer(text)
                # 2. Forward Pass for transformer (dict unpacking)
                # Convert tokenizer output to plain dict for Keras
                inputs_dict = {k: v for k, v in inputs.items()}
                predictions = self.model(inputs_dict, training=False)
            
            probabilities = predictions.numpy()

            # 3. Post-processing (Thresholding & Mapping)
            return self._format_results(probabilities)
            
        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}") from e

    def _preprocess_baseline(self, texts: List[str]):
        """Preprocess text for baseline model (raw text input)."""
        import tensorflow as tf
        from src.arxiv_classifier.models.baseline import preprocess_texts_batch
        
        # Apply NLP preprocessing (lemmatization, stop-word removal) to match training
        texts = preprocess_texts_batch(texts, use_lemmatization=True, use_stemming=False, use_stopwords=True)
        
        if hasattr(self, '_vectorizer_module'):
            # Text -> TF-IDF Vector
            with tf.device("/CPU:0"):
                # TFSMLayer expects tensor input
                if not isinstance(texts, tf.Tensor):
                    texts = tf.constant(texts)
                res = self._vectorizer_module(texts)
                if isinstance(res, dict):
                    # TFSMLayer returns a dict, likely {'output_0': tensor}
                    return list(res.values())[0]
                return res
        else:
            # Fallback if no vectorizer (legacy or error)
            logger.warning("No vectorizer loaded for baseline. Passing raw strings (model might fail).")
            return tf.constant(texts)

    def _preprocess_transformer(self, texts: List[str]) -> Dict[str, Any]:
        """Preprocess text for transformer models (tokenization)."""
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.model.max_length,
            return_tensors="tf"
        )
        return encoded

    def _format_results(self, probabilities: np.ndarray) -> List[Dict[str, Any]]:
        """
        Converts probability tensors into human-readable label lists.
        """
        batch_results = []
        
        for probs in probabilities:
            # Find indices above the confidence threshold
            category_indices = np.where(probs >= self.threshold)[0]
            
            # If no category exceeds threshold, return top prediction as fallback
            if len(category_indices) == 0:
                top_idx = np.argmax(probs)
                category_indices = [top_idx]
            
            labels = []
            for idx in category_indices:
                labels.append({
                    "label": settings.id2label.get(idx, f"unknown_{idx}"),
                    "probability": round(float(probs[idx]), 4)
                })
            
            # Sort by highest probability
            labels = sorted(labels, key=lambda x: x['probability'], reverse=True)
            batch_results.append({"predictions": labels})
            
        return batch_results

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self.model is not None