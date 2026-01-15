"""
API Serializers for request/response validation.

These serializers handle input validation and output formatting
for the ArXiv classifier inference API.
"""
from rest_framework import serializers


class PredictRequestSerializer(serializers.Serializer):
    """
    Validates incoming prediction requests.
    
    Accepts a list of abstracts for batch prediction.
    """
    abstracts = serializers.ListField(
        child=serializers.CharField(
            max_length=10000,
            help_text="Research article abstract text"
        ),
        min_length=1,
        max_length=32,  # Reasonable batch limit
        help_text="List of abstracts to classify"
    )
    
    def validate_abstracts(self, value):
        """
        Validate abstracts are meaningful text.
        
        Checks:
        - Not empty or whitespace only
        - Minimum length for meaningful classification
        - No control characters (except newlines/tabs)
        """
        import re
        
        MIN_LENGTH = 50  # Minimum chars for meaningful classification
        # Pattern to detect control characters (except newline, tab, carriage return)
        CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        
        sanitized = []
        for i, abstract in enumerate(value):
            # Strip whitespace
            abstract = abstract.strip()
            
            if not abstract:
                raise serializers.ValidationError(
                    f"Abstract at index {i} cannot be empty or whitespace only."
                )
            
            if len(abstract) < MIN_LENGTH:
                raise serializers.ValidationError(
                    f"Abstract at index {i} is too short ({len(abstract)} chars). "
                    f"Minimum length is {MIN_LENGTH} characters for meaningful classification."
                )
            
            # Check for control characters
            if CONTROL_CHARS.search(abstract):
                raise serializers.ValidationError(
                    f"Abstract at index {i} contains invalid control characters."
                )
            
            sanitized.append(abstract)
        
        return sanitized


class PredictionResultSerializer(serializers.Serializer):
    """
    Represents a single category prediction.
    """
    label = serializers.CharField(help_text="ArXiv category label (e.g., 'cs.LG')")
    probability = serializers.FloatField(
        min_value=0.0,
        max_value=1.0,
        help_text="Confidence score"
    )


class SinglePredictionSerializer(serializers.Serializer):
    """
    Represents predictions for a single abstract.
    """
    abstract_preview = serializers.CharField(
        help_text="Truncated preview of the input abstract"
    )
    predictions = PredictionResultSerializer(
        many=True,
        help_text="List of predicted categories"
    )


class PredictResponseSerializer(serializers.Serializer):
    """
    Response format for batch predictions.
    """
    results = SinglePredictionSerializer(
        many=True,
        help_text="Prediction results for each abstract"
    )
    model_version = serializers.CharField(
        default="scibert-baseline-v1",
        help_text="Version of the model used"
    )


class HealthResponseSerializer(serializers.Serializer):
    """
    Response format for health check endpoint.
    """
    status = serializers.CharField()
    model_loaded = serializers.BooleanField()
    model_type = serializers.CharField(allow_null=True, required=False)
    version = serializers.CharField()
