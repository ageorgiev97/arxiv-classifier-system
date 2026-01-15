from .base import ArxivClassifierBase
import tensorflow as tf
import keras
import re
import logging
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


# Module-level flag to ensure NLTK data is only downloaded once
_nltk_data_downloaded = False


def _ensure_nltk_data():
    """Download required NLTK data if not present."""
    global _nltk_data_downloaded
    if _nltk_data_downloaded:
        return
    
    required = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
    ]
    for path, name in required:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)
    
    _nltk_data_downloaded = True


def preprocess_text(
    text: str,
    use_lemmatization: bool = True,
    use_stemming: bool = False,
    use_stopwords: bool = True,
    min_word_length: int = 2,
    _stopwords_cache: Optional[set] = None,
    _lemmatizer: Optional[WordNetLemmatizer] = None,
    _stemmer: Optional[PorterStemmer] = None,
) -> str:
    """
    Preprocess a single text string for TF-IDF vectorization.
    
    Applies stop-word removal, lemmatization, and optional stemming.
    
    Args:
        text: Input text to preprocess
        use_lemmatization: Whether to apply lemmatization (default: True)
        use_stemming: Whether to apply stemming (default: False)
        use_stopwords: Whether to remove stop words (default: True)
        min_word_length: Minimum word length to keep (default: 2)
        
    Returns:
        Preprocessed text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Build stop words set (use cache if provided)
    if use_stopwords:
        if _stopwords_cache is not None:
            stop_set = _stopwords_cache
        else:
            stop_set = set(stopwords.words('english'))
            # Add scientific paper common words
            stop_set.update([
                'paper', 'study', 'method', 'result', 'show', 'propose',
                'proposed', 'approach', 'work', 'present', 'use', 'using',
                'used', 'based', 'new', 'novel', 'also', 'however', 'therefore',
                'et', 'al', 'fig', 'figure', 'table', 'section', 'equation',
            ])
    else:
        stop_set = set()
    
    # Initialize tools (use cache if provided)
    lemmatizer = _lemmatizer if _lemmatizer else (WordNetLemmatizer() if use_lemmatization else None)
    stemmer = _stemmer if _stemmer else (PorterStemmer() if use_stemming else None)
    
    # Process tokens
    processed_tokens = []
    for token in tokens:
        # Skip short words
        if len(token) < min_word_length:
            continue
        
        # Skip stop words
        if use_stopwords and token in stop_set:
            continue
        
        # Lemmatize
        if lemmatizer:
            token = lemmatizer.lemmatize(token, pos='v')  # Verb form
            token = lemmatizer.lemmatize(token, pos='n')  # Noun form
        
        # Stem (after lemmatization if both enabled)
        if stemmer:
            token = stemmer.stem(token)
        
        processed_tokens.append(token)
    
    return ' '.join(processed_tokens)


def preprocess_texts_batch(
    texts: List[str],
    use_lemmatization: bool = True,
    use_stemming: bool = False,
    use_stopwords: bool = True,
) -> List[str]:
    """
    Preprocess a batch of texts efficiently with shared NLTK resources.
    
    Args:
        texts: List of input texts
        use_lemmatization: Whether to apply lemmatization
        use_stemming: Whether to apply stemming
        use_stopwords: Whether to remove stop words
        
    Returns:
        List of preprocessed texts
    """
    _ensure_nltk_data()
    
    # Create shared resources for efficiency
    stop_set = None
    if use_stopwords:
        stop_set = set(stopwords.words('english'))
        stop_set.update([
            'paper', 'study', 'method', 'result', 'show', 'propose',
            'proposed', 'approach', 'work', 'present', 'use', 'using',
            'used', 'based', 'new', 'novel', 'also', 'however', 'therefore',
            'et', 'al', 'fig', 'figure', 'table', 'section', 'equation',
        ])
    
    lemmatizer = WordNetLemmatizer() if use_lemmatization else None
    stemmer = PorterStemmer() if use_stemming else None
    
    return [
        preprocess_text(
            text,
            use_lemmatization=use_lemmatization,
            use_stemming=use_stemming,
            use_stopwords=use_stopwords,
            _stopwords_cache=stop_set,
            _lemmatizer=lemmatizer,
            _stemmer=stemmer,
        )
        for text in texts
    ]


@keras.saving.register_keras_serializable(package="arxiv_classifier")
class BaselineClassifier(ArxivClassifierBase):
    """
    Simple MLP classifier that expects pre-vectorized TF-IDF inputs.
    TextVectorization is handled outside the model to ensure reliable serialization.
    
    Use preprocess_texts_batch() or preprocess_text() to preprocess raw text
    before passing to TextVectorization for TF-IDF feature extraction.
    """
    def __init__(self, num_classes, input_dim=20000, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.input_dim = input_dim
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        # Initialize with negative bias so sigmoid starts near 0 (sparse labels)
        self.classifier = tf.keras.layers.Dense(
            num_classes, 
            activation='sigmoid',
            bias_initializer=tf.keras.initializers.Constant(-2.0)
        )

    def call(self, inputs, training=False):
        # inputs is already a float tensor of TF-IDF vectors
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.classifier(x)

    def get_model_type(self): 
        return "baseline"

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)