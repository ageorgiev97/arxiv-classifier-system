#!/usr/bin/env python
"""
Evaluation Suite for ArXiv Classifier Models

This script provides comprehensive evaluation across all three model types:
- Baseline (TF-IDF + MLP)
- SciBERT 
- Specter

Results are logged to the console including:
- Per-class metrics (precision, recall, F1)
- Macro/Micro averaged metrics
- Comparison tables

Usage:
    python scripts/evaluate.py --models baseline scibert specter
    python scripts/evaluate.py --model_paths artifacts/test_baseline.keras artifacts/test_scibert.keras
    python scripts/evaluate.py --limit_samples 1000
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

# Ensure src is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.arxiv_classifier.config import settings
from src.arxiv_classifier.data.loader import load_hf_dataset
from src.arxiv_classifier.models import BaselineClassifier, SciBertClassifier, SpecterClassifier
from src.arxiv_classifier.utils.losses import MultiLabelFocalLoss  # Required for loading models

try:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        classification_report,
        hamming_loss,
        accuracy_score,
        roc_auc_score,
        average_precision_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not found. Install with: pip install scikit-learn")



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive evaluator for multi-label classification models."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.results = {}
        
    def compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray,
        label_names: Optional[list] = None
    ) -> dict:
        """
        Compute comprehensive multi-label classification metrics.
        
        Args:
            y_true: Ground truth binary labels (n_samples, n_classes)
            y_pred: Predicted binary labels (n_samples, n_classes)
            y_pred_proba: Prediction probabilities (n_samples, n_classes)
            label_names: Optional list of label names for per-class metrics
            
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Micro-averaged metrics (treating all samples equally)
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        metrics['micro_precision'] = float(micro_p)
        metrics['micro_recall'] = float(micro_r)
        metrics['micro_f1'] = float(micro_f1)
        
        # Macro-averaged metrics (equal weight per class)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['macro_precision'] = float(macro_p)
        metrics['macro_recall'] = float(macro_r)
        metrics['macro_f1'] = float(macro_f1)
        
        # Weighted metrics (weighted by support)
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['weighted_precision'] = float(weighted_p)
        metrics['weighted_recall'] = float(weighted_r)
        metrics['weighted_f1'] = float(weighted_f1)
        
        # Sample-averaged metrics (for multi-label)
        samples_p, samples_r, samples_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='samples', zero_division=0
        )
        metrics['samples_precision'] = float(samples_p)
        metrics['samples_recall'] = float(samples_r)
        metrics['samples_f1'] = float(samples_f1)
        
        # Hamming loss (fraction of wrong labels)
        metrics['hamming_loss'] = float(hamming_loss(y_true, y_pred))
        
        # Subset accuracy (exact match ratio)
        metrics['subset_accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # ROC-AUC (if we have probability predictions)
        try:
            # Micro-averaged ROC-AUC
            metrics['roc_auc_micro'] = float(roc_auc_score(
                y_true, y_pred_proba, average='micro'
            ))
            # Macro-averaged ROC-AUC
            metrics['roc_auc_macro'] = float(roc_auc_score(
                y_true, y_pred_proba, average='macro'
            ))
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics['roc_auc_micro'] = None
            metrics['roc_auc_macro'] = None
            
        # Average precision (mAP)
        try:
            metrics['average_precision_micro'] = float(average_precision_score(
                y_true, y_pred_proba, average='micro'
            ))
            metrics['average_precision_macro'] = float(average_precision_score(
                y_true, y_pred_proba, average='macro'
            ))
        except ValueError as e:
            logger.warning(f"Could not compute average precision: {e}")
            metrics['average_precision_micro'] = None
            metrics['average_precision_macro'] = None
        
        # Per-class metrics
        per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        per_class_metrics = {}
        for i, (p, r, f1, sup) in enumerate(zip(per_class_p, per_class_r, per_class_f1, support)):
            label = label_names[i] if label_names else str(i)
            per_class_metrics[label] = {
                'precision': float(p),
                'recall': float(r),
                'f1': float(f1),
                'support': int(sup)
            }
        metrics['per_class'] = per_class_metrics
        
        return metrics
    
    def evaluate_model(
        self,
        model,
        model_name: str,
        test_data,
        label_names: Optional[list] = None,
        vectorizer=None  # For baseline model
    ) -> dict:
        """
        Evaluate a single model on the test dataset.
        
        Args:
            model: Keras model to evaluate
            model_name: Name for this model in results
            test_data: Tuple of (inputs, labels) or tf.data.Dataset
            label_names: Optional category names
            vectorizer: Optional vectorizer for baseline model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        if isinstance(test_data, tuple):
            x_test, y_true = test_data
        else:
            # Collect from tf.data.Dataset
            x_test_list = []
            y_true_list = []
            for batch in test_data:
                x_batch, y_batch = batch
                x_test_list.append(x_batch)
                y_true_ list.append(y_batch)
            x_test = {k: np.concatenate([x[k] for x in x_test_list]) for k in x_test_list[0].keys()} \
                if isinstance(x_test_list[0], dict) else np.concatenate(x_test_list)
            y_true = np.concatenate(y_true_list)
        
        # Get predictions
        logger.info("Running inference...")
        start_time = datetime.now()
        y_pred_proba = model.predict(x_test, verbose=1)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Binarize predictions
        y_pred = (y_pred_proba > self.threshold).astype(np.float32)
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, y_pred_proba, label_names)
        metrics['inference_time_seconds'] = inference_time
        metrics['samples_evaluated'] = len(y_true)
        metrics['model_name'] = model_name
        
        self.results[model_name] = metrics
        
        logger.info(f"  Micro F1: {metrics['micro_f1']:.4f}")
        logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
        
        return metrics


def load_model_and_vectorizer(model_path: Path, model_type: str):
    """Load a model and its vectorizer (if baseline)."""
    logger.info(f"Loading model from {model_path}")
    
    # Load the model (compile=False since we only need inference, avoids custom loss issues)
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load vectorizer for baseline
    vectorizer = None
    if model_type == "baseline":
        vec_path = model_path.parent / f"{model_path.stem}_vectorizer_tf"
        if vec_path.exists():
            logger.info(f"Loading vectorizer from {vec_path}")
            # Use TFSMLayer to make it callable (like inference engine does)
            vectorizer = tf.keras.layers.TFSMLayer(str(vec_path), call_endpoint="serving_default")
        else:
            logger.warning(f"Vectorizer not found at {vec_path}")
    
    return model, vectorizer


def prepare_evaluation_data(dataset, model_type: str, num_classes: int, vectorizer=None, tokenizer=None):
    """Prepare data for evaluation based on model type."""
    
    if model_type == "baseline":
        from src.arxiv_classifier.models.baseline import preprocess_texts_batch
        
        def preprocess(examples):
            texts = [f"{t} [SEP] {a}" for t, a in zip(examples['title'], examples['abstract'])]
            # Apply NLP preprocessing to match training
            texts = preprocess_texts_batch(texts, use_lemmatization=True, use_stemming=False, use_stopwords=True)
            
            with tf.device("/CPU:0"):
                # TFSMLayer expects tensor input
                text_tensor = tf.constant(texts)
                result = vectorizer(text_tensor)
                # TFSMLayer returns a dict, extract the tensor
                if isinstance(result, dict):
                    vectors = list(result.values())[0]
                else:
                    vectors = result
            
            labels = []
            for cats in examples['categories']:
                multi_hot = [0.0] * num_classes
                for cat in cats:
                    if cat in settings.label2id:
                        multi_hot[settings.label2id[cat]] = 1.0
                labels.append(multi_hot)
            
            return {"text": vectors.numpy(), "labels": labels}
        
        tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
        x_test = np.array(tokenized["text"])
        y_test = np.array(tokenized["labels"])
        return x_test, y_test
        
    else:
        # Transformer preprocessing
        def preprocess(examples):
            texts = [f"{t} [SEP] {a}" for t, a in zip(examples['title'], examples['abstract'])]
            result = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=settings.model.max_length,
                return_tensors=None
            )
            
            labels = []
            for cats in examples['categories']:
                multi_hot = [0.0] * num_classes
                for cat in cats:
                    if cat in settings.label2id:
                        multi_hot[settings.label2id[cat]] = 1.0
                labels.append(multi_hot)
            
            result["labels"] = labels
            return result
        
        tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
        
        x_test = {
            'input_ids': np.array(tokenized['input_ids']),
            'attention_mask': np.array(tokenized['attention_mask'])
        }
        y_test = np.array(tokenized['labels'])
        return x_test, y_test


def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create a comparison DataFrame from results."""
    rows = []
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Micro F1': metrics['micro_f1'],
            'Macro F1': metrics['macro_f1'],
            'Weighted F1': metrics['weighted_f1'],
            'Micro Precision': metrics['micro_precision'],
            'Micro Recall': metrics['micro_recall'],
            'Hamming Loss': metrics['hamming_loss'],
            'Subset Accuracy': metrics['subset_accuracy'],
            'ROC-AUC (Micro)': metrics.get('roc_auc_micro'),
            'ROC-AUC (Macro)': metrics.get('roc_auc_macro'),
            'Avg Precision (Micro)': metrics.get('average_precision_micro'),
            'Inference Time (s)': metrics['inference_time_seconds'],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.set_index('Model')





def main():
    parser = argparse.ArgumentParser(description="Evaluate ArXiv Classifier Models")
    parser.add_argument(
        "--models", 
        nargs='+', 
        choices=['baseline', 'scibert', 'specter'],
        default=['baseline', 'scibert', 'specter'],
        help="Model types to evaluate"
    )
    parser.add_argument(
        "--model_paths",
        nargs='+',
        type=str,
        help="Custom paths to model files (overrides --models)"
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="test_",
        help="Prefix for model file names in artifacts directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for predictions"
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=None,
        help="Limit test samples for faster evaluation"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to use for evaluation"
    )
    args = parser.parse_args()
    

    
    # Load category config
    settings.load_category_config()
    num_classes = len(settings.id2label)
    label_names = [settings.id2label[i] for i in range(num_classes)]
    
    logger.info(f"Loaded {num_classes} categories")
    
    # Load test dataset
    logger.info(f"Loading {args.split} dataset...")
    test_raw = load_hf_dataset(args.split)
    
    if args.limit_samples:
        test_limit = min(args.limit_samples, len(test_raw))
        test_raw = test_raw.select(range(test_limit))
        logger.info(f"Limited to {test_limit} samples")
    
    logger.info(f"Test dataset size: {len(test_raw)}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(threshold=args.threshold)
    
    # Determine which models to evaluate
    if args.model_paths:
        # Custom paths provided
        model_configs = []
        for path in args.model_paths:
            path = Path(path)
            # Infer model type from filename
            if 'baseline' in path.stem.lower():
                model_type = 'baseline'
            elif 'scibert' in path.stem.lower():
                model_type = 'scibert'
            elif 'specter' in path.stem.lower():
                model_type = 'specter'
            else:
                logger.warning(f"Could not infer model type for {path}, assuming transformer")
                model_type = 'scibert'
            model_configs.append((path, model_type, path.stem))
    else:
        # Use default naming convention
        model_configs = []
        for model_type in args.models:
            model_path = settings.paths.ARTIFACTS_DIR / f"{args.model_prefix}{model_type}.keras"
            if model_path.exists():
                model_configs.append((model_path, model_type, f"{args.model_prefix}{model_type}"))
            else:
                logger.warning(f"Model not found: {model_path}")
    
    if not model_configs:
        logger.error("No models found to evaluate!")
        sys.exit(1)
    
    logger.info(f"Will evaluate {len(model_configs)} models: {[c[2] for c in model_configs]}")
    
    # Evaluate each model
    all_results = {}
    
    for model_path, model_type, model_name in model_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name} ({model_type})")
        logger.info(f"{'='*60}")
        
        try:
            # Load model and vectorizer
            model, vectorizer = load_model_and_vectorizer(model_path, model_type)
            
            # Prepare data
            if model_type == "baseline":
                if vectorizer is None:
                    logger.error(f"Vectorizer required for baseline model but not found")
                    continue
                x_test, y_test = prepare_evaluation_data(
                    test_raw, model_type, num_classes, vectorizer=vectorizer
                )
            else:
                from transformers import AutoTokenizer
                if model_type == "specter":
                    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
                else:
                    # SciBERT - use explicit model name for consistency
                    tokenizer = AutoTokenizer.from_pretrained("giacomomiolo/scibert_reupload")
                
                x_test, y_test = prepare_evaluation_data(
                    test_raw, model_type, num_classes, tokenizer=tokenizer
                )
            
            # Evaluate
            metrics = evaluator.evaluate_model(
                model, model_name, (x_test, y_test), label_names
            )
            all_results[model_name] = metrics
            
            # Clear memory
            del model
            tf.keras.backend.clear_session()
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        logger.error("No models were successfully evaluated!")
        sys.exit(1)
    
    # Generate outputs
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*60}")
    
    # 1. Comparison table
    comparison_df = create_comparison_table(all_results)
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(comparison_df.to_string())
    print("="*80 + "\n")
    

    
    # 4. Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    best_micro_f1 = max(all_results.items(), key=lambda x: x[1]['micro_f1'])
    best_macro_f1 = max(all_results.items(), key=lambda x: x[1]['macro_f1'])
    fastest = min(all_results.items(), key=lambda x: x[1]['inference_time_seconds'])
    
    print(f"Best Micro F1:  {best_micro_f1[0]} ({best_micro_f1[1]['micro_f1']:.4f})")
    print(f"Best Macro F1:  {best_macro_f1[0]} ({best_macro_f1[1]['macro_f1']:.4f})")
    print(f"Fastest Model:  {fastest[0]} ({fastest[1]['inference_time_seconds']:.2f}s)")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    main()
