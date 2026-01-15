#!/usr/bin/env python
"""
W&B Sweep Training Script for ArXiv Classifier

This script is designed to be called by wandb.agent() for hyperparameter sweeps.
It can also be run standalone to create and launch a sweep.

Usage:
    # Create a new sweep and run it:
    python scripts/train_sweep.py --create_sweep --count 10

    # Run as an agent for an existing sweep:
    python scripts/train_sweep.py --sweep_id <sweep_id> --count 5

    # Just run a single training with current wandb.config (called by agent):
    python scripts/train_sweep.py
"""
import os
import argparse
import logging
import sys
from pathlib import Path

# Ensure src is in the path if not installed as a package
sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml
import tensorflow as tf
import wandb

try:
    from wandb.integration.keras import WandbMetricsLogger
except ImportError:
    try:
        from wandb.keras import WandbMetricsLogger
    except ImportError:
        WandbMetricsLogger = None

from src.arxiv_classifier.config import settings
from src.arxiv_classifier.training import ArxivTrainer
from src.arxiv_classifier.models import SciBertClassifier, BaselineClassifier, SpecterClassifier
from src.arxiv_classifier.data.loader import load_hf_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default sweep project
WANDB_PROJECT = "arxiv-classifier-sweeps"


def get_sweep_config() -> dict:
    """Load sweep configuration from YAML file."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "sweep_config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    # Fallback default config
    return {
        "program": "scripts/train_sweep.py",
        "method": "bayes",
        "metric": {"name": "best_val_f1_score", "goal": "maximize"},
        "parameters": {
            "model_type": {"values": ["scibert", "baseline"]},
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-4},
            "batch_size": {"values": [16, 32]},
            "epochs": {"value": 5},
            "dropout_rate": {"distribution": "uniform", "min": 0.1, "max": 0.4},
            "focal_gamma": {"value": 2.0},
            "focal_alpha": {"value": 0.25},
        },
        "early_terminate": {"type": "hyperband", "min_iter": 2, "eta": 2},
    }


def sweep_train():
    """
    Main training function called by wandb.agent().
    Uses wandb.config for hyperparameters.
    """
    # Initialize run (will use sweep config)
    run = wandb.init()
    config = wandb.config
    
    logger.info(f"Starting sweep run: {run.name}")
    logger.info(f"Config: {dict(config)}")
    
    # Load category mappings
    settings.load_category_config()
    num_classes = len(settings.id2label)
    
    # Get hyperparameters from sweep config with defaults
    model_type = config.get("model_type", "scibert")
    learning_rate = config.get("learning_rate", 5e-5)  # Increased from 2e-5
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 5)
    dropout_rate = config.get("dropout_rate", 0.1)
    max_length = config.get("max_length", 256)
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.7)  # Higher alpha for sparse multi-label
    limit_samples = config.get("limit_samples", 10000)
    
    # Instantiate model based on sweep config
    if model_type == "scibert":
        model = SciBertClassifier(
            num_classes=num_classes,
            model_name=settings.model.model_name,
            dropout_rate=dropout_rate
        )
        logger.info(f"Initialized SciBERT model for {num_classes} classes.")
    elif model_type == "specter":
        model = SpecterClassifier(
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        logger.info(f"Initialized Specter model for {num_classes} classes.")
    else:
        model = BaselineClassifier(num_classes=num_classes)
        logger.info("Initialized TF-IDF Baseline model.")
    
    # Load data
    train_raw = load_hf_dataset("train")
    val_raw = load_hf_dataset("val")
    
    # Limit samples for sweep efficiency
    if limit_samples:
        logger.info(f"Limiting dataset to {limit_samples} samples.")
        train_limit = min(limit_samples, len(train_raw))
        val_limit = min(limit_samples, len(val_raw))
        train_raw = train_raw.select(range(train_limit))
        val_raw = val_raw.select(range(val_limit))
    
    # Check for GPU/MPS
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
    else:
        logger.warning("No GPU found. Training will be slow on CPU.")
    
    # Preprocess data
    vocab_size = 20000
    if model_type == "baseline":
        # Import text preprocessing from baseline module
        from src.arxiv_classifier.models.baseline import preprocess_texts_batch
        
        # TextVectorization for baseline
        logger.info("Preparing TextVectorization for baseline...")
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode='tf_idf'
        )
        
        def preprocess_for_adaptation(examples):
            # Combine title and abstract
            texts = [f"{t} [SEP] {a}" for t, a in zip(examples['title'], examples['abstract'])]
            # Apply NLP preprocessing (lemmatization, stop-word removal)
            return preprocess_texts_batch(texts, use_lemmatization=True, use_stemming=False, use_stopwords=True)
        
        logger.info("Applying NLP preprocessing (lemmatization, stop-word removal)...")
        to_adapt = train_raw.map(
            lambda x: {"text": preprocess_for_adaptation(x)},
            batched=True,
            remove_columns=train_raw.column_names
        )
        adaptation_texts = to_adapt["text"]
        
        logger.info("Adapting vectorizer...")
        with tf.device("/CPU:0"):
            vectorizer.adapt(adaptation_texts)
        
        def preprocess_baseline_vectorized(examples):
            texts = preprocess_for_adaptation(examples)
            with tf.device("/CPU:0"):
                vectors = vectorizer(texts)
            
            labels = []
            for cats in examples['categories']:
                multi_hot = [0.0] * num_classes
                for cat in cats:
                    if cat in settings.label2id:
                        multi_hot[settings.label2id[cat]] = 1.0
                labels.append(multi_hot)
            return {"text": vectors.numpy(), "labels": labels}
        
        logger.info("Vectorizing and preprocessing datasets...")
        train_tokenized = train_raw.map(preprocess_baseline_vectorized, batched=True, remove_columns=train_raw.column_names)
        val_tokenized = val_raw.map(preprocess_baseline_vectorized, batched=True, remove_columns=val_raw.column_names)
        cols = ['text']
        
    else:
        # Transformer preprocessing
        from transformers import AutoTokenizer
        
        tokenizer_name = "allenai/specter" if model_type == "specter" else settings.model.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        def preprocess_transformer(examples):
            texts = [f"{t} [SEP] {a}" for t, a in zip(examples['title'], examples['abstract'])]
            result = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length
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
        
        logger.info("Preprocessing datasets for transformer...")
        train_tokenized = train_raw.map(preprocess_transformer, batched=True, remove_columns=train_raw.column_names)
        val_tokenized = val_raw.map(preprocess_transformer, batched=True, remove_columns=val_raw.column_names)
        cols = ['input_ids', 'attention_mask']
    
    # Convert to TF datasets
    train_ds = train_tokenized.to_tf_dataset(
        columns=cols,
        label_cols=['labels'],
        shuffle=True,
        batch_size=batch_size
    )
    val_ds = val_tokenized.to_tf_dataset(
        columns=cols,
        label_cols=['labels'],
        shuffle=False,
        batch_size=batch_size
    )
    
    # Optimize datasets
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Setup callbacks
    callbacks = []
    if WandbMetricsLogger:
        callbacks.append(WandbMetricsLogger())
    
    # Add best model checkpoint callback that logs to WandB
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            patience=3,
            mode='max',
            restore_best_weights=True
        )
    )
    
    # Build trainer config
    trainer_config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
    }
    
    trainer = ArxivTrainer(model=model, config=trainer_config)
    
    # Train
    logger.info(f"Starting training for {epochs} epochs...")
    history = trainer.compile_and_fit(
        train_ds=train_ds,
        val_ds=val_ds,
        callbacks=callbacks
    )
    
    # Log best metrics to WandB summary for sweep optimization
    best_val_f1 = max(history.history.get('val_f1_score', [0]))
    best_val_auc = max(history.history.get('val_auc', [0]))
    best_val_accuracy = max(history.history.get('val_accuracy', [0]))
    
    wandb.log({
        "best_val_f1_score": best_val_f1,
        "best_val_auc": best_val_auc,
        "best_val_accuracy": best_val_accuracy,
    })
    
    wandb.summary["best_val_f1_score"] = best_val_f1
    wandb.summary["best_val_auc"] = best_val_auc
    wandb.summary["best_val_accuracy"] = best_val_accuracy
    
    logger.info(f"Sweep run complete. Best val F1: {best_val_f1:.4f}")
    
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="W&B Sweep Training for ArXiv Classifier")
    parser.add_argument("--create_sweep", action="store_true", help="Create a new sweep and start running it")
    parser.add_argument("--sweep_id", type=str, help="Existing sweep ID to join as an agent")
    parser.add_argument("--count", type=int, default=10, help="Number of runs for the agent")
    parser.add_argument("--project", type=str, default=WANDB_PROJECT, help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (team/username)")
    args = parser.parse_args()
    
    if args.create_sweep:
        # Create a new sweep
        sweep_config = get_sweep_config()
        logger.info(f"Creating sweep with config: {sweep_config}")
        
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=args.project,
            entity=args.entity
        )
        logger.info(f"Created sweep with ID: {sweep_id}")
        
        # Run the agent
        logger.info(f"Starting sweep agent for {args.count} runs...")
        wandb.agent(
            sweep_id,
            function=sweep_train,
            count=args.count,
            project=args.project,
            entity=args.entity
        )
        
    elif args.sweep_id:
        # Join existing sweep as agent
        logger.info(f"Joining sweep {args.sweep_id} as agent...")
        wandb.agent(
            args.sweep_id,
            function=sweep_train,
            count=args.count,
            project=args.project,
            entity=args.entity
        )
        
    else:
        # Called directly by wandb agent (no args means sweep is calling us)
        sweep_train()


if __name__ == "__main__":
    main()
