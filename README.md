# ArXiv Paper Classifier

A production-ready multi-label text classification system for categorizing research paper abstracts into ArXiv subject categories. Built with TensorFlow/Keras, featuring multiple model architectures, a Django REST API, and an interactive Gradio demo.

## Project Overview

This system automatically classifies research paper abstracts into one or more of 50 ArXiv categories spanning Computer Science, Physics, Mathematics, and Statistics. It supports three different model architectures with varying complexity/accuracy trade-offs, and provides both a REST API and a web interface for inference.

### Key Features

- Multi-label classification: Papers can belong to multiple categories simultaneously
- Three model architectures: Baseline (TF-IDF + MLP), SciBERT, and SPECTER transformers
- Production REST API: Django REST Framework with health checks and batch inference
- Interactive demo: Gradio-based web interface for testing
- Experiment tracking: Integrated Weights & Biases (WandB) logging
- Hyperparameter optimization: W&B Sweeps for automated tuning
- Docker support: Multi-stage builds with dev/prod configurations
- GPU acceleration: Apple Silicon MPS and CUDA support
- Comprehensive evaluation: 10+ metrics including F1, ROC-AUC, mAP, and per-class analysis
- Focal loss: Custom loss function for handling class imbalance
- Batch inference: API supports batch predictions for efficiency
- Test suite: Pytest-based testing for API and model validation
- HuggingFace integration: Direct dataset loading from HuggingFace Hub

## Project Structure

```
arxiv-classifier-system/
├── src/arxiv_classifier/          # Core ML library
│   ├── models/                    # Model architectures
│   │   ├── base.py                # Abstract base class
│   │   ├── baseline.py            # TF-IDF + MLP classifier
│   │   ├── sci_bert_classifier.py # SciBERT transformer
│   │   └── specter.py             # SPECTER transformer
│   ├── training/                  # Training utilities
│   │   ├── trainer.py             # ArxivTrainer class
│   │   ├── metrics.py             # Custom metrics
│   │   └── sweep_trainer.py       # Hyperparameter sweeps
│   ├── inference/                 # Inference engine
│   │   ├── engine.py              # ArxivInferenceEngine
│   │   └── schemas.py             # Data schemas
│   ├── data/                      # Data loading
│   │   ├── loader.py              # HuggingFace dataset loader
│   │   └── processor.py           # Preprocessing utilities
│   ├── utils/                     # Utilities
│   │   ├── losses.py              # Focal loss implementation
│   │   └── metrics.py             # F1 score metric
│   └── config.py                  # Centralized configuration
├── backend/                       # Django REST API
│   ├── api/                       # API application
│   │   ├── views.py               # Predict and Health endpoints
│   │   ├── serializers.py         # Request/response validation
│   │   ├── apps.py                # App configuration and model loading
│   │   └── urls.py                # URL routing
│   ├── core/                      # Django project settings
│   └── manage.py
├── scripts/
│   ├── train.py                   # Training entry point
│   ├── train_sweep.py             # W&B hyperparameter sweeps
│   └── evaluate.py                # Model evaluation suite
├── tests/                         # Pytest test suite
│   ├── test_api.py                # API integration tests
│   ├── test_all_models.py         # Model unit tests
│   └── conftest.py                # Pytest fixtures
├── configs/
│   ├── category_config.json       # Category mappings (50 classes)
│   ├── training_config.json       # Training hyperparameters
│   └── sweep_config.yaml          # W&B sweep configuration
├── docker/                        # Docker configuration
├── artifacts/                     # Trained model artifacts
├── data/                          # Dataset storage
├── notebooks/                     # Exploratory notebooks
├── gradio_demo.py                 # Interactive web demo
└── pyproject.toml                 # Project dependencies
```

## Architecture

### Model Architectures

| Model    | Description                                                          | Parameters | Use Case                         |
| -------- | -------------------------------------------------------------------- | ---------- | -------------------------------- |
| Baseline | TF-IDF + Dense MLP (with stemming, lemmatization, stop-word removal) | ~10M       | Fast iteration, debugging        |
| SciBERT  | Scientific BERT transformer                                          | ~110M      | High accuracy on scientific text |
| SPECTER  | Citation-aware embeddings                                            | ~110M      | Document similarity tasks        |

All models inherit from `ArxivClassifierBase` and output sigmoid-activated probabilities for 50 categories.

### Key Design Decisions

1. **Abstract base class pattern**: All models implement a common interface (`ArxivClassifierBase`) enabling swappable model backends without changing inference code.

2. **Multi-label classification**: EDA revealed that 47.7% of papers have 2+ categories. This informed the choice of multi-label sigmoid outputs over multi-class softmax.

3. **Minimal preprocessing for Transformers**: While the baseline model employs traditional NLTK-based preprocessing (stemming, stop-word removal), transformer models utilize raw text. This preserves the grammatical context and linguistic structure essential for BERT-based architectures to learn high-quality contextual representations.

4. **Focal loss for class imbalance**: ArXiv categories have highly imbalanced distributions. Focal loss down-weights easy examples, focusing training on hard cases.

5. **Inference engine abstraction**: `ArxivInferenceEngine` handles model loading, preprocessing, and prediction, decoupling the API from model internals.

6. **Environment-based model selection**: The API loads models based on `MODEL_PATH` and `MODEL_TYPE` environment variables, supporting deployment flexibility.

### System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Gradio Demo   │────>│  Django REST    │────>│   Inference     │
│   (Port 7860)   │     │  API (Port 8000)│     │   Engine        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         v
                                               ┌─────────────────┐
                                               │  Keras Model    │
                                               │  (.keras file)  │
                                               └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
git clone https://github.com/your-username/arxiv-classifier-system.git
cd arxiv-classifier-system

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Running the API

```bash
export MODEL_PATH="artifacts/baseline.keras"
export MODEL_TYPE="baseline"

cd backend
python manage.py runserver
```

The API will be available at `http://localhost:8000/api/v1/`

### Running the Gradio Demo

```bash
# Make sure Django server is running first
python gradio_demo.py
```

Open `http://localhost:7860` in your browser.

## Docker Deployment

```bash
# Development mode (with hot reload)
cd docker
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Production mode
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

See [docker/README.md](docker/README.md) for detailed Docker configuration options.

## Training

### Basic Training

```bash
# Train baseline model (faster)
python scripts/train.py \
    --model_type baseline \
    --run_name baseline_v1 \
    --epochs 10

# Train SciBERT model
python scripts/train.py \
    --model_type scibert \
    --run_name scibert_v1 \
    --epochs 5 \
    --batch_size 32 \
    --limit_samples 10000

# Train SPECTER model
python scripts/train.py \
    --model_type specter \
    --run_name specter_v1 \
    --epochs 5
```

### Hyperparameter Sweeps

```bash
# Run W&B hyperparameter sweep
python scripts/train_sweep.py --sweep_count 20
```

### Training Arguments

| Argument          | Description                                          | Default      |
| ----------------- | ---------------------------------------------------- | ------------ |
| `--model_type`    | Model architecture: `scibert`, `baseline`, `specter` | `scibert`    |
| `--run_name`      | Experiment name for W&B and saved artifacts          | `scibert_v1` |
| `--epochs`        | Number of training epochs                            | 5            |
| `--batch_size`    | Training batch size                                  | 32           |
| `--limit_samples` | Limit dataset size (for debugging)                   | 10000        |

## Evaluation

The evaluation script (`scripts/evaluate.py`) provides comprehensive multi-label classification metrics:

```bash
python scripts/evaluate.py --models baseline scibert specter
```

### Metrics Computed

| Metric                           | Description                                                        |
| -------------------------------- | ------------------------------------------------------------------ |
| **Micro F1/Precision/Recall**    | Aggregates TP/FP/FN across all labels, then computes metrics       |
| **Macro F1/Precision/Recall**    | Computes per-class metrics, then averages (equal weight per class) |
| **Weighted F1/Precision/Recall** | Per-class metrics weighted by class support                        |
| **Samples F1/Precision/Recall**  | Computes per-sample metrics, then averages (for multi-label)       |
| **Hamming Loss**                 | Fraction of incorrectly predicted labels                           |
| **Subset Accuracy**              | Exact match ratio (all labels must match)                          |
| **ROC-AUC (Micro/Macro)**        | Area under ROC curve for probability rankings                      |
| **Average Precision (mAP)**      | Mean average precision across classes                              |
| **Per-class metrics**            | Individual precision, recall, F1, and support for each category    |



## API Reference

### Health Check

```http
GET /api/v1/health/
```

Response:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_type": "scibert",
    "version": "1.0.0"
}
```

### Predict

```http
POST /api/v1/predict/
Content-Type: application/json
```

Request:
```json
{
    "abstracts": [
        "We propose a novel deep learning architecture for image classification..."
    ]
}
```

Response:
```json
{
    "results": [
        {
            "abstract_preview": "We propose a novel deep learning...",
            "predictions": [
                {"label": "cs.CV", "probability": 0.9234},
                {"label": "cs.LG", "probability": 0.8821}
            ]
        }
    ],
    "model_version": "scibert-v1"
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Test specific model type
pytest tests/test_api.py -v --model-type scibert

# Test direct model loading only (no Django)
pytest tests/test_api.py -v --direct-only

# Test API endpoints only
pytest tests/test_api.py -v --api-only
```

## Configuration

### Environment Variables

| Variable     | Description                                          | Default  |
| ------------ | ---------------------------------------------------- | -------- |
| `MODEL_PATH` | Path to trained model file                           | Required |
| `MODEL_TYPE` | Model type: `baseline`, `scibert`, `specter`, `auto` | `auto`   |
| `DEBUG`      | Django debug mode                                    | `False`  |

### Training Configuration

See `configs/training_config.json` for model and training hyperparameters.

## Technology Stack

| Component           | Technology                         |
| ------------------- | ---------------------------------- |
| ML Framework        | TensorFlow 2.16 / Keras 3          |
| Transformers        | HuggingFace Transformers           |
| API Framework       | Django 5.0 + Django REST Framework |
| Demo Interface      | Gradio 4.x                         |
| Experiment Tracking | Weights & Biases                   |
| Testing             | Pytest                             |
| Package Manager     | uv                                 |
| GPU Support         | Apple MPS / CUDA                   |

## Future Improvements

Given additional time, the following enhancements would improve the system:


1. **Extended model training**: Current models were trained on limited data and epochs due to time constraints. Full training on the complete dataset with proper hyperparameter tuning would significantly improve classification accuracy.

2. **Model performance benchmarking**: Implement systematic evaluation comparing all three architectures across precision, recall, F1, and inference latency.

3. **Expanded test coverage**: Add unit tests for individual model components, integration tests for the training pipeline, and end-to-end tests for the complete workflow.

4. **Migrate to PyTorch backend**: The HuggingFace Transformers library is deprecating TensorFlow support. PyTorch has become the dominant framework in ML research, offering better pretrained model availability and community support for transformer architectures.

5. **Add more consistent logging**: Add more consistent logging to the system to make it easier to debug and monitor.

6. **Kubernetes deployment**: Add Helm charts and Kubernetes manifests for production-grade orchestration with horizontal scaling.

7. **Model versioning and A/B testing**: Integrate MLflow or W&B for model registry and enable A/B testing between model versions in production.

## License

This project is licensed under the MIT License.

## References

- [SciBERT: A Pretrained Language Model for Scientific Text](https://arxiv.org/abs/1903.10676)
- [SPECTER: Document-level Representation Learning](https://arxiv.org/abs/2004.07180)
- [ArXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
