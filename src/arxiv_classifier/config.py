import json
from pathlib import Path
from typing import List, Dict, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class ProjectPaths(BaseSettings):
    """Dynamic path resolution based on where the code is running."""
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = ROOT_DIR / "data"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "hf_dataset"
    CONFIG_DIR: Path = ROOT_DIR / "configs"
    ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"

class ModelConfig(BaseSettings):
    """Model hyperparameters."""
    model_name: str = "giacomomiolo/scibert_reupload"
    max_length: int = 256
    dropout_rate: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 5
    threshold: float = 0.5  # Decision threshold for multi-label

class AppConfig(BaseSettings):
    """Global application registry."""
    paths: ProjectPaths = Field(default_factory=ProjectPaths)
    model: ModelConfig = Field(default_factory=ModelConfig)
    
    # Placeholder for category mapping (loaded at runtime)
    id2label: Dict[int, str] = {}
    label2id: Dict[str, int] = {}

    def load_category_config(self):
        """Loads the category mapping generated during EDA."""
        config_path = self.paths.CONFIG_DIR / "category_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Category config not found at {config_path}")
            
        with open(config_path, "r") as f:
            data = json.load(f)
            self.id2label = {int(k): v for k, v in data['idx_to_category'].items()}
            self.label2id = data['category_to_idx']

settings = AppConfig()