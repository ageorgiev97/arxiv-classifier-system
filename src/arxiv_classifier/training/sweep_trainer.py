import wandb
from .trainer import ArxivTrainer
try:
    from wandb.integration.keras import WandbMetricsLogger
except ImportError:
    try:
        from wandb.keras import WandbMetricsLogger
    except ImportError:
        WandbMetricsLogger = None

def run_sweep_agent():
    """Entry point for wandb agent."""
    with wandb.init() as run:
        config = wandb.config
        
        # Logic to instantiate the specific model based on sweep config
        from ..models import SciBertClassifier, BaselineClassifier
        if config.model_type == "transformer":
            model = SciBertClassifier(num_classes=config.num_classes, model_name=config.model_name)
        else:
            model = BaselineClassifier(num_classes=config.num_classes)

        # Standard Training Logic
        trainer = ArxivTrainer(model, config)
        
        # Load data (Helper needed here to fetch datasets)
        # train_ds, val_ds = load_data_for_model(config.model_type)

        trainer.compile_and_fit(
            train_ds, 
            val_ds, 
            callbacks=[WandbMetricsLogger()]
        )

# Example of how you would trigger this in a script:
# wandb.agent(sweep_id, function=run_sweep_agent)