import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# --- Project Root ---
BASE_DIR = Path(__file__).resolve().parent.parent # Assumes config.py is in a 'config' or similar subfolder

# --- Path Configurations ---
class PathConfig:
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_SAVE_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model Parameters ---
class ModelConfig:
    # Example parameters - adjust based on the actual model architecture
    MODEL_NAME = os.getenv("MODEL_NAME", "t5-small") # Or another base model like BART, etc.
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 128
    NUM_BEAMS = int(os.getenv("NUM_BEAMS", 4))
    LENGTH_PENALTY = float(os.getenv("LENGTH_PENALTY", 2.0))
    EARLY_STOPPING = bool(os.getenv("EARLY_STOPPING", True))
    # Add other model-specific hyperparameters here
    # e.g., EMBEDDING_DIM, HIDDEN_UNITS, NUM_LAYERS etc. if building from scratch

# --- Training Parameters ---
class TrainingConfig:
    # Training specific parameters
    EPOCHS = int(os.getenv("EPOCHS", 3))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 5e-5))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
    OPTIMIZER = os.getenv("OPTIMIZER", "AdamW") # e.g., AdamW, SGD
    SCHEDULER = os.getenv("SCHEDULER", "linear") # e.g., linear, cosine
    WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", 500))
    GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 1))
    MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", 1.0))
    SEED = int(os.getenv("SEED", 42))
    EVALUATION_STRATEGY = os.getenv("EVALUATION_STRATEGY", "epoch") # "steps" or "epoch"
    EVAL_STEPS = int(os.getenv("EVAL_STEPS", 500)) # Relevant if EVALUATION_STRATEGY is "steps"
    SAVE_STRATEGY = os.getenv("SAVE_STRATEGY", "epoch") # "steps" or "epoch"
    SAVE_STEPS = int(os.getenv("SAVE_STEPS", 500)) # Relevant if SAVE_STRATEGY is "steps"
    SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", 2)) # Max checkpoints to keep
    FP16 = bool(os.getenv("FP16", False)) # Use mixed precision training

# --- Environment Configuration ---
class EnvConfig:
    # Environment settings
    DEVICE = os.getenv("DEVICE", "cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ContextualConvoCondenser")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY") # Your wandb entity (username or team)
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    WANDB_DISABLED = os.getenv("WANDB_DISABLED", "false").lower() == "true"
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", min(os.cpu_count(), 8))) # Dataloader workers

# --- Logging Setup ---
def setup_logging(log_level_str: str = EnvConfig.LOG_LEVEL, log_dir: Path = PathConfig.LOGS_DIR):
    log_file = log_dir / "app.log"
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Basic configuration
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # Also log to console
        ]
    )

    # Set higher level for noisy libraries if needed
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# --- Instantiate Configs ---
# Make configs easily accessible
paths = PathConfig()
model_params = ModelConfig()
train_params = TrainingConfig()
env_config = EnvConfig()

# --- Initialize Logging ---
# Call this early in your main script
# setup_logging() # You might want to call this explicitly in your main script entry point

if __name__ == "__main__":
    # Example of how to access configurations
    print(f"--- Path Configurations ---")
    print(f"Data Directory: {paths.DATA_DIR}")
    print(f"Model Save Directory: {paths.MODEL_SAVE_DIR}")
    print(f"Logs Directory: {paths.LOGS_DIR}")

    print(f"\n--- Model Parameters ---")
    print(f"Model Name: {model_params.MODEL_NAME}")
    print(f"Max Input Length: {model_params.MAX_INPUT_LENGTH}")

    print(f"\n--- Training Parameters ---")
    print(f"Epochs: {train_params.EPOCHS}")
    print(f"Batch Size: {train_params.BATCH_SIZE}")
    print(f"Learning Rate: {train_params.LEARNING_RATE}")
    print(f"Seed: {train_params.SEED}")

    print(f"\n--- Environment Configuration ---")
    print(f"Device: {env_config.DEVICE}")
    print(f"Log Level: {env_config.LOG_LEVEL}")
    print(f"WandB Project: {env_config.WANDB_PROJECT}")
    print(f"WandB Disabled: {env_config.WANDB_DISABLED}")

    # Setup logging example
    # setup_logging()
    # logging.info("Configuration loaded successfully.")
    # logging.warning("This is a warning example.")
    # logging.error("This is an error example.")