"""
Utility for downloading SPLADE models from Hugging Face
"""

import logging
import os

from transformers import AutoTokenizer, AutoModelForMaskedLM

logger = logging.getLogger("model_downloader")


def download_model_if_needed(model_dir: str, model_id: str) -> bool:
    """
    Check if model exists in the specified directory, download from Hugging Face if not
    
    Args:
        model_dir: Directory where model should be stored
        model_id: Hugging Face model ID to download
        
    Returns:
        True if model was downloaded or already exists, False on failure
    """
    # Check if directory exists
    if not os.path.exists(model_dir):
        try:
            logger.info(f"Creating model directory: {model_dir}")
            os.makedirs(model_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create model directory {model_dir}: {e}")
            return False

    # Check if model files exist
    has_config = os.path.exists(os.path.join(model_dir, "config.json"))
    has_tokenizer = os.path.exists(os.path.join(model_dir, "tokenizer_config.json"))
    has_model = os.path.exists(os.path.join(model_dir, "pytorch_model.bin")) or \
                os.path.exists(os.path.join(model_dir, "model.safetensors"))

    # If all model files exist, we're done
    if has_config and has_tokenizer and has_model:
        logger.info(f"Model already exists at {model_dir}")
        return True

    # Download the model if not found
    try:
        logger.info(f"Downloading model {model_id} to {model_dir}...")

        # Download tokenizer
        if not has_tokenizer:
            logger.info("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(model_dir)
            logger.info(f"Tokenizer saved to {model_dir}")

        # Download model
        if not has_model or not has_config:
            logger.info("Downloading model weights...")
            model = AutoModelForMaskedLM.from_pretrained(model_id, use_safetensors=True)
            model.save_pretrained(model_dir)
            logger.info(f"Model saved to {model_dir}")

        logger.info("Model download complete")
        return True

    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        return False
