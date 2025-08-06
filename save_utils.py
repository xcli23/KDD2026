import os
from PIL import Image
import io

def save_result(image, prompt, save_dir, epoch, method, batch_idx, sample_idx):
    """Save images and prompts to the specified directory
    
    Args:
        image: PIL Image object
        prompt: Prompt string
        save_dir: Directory path for saving
        epoch: Current epoch
        method: Method name ('ori' or 'ours')
        batch_idx: Batch index
        sample_idx: Sample index in the batch
    """
    # Create method directory
    method_dir = os.path.join(save_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    # Create epoch directory
    epoch_dir = os.path.join(method_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Generate unique filename (based on batch and sample index)
    file_prefix = f"sample_{batch_idx}_{sample_idx}"
    
    # Save image
    image_path = os.path.join(epoch_dir, f"{file_prefix}.png")
    if image is None:
        print(f"Warning: Image for {file_prefix} is None, skipping save.")
    else:
        image.save(image_path)
    
    # Save prompt
    prompt_path = os.path.join(epoch_dir, f"{file_prefix}.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

def save_batch_results(images, prompts, save_dir, epoch, method, batch_idx):
    """Batch save images and prompts
    
    Args:
        images: List of images
        prompts: List of prompts
        save_dir: Directory path for saving
        epoch: Current epoch
        method: Method name ('ori' or 'ours')
        batch_idx: Index of the current batch
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for sample_idx, (image, prompt) in enumerate(zip(images, prompts)):
        save_result(image, prompt, save_dir, epoch, method, batch_idx, sample_idx)
