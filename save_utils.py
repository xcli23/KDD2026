import os
from PIL import Image
import io

def save_result(image, prompt, save_dir, epoch, method, batch_idx, sample_idx):
    """保存图片和提示词到指定目录
    
    Args:
        image: PIL Image对象
        prompt: 提示词字符串
        save_dir: 保存目录路径
        epoch: 当前轮次
        method: 方法名称 ('ori' 或 'ours')
        batch_idx: 批次索引
        sample_idx: 样本在批次中的索引
    """
    # 创建方法目录
    method_dir = os.path.join(save_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    # 创建epoch目录
    epoch_dir = os.path.join(method_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # 构建唯一的文件名（基于batch和sample索引）
    file_prefix = f"sample_{batch_idx}_{sample_idx}"
    
    # 保存图片
    image_path = os.path.join(epoch_dir, f"{file_prefix}.png")
    if image is None:
        print(f"Warning: Image for {file_prefix} is None, skipping save.")
    else:
        image.save(image_path)
    
    # 保存提示词
    prompt_path = os.path.join(epoch_dir, f"{file_prefix}.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

def save_batch_results(images, prompts, save_dir, epoch, method, batch_idx):
    """批量保存图片和提示词
    
    Args:
        images: 图片列表
        prompts: 提示词列表
        save_dir: 保存目录路径
        epoch: 当前轮次
        method: 方法名称 ('ori' 或 'ours')
        batch_idx: 当前批次的索引
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for sample_idx, (image, prompt) in enumerate(zip(images, prompts)):
        save_result(image, prompt, save_dir, epoch, method, batch_idx, sample_idx)