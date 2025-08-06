import os
import torch
import numpy as np
import asyncio
import clip
import openai
import functools
import base64
import regex as re
import template
import time
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_datasets as tfds
import io 
import httpx 
from PIL import Image
from rouge import Rouge
from datasets import load_dataset
from openai import OpenAI
from mydataset import Dataset, subsampled_data
from torch.utils.data import DataLoader
from log import TrainingLogger
from configs import parse_args, set_all_seed
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from torch import autocast, nn
from alignscore import AlignScore
from bert_score import score as bert_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline, DDIMScheduler,DiffusionPipeline
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer, logging
from peft import LoraConfig, get_peft_model, PeftModel
from save_utils import save_batch_results, save_result
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
# -------------------- 软提示词优化器 --------------------
class SoftPromptOptimizer:
    """
    使用零阶优化方法(ZO-OGD)优化软提示
    """
    def __init__(self, whitebox_model, blackbox_model, evaluator, args):
        """
        初始化软提示优化器
        
        Args:
            whitebox_model: 白盒模型，用于生成优化的提示
            blackbox_model: 黑盒模型，用于生成最终结果
            evaluator: 评估器，用于计算生成结果的分数
            args: 参数配置
        """
        self.whitebox = whitebox_model
        self.blackbox = blackbox_model
        self.evaluator = evaluator
        self.args = args
        
        
        # 获取嵌入维度
        self.embed_dim = self.whitebox.embedding.shape[1]
        
        # 初始化软提示嵌入
        self.z_t = torch.zeros(args.intrinsic_dim).to(args.cuda)
        
        
        
        # 初始化软提示参数
        self.mu = getattr(args, "mu", 0.1)  # 扰动幅度
        self.lr = getattr(args, "soft_lr", 0.1)  # 学习率
        self.n_prompt_tokens = getattr(args, "n_prompt_tokens", 5)  # 软提示的token数量
        # 初始化投影矩阵
        self.A = self._initialize_projection_matrix()
        self.n_directions = getattr(args,'soft_n_directions',5)
    
    def _initialize_projection_matrix(self):
        """初始化从低维空间到高维嵌入空间的投影矩阵"""
        A = torch.nn.Linear(
            self.args.intrinsic_dim, 
            self.n_prompt_tokens * self.embed_dim, 
            bias=False
        ).to(self.args.cuda)
        
        # 初始化策略
        random_proj = getattr(self.args, "random_proj", "uniform")
        
        if random_proj == "normal":
            # 使用正态分布初始化
            mu_hat = self.whitebox.embedding.mean().item()
            std_hat = self.whitebox.embedding.std().item()
            
            # 计算缩放因子
            alpha = getattr(self.args, "alpha", 1.0)
            sigma = getattr(self.args, "sigma", 1.0)
            
            mu = 0.0
            std = alpha * std_hat / (np.sqrt(self.args.intrinsic_dim) * sigma)
            
            print(f"[Embedding] mu: {mu_hat}, std: {std_hat} [RandProj] mu: {mu}, std: {std}")
            torch.nn.init.normal_(A.weight, mean=mu, std=std)
            
        elif random_proj == "uniform":
            # 使用均匀分布初始化
            torch.nn.init.uniform_(A.weight, -1, 1)
        
        else:
            raise ValueError(f"Unknown random_proj type: {random_proj}")
        
        print(f"A weight mean: {A.weight.mean().item()}, std: {A.weight.std().item()}")
        return A
    
    def get_soft_prompt_embeds(self):
        """获取当前软提示的嵌入表示"""
        # 使用投影矩阵将低维表示映射到嵌入空间
        z_projected = self.A(self.z_t.unsqueeze(0))  # [1, n_prompt_tokens * embed_dim]
        
        # 重塑为 [1, n_prompt_tokens, embed_dim] 形状
        embeds = z_projected.view(1, self.n_prompt_tokens, self.embed_dim)
        
        # 确保嵌入类型与模型类型一致 (添加这行)
        embeds = embeds.to(self.whitebox.model.dtype)

        # 重塑为 [1, n_prompt_tokens, embed_dim] 形状
        return embeds
    
    async def optimize_step(self, prompts, batch_idx=None, epoch=None, output_dir=None, reference=None):
        """执行一步软提示优化"""
        z_t_tmp = self.z_t.unsqueeze(0)  # [1, intrinsic_dim]
        total_grad = torch.zeros_like(self.z_t)  # Initialize accumulator
        
        for _ in range(self.n_directions):

            # 生成随机噪声
            noise = torch.normal(mean=0.0, std=1.0, size=z_t_tmp.shape).to(self.args.device)
            # 产生正负扰动
            z_t_pos = z_t_tmp + self.mu * noise
            z_t_neg = z_t_tmp - self.mu * noise
            
            # 计算正负扰动的嵌入表示
            Az_pos = self.A(z_t_pos).view(1, self.n_prompt_tokens, self.embed_dim)
            Az_neg = self.A(z_t_neg).view(1, self.n_prompt_tokens, self.embed_dim)

            # 确保嵌入类型与模型类型一致
            Az_pos = Az_pos.to(self.whitebox.model.dtype)
            Az_neg = Az_neg.to(self.whitebox.model.dtype)
            # 使用白盒模型生成正负扰动的提示
            if args.dataset == "cnn_dailymail":
                black_prompts = "You are a professional text summarization bot. Condense the user-provided text into a 3-5 sentence summary using clear and concise language, preserving core facts and key information. Requirements: 1.Base strictly on original content without subjective interpretation. 2.Focus on main ideas, omit minor details. 3.The assistant provides sentence without additional content for user. Here is the article:\n"
                prompts = [black_prompts + p for p in prompts]
            pos_prompts = self.whitebox.generate(prompts, soft_prompt_embeds=Az_pos)
            neg_prompts = self.whitebox.generate(prompts, soft_prompt_embeds=Az_neg)
            
            # 使用黑盒模型生成最终结果
            pos_results = await self.blackbox.generate(pos_prompts)
            neg_results = await self.blackbox.generate(neg_prompts)
            
            # 评估结果
            if self.evaluator.eval_type == "image":
                pos_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(pos_prompts, pos_results)]
                neg_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(neg_prompts, neg_results)]
            else:
                pos_scores = [self.evaluator.evaluate(o, t, r)[self.args.metric] for o, t, r in zip(prompts, reference, pos_results)]
                neg_scores = [self.evaluator.evaluate(o, t, r)[self.args.metric] for o, t, r in zip(prompts, reference, neg_results)]
            # # 计算平均分数
            # pos_score_avg = np.mean(pos_scores)
            # neg_score_avg = np.mean(neg_scores)

        
            # print(f"{_}: Pos Score: {pos_score_avg:.4f}, Neg Score: {neg_score_avg:.4f}")
            
            # # 计算梯度估计
            # score_diff = pos_score_avg - neg_score_avg
            #g_t_hat = ((score_diff / (2 * self.mu)) * noise).squeeze(0)
            # 更新软提示参数
            #self.z_t = self.z_t + self.lr * g_t_hat
            score_diff = np.mean(pos_scores) - np.mean(neg_scores)
            
            # 累积梯度估计
            total_grad += (score_diff / (2 * self.mu)) * noise.squeeze(0)
        
        # 平均梯度
        avg_grad = total_grad / self.n_directions
        self.z_t += self.lr * avg_grad  # 更新参数
        
        # 使用当前的软提示生成结果
        current_soft_prompt = self.get_soft_prompt_embeds()
        current_prompts = self.whitebox.generate(prompts, soft_prompt_embeds=current_soft_prompt)
        current_results = await self.blackbox.generate(current_prompts)
        
        # 保存每批次结果（如果提供了批次信息和输出目录）
        if output_dir is not None and batch_idx is not None and epoch is not None and self.args.gene_image == 'True':
            save_batch_results(
                current_results,
                current_prompts,
                output_dir,
                epoch=epoch,
                method="soft_prompt",
                batch_idx=batch_idx
            )
        
        # 返回当前分数
        if self.evaluator.eval_type == "image":
            current_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(current_prompts, current_results)]
        else:
            current_scores = [self.evaluator.evaluate(o, t, r)[self.args.metric] for o, t, r in zip(current_prompts, reference, current_results)]
        
        current_score_avg = np.mean(current_scores)
        print(f"Current Score: {current_score_avg:.4f}")
        
        return {
            "score": current_score_avg,
            "soft_prompt": self.z_t.cpu().detach().clone(),
            "prompts": current_prompts,
            "results": current_results
        }
    
    async def optimize(self, prompts, epochs, batch_idx=None, output_dir=None, reference=None):
        """执行多轮优化"""
        best_score = -float('inf')
        best_z_t = None
        scores_history = []
        
        for epoch in range(epochs):
            print(f"Soft Prompt Optimization Epoch {epoch+1}/{epochs}")
            result = await self.optimize_step(
                prompts, 
                batch_idx=batch_idx, 
                epoch=epoch,
                output_dir=output_dir,
                reference=reference
            )
            scores_history.append(result["score"])
            
            # 保存最佳结果
            if result["score"] > best_score:
                best_score = result["score"]
                best_z_t = self.z_t.cpu().detach().clone()
            
            print(f"Epoch {epoch+1}, Score: {result['score']:.4f}, Best Score: {best_score:.4f}")
        
        # 恢复最佳结果
        if best_z_t is not None:
            self.z_t = best_z_t.to(self.args.device)
        
        return {
            "best_score": best_score,
            "best_soft_prompt": self.z_t,
            "scores_history": scores_history,
            "final_soft_prompt_embeds": self.get_soft_prompt_embeds()
        }
    
    def save(self, path):
        """保存优化后的软提示和投影矩阵"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            "z_t": self.z_t,
            "A_state_dict": self.A.state_dict(),
            "n_prompt_tokens": self.n_prompt_tokens,
            "embed_dim": self.embed_dim,
            "intrinsic_dim": self.args.intrinsic_dim
        }, os.path.join(path, "soft_prompt.pt"))
        print(f"Saved soft prompt to {path}")
    
    @classmethod
    def load(cls, path, whitebox_model, blackbox_model, evaluator, args):
        """加载保存的软提示和投影矩阵"""
        checkpoint = torch.load(os.path.join(path, "soft_prompt.pt"), map_location=whitebox_model.device)
        
        # 更新参数
        args.n_prompt_tokens = checkpoint["n_prompt_tokens"]
        args.intrinsic_dim = checkpoint["intrinsic_dim"]
        
        # 创建优化器实例
        optimizer = cls(whitebox_model, blackbox_model, evaluator, args)
        
        # 加载参数
        optimizer.z_t = checkpoint["z_t"].to(whitebox_model.device)
        optimizer.A.load_state_dict(checkpoint["A_state_dict"])
        
        return optimizer
# -------------------- 评价指标 --------------------

class TextEvaluator:
    def __init__(self, args):
        """文生文评价器，当前采用 F1-score"""
        self.args = args
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.cuda = args.cuda
        # BERTScore
        self.bert_model_type = getattr(args, "bert_model", "bert-base-chinese")
        self.bert_batch_size = getattr(args, "bert_batch_size", 32)

        # PPL
        gpt_name = getattr(args, "gpt_model", "gpt2")
        self.tok = GPT2Tokenizer.from_pretrained(gpt_name)
        self.lm = GPT2LMHeadModel.from_pretrained(gpt_name).to(self.cuda).eval()

        # AlignScore
        # self.scorer = AlignScore(model='alignscore-base', batch_size=32, device=args.cuda, ckpt_path='./cache/', evaluation_mode='nli_sp')
        self.scorer = AlignScore(model='roberta-base', batch_size=32, device=args.cuda, ckpt_path='./cache/alignscore/AlignScore-base.ckpt', evaluation_mode='nli_sp')
    def _bert_score(self, reference: str, result: str) -> float:
        P, R, F1 = bert_score(
            [result],
            [reference],
            model_type=self.bert_model_type,
            lang="zh",
            batch_size=self.bert_batch_size,
            verbose=False,
            device=self.cuda 
        )
        return F1.mean().item()

    def _ppl(self, result: str) -> float:
        print(f"result_type:{type(result)}, result:{type(result)}") 
        if result == '':
            print("Warning: 输入文本为空，无法计算 PPL")
            return 0.0
        ids = self.tok(result, return_tensors="pt", truncation=True, max_length=512).to(self.cuda) 
        with torch.no_grad():
            loss = self.lm(**ids, labels=ids["input_ids"]).loss
        print(f"计算 PPL,输入文本长度:{len(result)},对应分数：{loss.item()}")
        return torch.exp(loss).item()

    def _align_score(self, source: str, result: str) -> float:
        print(f"source:{type(source)}, result:{type(result)}")
        print(f"source-----------------------------------:{len(source)}, result:{len(result)}")
        # score = self.align_scorer.score([{"context": source, "claim": result}])[0]
        score = self.scorer.score(contexts=[source], claims=[result])[0]
        print(f"score:{score}, score_type:{type(score)}")
        return score
    
    def evaluate(self, ori, reference, result):
        """计算评价指标"""
        if reference == '':
            return {
            'total': 0,
            'bert_score': 0,
            'ppl_score': 0,
            'align_score': 0
        }
        # f1_score = self.get_f1_score(reference, result)
        # perplexity_score = self.calculate_perplexity(reference)
        # text2_score = self.get_text2_score(reference, result)
        bert_score = self._bert_score(reference, result)
        ppl_score = self._ppl(result)
        align_score = self._align_score(ori, result)
        final_score = (
            bert_score * self.lambda_1 + 
            ppl_score * self.lambda_2 + 
            align_score * self.lambda_3
        )
        # 返回详细分数
        return {
            'total': final_score,
            'bert_score': bert_score,
            'ppl_score': ppl_score,
            'align_score': align_score
        }

class AestheticMlp(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

class ImageEvaluator:
    """
    文生图评价器，使用 Aesthetic、CLIPScore、PickScore 三个指标，
    """
    def __init__(self, cache_dir, device, args):
        self.device = device
        # 初始化 CLIP 模型
        clip_cache_dir = os.path.join(cache_dir, "clip_model")
        # print(f"clip_cache_dir:{clip_cache_dir}")
        os.makedirs(clip_cache_dir, exist_ok=True)
        model_path = os.path.join(clip_cache_dir, "ViT-L-14.pt")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device, download_root=clip_cache_dir)
        
        # 初始化 Aesthetic 模型
        self.aes_model = AestheticMlp(768)
        state_dict = torch.load("./cache/aesthetic/sac+logos+ava1-l14-linearMSE.pth", map_location=device)
        self.aes_model.load_state_dict(state_dict)
        self.aes_model.to(device)
        self.aes_model.eval()

        # 初始化 PickScore 模型
        pickscore_cache_dir = os.path.join(cache_dir, "pickscore_model")
        os.makedirs(pickscore_cache_dir, exist_ok=True)
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path_pick = "yuvalkirstain/PickScore_v1"
        self.pickscore_processor = AutoProcessor.from_pretrained(processor_path, cache_dir=pickscore_cache_dir, device=device)
        self.pickscore_model = AutoModel.from_pretrained(model_path_pick, cache_dir=pickscore_cache_dir).eval().to(device)

        # 指标权重
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
    
    def get_clip_features(self, image, is_batched=False):
        if not is_batched:
            image = self.clip_preprocess(image).unsqueeze(0)
            image = image.to(self.device)
        else:
            images = [self.clip_preprocess(i) for i in image]
            image = torch.stack(images).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
        return image_features

    def get_clip_score(self, image_features, prompt):
        tokens = clip.tokenize(prompt[:77], truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens[:77])
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            score = (image_features @ text_features.t()).item()
        return score

    def get_aesthetic_score(self, image_features):
        features = image_features.cpu().detach().numpy()
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        norm[norm == 0] = 1
        normalized = features / norm
        tensor_features = torch.tensor(normalized, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            prediction = self.aes_model(tensor_features)
        return prediction.item()

    def get_pick_score(self, prompt, image):
        image_inputs = self.pickscore_processor(images=image, return_tensors="pt").to(self.device)
        text_inputs = self.pickscore_processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embs = self.pickscore_model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = self.pickscore_model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            score = (self.pickscore_model.logit_scale.exp() * (text_embs * image_embs).sum()).item()
        return score

    def evaluate(self, prompt, image):
        if image is None:
            return {
                'total': 0,
                'aesthetic': 0,
                'clip': 0,
                'pick': 0
            }
        features = self.get_clip_features(image)
        aes_score = self.get_aesthetic_score(features)
        clip_score = self.get_clip_score(features, prompt)
        pick_score = self.get_pick_score(prompt, image)
        final_score = (
            aes_score * self.lambda_1 + 
            clip_score * self.lambda_2 + 
            pick_score * self.lambda_3
        )
        # 返回详细分数
        return {
            'total': final_score,
            'aesthetic': aes_score,
            'clip': clip_score,
            'pick': pick_score
        }

class Evaluator:
    """
      - eval_type=="text"：使用 TextEvaluator
      - eval_type=="image"：使用 ImageEvaluator
    """
    def __init__(self, eval_type, **kwargs):
        self.eval_type = eval_type
        if eval_type == "text":
            self.evaluator = TextEvaluator(**kwargs)
        elif eval_type == "image":
            self.evaluator = ImageEvaluator(**kwargs)
        else:
            raise ValueError("Unknown evaluation type")

    def evaluate(self, *args):
        if self.eval_type == "text":
            return self.evaluator.evaluate(ori = args[0], reference = args[1], result = args[2])
        else:
            return self.evaluator.evaluate( args[0], args[1])
# -------------------- 白盒模型：Prompt 生成器（附加 LoRA） --------------------
class WhiteBoxModel:
    MODEL_TARGET_MAP = {
        "llama": ["q_proj", "v_proj"],
        "vicuna": ["q_proj", "v_proj"],
        "gpt2": ["c_attn"],
        "bert": ["query", "value"],
        "t5": ["q", "v"],
        "default": ["k_proj", "v_proj"]
    }

    MODEL_LAYER_PATTERNS = {
        "llama": r'\.layers\.(\d+)',
        "vicuna": r'\.layers\.(\d+)',
        "gpt2": r'h\.(\d+)',
        "bert": r'encoder\.layer\.(\d+)',
        "t5": r'block\.(\d+)',
        "default": r'\.layers\.(\d+)'
    }

    def __init__(self, model_name: str, hf_token: str = None, n_last_layers: int = 1, device: str = None, lora_rank: int = 4, blackbox_mode: str = "image"):
        self.blackbox_mode = blackbox_mode
        self.model_type = self._detect_model_type(model_name)
        self.model, self.tokenizer = self._load_model(model_name, hf_token, device)
        self._inject_lora(n_last_layers=n_last_layers, lora_rank=lora_rank)
        self.embedding = self.model.get_input_embeddings().weight.clone().to(device)

    def load_lora(self, lora_dir: str):
        """加载训练好的LoRA适配器"""
        # 检查适配器文件
        required_files = ["adapter_model.safetensors", "adapter_config.json"]
        for f in required_files:
            if not os.path.exists(os.path.join(lora_dir, f)):
                print(f"os.path.exists(os.path.join(lora_dir, f)):{os.path.join(lora_dir, f)}")
                raise FileNotFoundError(f"Missing LoRA file: {f}")
     
        # 加载适配器
        self.model.load_adapter(lora_dir, adapter_name="adapter_model")
        self.model.set_adapter("adapter_model")
        print(f"成功加载LoRA适配器: {lora_dir}")

    def merge_lora(self):
        """合并LoRA参数到基础模型"""
        original_weight = self.model.transformer.h[-1].attn.c_attn.weight.clone()
        self.model.merge_and_unload()
        merged_weight = self.model.transformer.h[-1].attn.c_attn.weight
        if torch.allclose(original_weight, merged_weight, atol=1e-5):
            raise RuntimeError("LoRA参数合并失败!")
        print("LoRA参数合并成功,模型已准备好推理")

    def _detect_model_type(self, model_name: str) -> str:
        model_name = model_name.lower()
        if "llama" in model_name:
            return "llama"
        elif "vicuna" in model_name:
            return "vicuna"
        elif "gpt2" in model_name:
            return "gpt2"
        elif "promtist" in model_name:
            return "gpt2"
        elif "sft" in model_name:
            return "gpt2"
        elif "bert" in model_name:
            return "bert"
        elif "t5" in model_name:
            return "t5"
        return "default"


    def _load_model(self, model_name: str, hf_token: str, device: str = None):
        if "vicuna" in model_name.lower():
            return self._load_vicuna_model(model_name, hf_token, device)
        
        if "promtist" in model_name.lower():
            return self.load_promtist(model_name, hf_token, device)
        
        if "sft" in model_name.lower():
            return self.load_sft(model_name, hf_token, device)
        
        if hf_token:
            from huggingface_hub import login, HfFolder
            login(token=hf_token)
            HfFolder.save_token(hf_token)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map= device,
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f"./cache/{model_name}")
        tokenizer.pad_token = tokenizer.eos_token if not tokenizer.pad_token else tokenizer.pad_token
        return model, tokenizer
    def load_promtist(self, model_name: str, hf_token: str = None, device: str = None):
        print(f"load_promtist模型，使用gpt2架构")
        prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist", device_map = device, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=f"./cache/{model_name}")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        return prompter_model, tokenizer
    
    def load_sft(self, model_name: str, hf_token: str = None, device: str = None):
        print("Loading SFT model...")
        state_dict_path = "./cache/sft/sft_gpt.bin"
        
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_device = torch.device(target_device)

        def load_sft_model(state_dict_path, base_model_name="gpt2", device=torch_device):
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16
            ).to(device)
            state_dict = torch.load(state_dict_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model

        sft_model = load_sft_model(state_dict_path, device=torch_device)
        
        sft_model.to(torch_device)
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        param_device = next(sft_model.parameters()).device
        print(f"模型参数已加载到: {param_device}")
        
        print("SFT model loaded successfully.")
        return sft_model, tokenizer
    
    def _load_vicuna_model(self, model_name: str, hf_token: str = None, device: str = None):
        """
          "lmsys/vicuna-13b-v1.3"
        """
        from huggingface_hub import snapshot_download, HfApi

        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        try:
            if not os.path.exists(model_name):

                if not model_name.startswith("lmsys/"):
                    repo_id = f"lmsys/{model_name}"
                else:
                    repo_id = model_name
                
                try:
                    HfApi().model_info(repo_id, token=hf_token)
                except Exception as e:
                    raise ValueError(
                        f"无法访问模型 {repo_id}"
                    ) from e

                model_path = snapshot_download(
                    repo_id=repo_id,
                    token=hf_token,
                    ignore_patterns=["*.safetensors"],  
                    resume_download=True,
                    local_dir = f"./cache/{repo_id}"
                )
            else:
                model_path = model_name

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                token=hf_token,
                model_max_length=1024,
                padding_side="left"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map= device,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                token=hf_token
            )

            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            
            return model, tokenizer

        except Exception as e:
            raise RuntimeError(
                f"加载Vicuna模型失败:{str(e)}"
            ) from e
    
    def _inject_lora(self, lora_rank: int = 4, custom_targets: list = None, n_last_layers: int = 1):
        base_targets = custom_targets or self.MODEL_TARGET_MAP.get(self.model_type, self.MODEL_TARGET_MAP["default"])
        
        # 筛选候选模块
        candidate_modules = []
        for name, _ in self.model.named_modules():
            if any(target in name for target in base_targets):
                candidate_modules.append(name)
        
        # 筛选后n层
        layer_pattern = self.MODEL_LAYER_PATTERNS[self.model_type]
        layer_re = re.compile(layer_pattern)
        
        layer_info = []
        for name in candidate_modules:
            match = layer_re.search(name)
            if match:
                layer_num = int(match.group(1))
                layer_info.append((layer_num, name))
        
        if not layer_info:
            raise ValueError("未找到符合条件的目标模块")
        
        max_layer = max(layer_num for layer_num, _ in layer_info)
        selected_layers = range(max_layer - n_last_layers + 1, max_layer + 1)
        target_modules = [name for layer_num, name in layer_info if layer_num in selected_layers]
        
        # 创建LoRA配置
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # 初始化参数
        for name, param in self.model.named_parameters():
            # if 'lora_A' in name or 'lora_B' in name:
            if 'lora_B' in name:
                nn.init.zeros_(param)
        
        self.model.print_trainable_parameters()
        for name, param in self.model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                print(f"Parameter {name} set_flat_params: {param.data}")

    def get_flat_params(self) -> torch.Tensor:
        lora =  torch.cat([p.detach().flatten() for p in self.model.parameters() if p.requires_grad])
        # print(f"get_flat_lora :{lora} ")
        return lora

    def set_flat_params(self, flat_params: torch.Tensor):
        ptr = 0
        for p in self.model.parameters():
            if p.requires_grad:
                numel = p.numel()
                p.data.copy_(flat_params[ptr:ptr+numel].reshape(p.shape))
                ptr += numel

    def generate(self, prompts: list, max_new_tokens: int = 75, soft_prompt_embeds=None) -> list:
        start_time = time.time()
        if self.blackbox_mode == "text":
            max_new_tokens = 300
        print(f"白盒模型输入：\n{prompts}")
        
        if self.model_type == "gpt2":
            new_prompts = [p + " Rephrase:" for p in prompts]
            
            input_ids = self.tokenizer(
                new_prompts, 
                return_tensors="pt", 
                padding=True
            ).input_ids.to(self.model.device)
            
            if soft_prompt_embeds is not None:
                # 获取输入嵌入
                input_embeds = self.model.get_input_embeddings()(input_ids)
                
                # 将软提示嵌入插入到输入嵌入的适当位置
                batch_size = input_embeds.shape[0]
                if soft_prompt_embeds.shape[0] == 1 and batch_size > 1:
                    soft_prompt_embeds = soft_prompt_embeds.repeat(batch_size, 1, 1)
                
                # 拼接: [BOS, soft_prompt, input_text]
                input_embeds_with_soft = torch.cat([
                    input_embeds[:, :1, :],  # 保留开始标记 (BOS)
                    soft_prompt_embeds,      # 插入软提示
                    input_embeds[:, 1:, :]   # 保留剩余的输入
                ], dim=1)
                
                # 更新 attention_mask
                attention_mask = torch.ones(
                    (batch_size, input_embeds_with_soft.shape[1]),
                    device=input_embeds.device
                )
                
                # 使用嵌入生成文本
                outputs = self.model.generate(
                    inputs_embeds=input_embeds_with_soft,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    num_beams=8,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    length_penalty=-1.0
                )
            else:
                # 不使用软提示，直接使用 input_ids 生成
                outputs = self.model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    num_beams=8,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    length_penalty=-1.0
                )
            
            # 处理输出
            output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print(f"白盒模型原始输出：\n{output_texts}")
            
            results = []
            for i, text in enumerate(output_texts):
                # print(f"白盒模型原始输出{i}:{text}")
                processed = text.replace(new_prompts[i], "").strip()
                results.append(processed)

            print(f"白盒模型处理后输出：\n{results}")
            print(f"白盒生成耗时: {time.time()-start_time:.2f}s")
            return results 
        
        input_token = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = input_token.input_ids.to(self.model.device)
        attention_mask = input_token.attention_mask.to(self.model.device)

        if soft_prompt_embeds is not None:
            # 获取输入嵌入
            input_embed = self.embedding[input_ids]
            
            # 确保 soft_prompt_embeds 的 batch_size 与 input_embed 匹配
            batch_size = input_embed.shape[0]
            if soft_prompt_embeds.shape[0] == 1 and batch_size > 1:
                soft_prompt_embeds = soft_prompt_embeds.repeat(batch_size, 1, 1)
            
            # 拼接: [BOS, soft_prompt, input_text]
            input_embed_with_soft = torch.cat([
                input_embed[:, :1, :],  # 保留开始标记 (BOS)
                soft_prompt_embeds,     # 插入软提示
                input_embed[:, 1:, :]   # 保留剩余的输入
            ], dim=1)
            
            # 更新 attention_mask
            soft_prompt_length = soft_prompt_embeds.shape[1]
            soft_attention_mask = torch.ones(
                (batch_size, soft_prompt_length),
                device=attention_mask.device
            )
            attention_mask_with_soft = torch.cat([
                attention_mask[:, :1],    # 保留开始标记的 mask
                soft_attention_mask,      # 软提示的 mask
                attention_mask[:, 1:]     # 保留剩余输入的 mask
            ], dim=1)
            
            # 使用嵌入生成文本
            outputs = self.model.generate(
                inputs_embeds=input_embed_with_soft,
                attention_mask=attention_mask_with_soft,
                do_sample=False, 
                max_new_tokens=max_new_tokens, 
                num_beams=8, 
                num_return_sequences=1,  
                eos_token_id=self.tokenizer.eos_token_id, 
                pad_token_id=self.tokenizer.eos_token_id, 
                length_penalty=-1.0
            )
        else:
            # 不使用软提示，直接使用 input_embed 生成
            input_embed = self.embedding[input_ids]
            outputs = self.model.generate(
                inputs_embeds=input_embed,
                attention_mask=attention_mask,
                do_sample=False, 
                max_new_tokens=max_new_tokens, 
                num_beams=8, 
                num_return_sequences=1,  
                eos_token_id=self.tokenizer.eos_token_id, 
                pad_token_id=self.tokenizer.eos_token_id, 
                length_penalty=-1.0
            )
        
        results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(f"白盒模型输出：\n{results}")
        print(f"白盒生成耗时: {time.time()-start_time:.2f}s")
        return results
# -------------------- 黑盒模型：支持文生文与文生图 --------------------
class BlackBoxModel:
    """
    黑盒模型接口，支持三种模式：
      - model_type=="text"：例如使用 gpt2-xl 或 Claude 4 生成文本
      - model_type=="image"：例如使用 Stable Diffusion 1.4 生成图像
    """
    def __init__(self, model_type: str = "text", model_name: str = None, device: str = "cuda:0", batch_size: int = 1, black_token: str = None, max_workers: int = 4, api_base: str = None):
        self.device = device
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.model_type = model_type
        self.api_key = black_token
        self.api_base = api_base

        if model_type == "text":
            if "gpt" in model_name.lower():
                self.openai_model = model_name
                self.model = None
                self.tokenizer = None
            elif model_name.lower() == "claude-3-5-sonnet-all":
                self.openai_model = model_name  # 使用 OpenAI 兼容的 Claude 模型名称
                self.model = None
                self.tokenizer = None
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
        elif model_type == "image":
            if model_name == "gpt-image-1":
                self.openai_model = model_name
            elif model_name == "stabilityai/stable-diffusion-xl-base-1.0":
                self.pipe = DiffusionPipeline.from_pretrained(
                    model_name, torch_dtype=torch.float16
                ).to(device)
                self.pipe.set_progress_bar_config(disable=True)
            else:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    use_auth_token=True,
                ).to(device)
                self.pipe.set_progress_bar_config(disable=True)
        else:
            raise ValueError("Unknown model_type for BlackBoxModel")

    async def generate(self, prompts: list, **kwargs) -> list:
        print(f"黑盒模型输入：{prompts}")
        results = []
        
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i+self.batch_size]
            try:
                if self.model_type == "text":
                    batch_results = await self._generate_text_batch(batch, kwargs)
                elif self.model_type == "image":
                    batch_results = await self._generate_image_batch(batch, kwargs)
                results.extend(batch_results)
            except RuntimeError as e:
                print(f"黑盒生成失败: {str(e)}")
                results.extend([None]*len(batch))
                
        return results
    
    async def _generate_text_batch(self, batch, gen_kwargs):
        if hasattr(self, 'openai_model') and self.openai_model is not None:
            # 统一使用 OpenAI 格式的 API 调用 Claude
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base if self.api_base else "https://api.openai.com/v1"  # 如果 api_base 未指定，默认用 OpenAI
            )
            
            responses = []
            for prompt in batch:
                try:
                    # 使用 OpenAI 格式调用 Claude
                    response = client.chat.completions.create(
                        model=self.openai_model,  # 例如 "claude-4"
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=gen_kwargs.get("max_new_tokens", 2000),
                        temperature=0.7,
                    )
                    responses.append(response.choices[0].message.content.strip())
                except Exception as e:
                    print(f"API 调用失败: {str(e)}")
                    responses.append(None)
            
            print(f"黑盒模型生成成功（{self.openai_model}）输出：{responses}")
            return responses
        else:
            # 本地模型生成
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_kwargs.get("max_new_tokens", 300),
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    async def _generate_openai_image_batch(self, prompts: list, kwargs: dict) -> list:

        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base 
        )

        size   = kwargs.pop("size", "1024x1024")
        quality= kwargs.pop("quality", "low")
        n      = kwargs.pop("num_images_per_prompt", 1)

        async def _single(prompt: str):
            try:
                response = await client.images.generate(
                    model="gpt-image-1",
                    prompt=prompt,
                    n=n,
                    quality=quality,
                    size=size
                )
                b64_str = response.data[0].b64_json
                if b64_str is None:
                    return None
                raw = base64.b64decode(b64_str)
                return Image.open(io.BytesIO(raw))
            except Exception:
                return None

        result = await asyncio.gather(*[_single(p) for p in prompts])

        return result

    async def _generate_image_batch(self, batch, kwargs):
        """优化图像批量生成"""
        if hasattr(self, 'openai_model') and self.openai_model == "gpt-image-1":
            return await self._generate_openai_image_batch(batch, kwargs)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        generator = torch.Generator(self.device).manual_seed(1234)
        # 批量生成
        if hasattr(self.pipe, '_encode_prompt'):
            with autocast(self.device):
                outputs = self.pipe(
                    prompt=batch,
                    num_images_per_prompt=1,
                    guidance_scale=15,
                    eta=0.0,
                    num_inference_steps=70,
                    generator=generator,
                    **kwargs
                )
                return outputs.images
        else:
            # 内存不够的话，逐个生成
            return await asyncio.gather(*[
                self._run_image_generation([prompt], kwargs) 
                for prompt in batch
            ])

    def _run_text_generation(self, batch: list, gen_kwargs: dict) -> list:
        if hasattr(self, 'openai_model') and self.openai_model is not None:
            if not hasattr(self, 'api_key') or not self.api_key:
                raise ValueError("API key is required for OpenAI API calls")
                
            results = []
            for prompt in batch:
                client = OpenAI(
                    api_key=self.api_key, 
                    base_url=self.api_base if hasattr(self, 'api_base') and self.api_base else None
                )
                response = client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False
                )
                results.append(response.choices[0].message.content.strip())
            print(f"黑盒模型{self.openai_model}输出：{results}")
            return results
        else:
            # 原有本地模型代码
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            outputs = self.model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **gen_kwargs)
            return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]

    # def _run_image_generation(self, batch: list, gen_kwargs: dict) -> list:
    #     results = []
    #     for prompt in batch:
    #         with autocast(self.pipe.device.type):
    #             image = self.pipe(prompt, **gen_kwargs).images[0]
    #         results.append(image)
    #     return results
    async def _run_image_generation(self, batch: list, gen_kwargs: dict) -> list:
        image = await asyncio.to_thread(self.pipe, batch, **gen_kwargs)
        image = image.images[0] if hasattr(image, 'images') else image
        return image

# -------------------- 零阶优化器 --------------------
class MomentumOptimizer:
    """
    使用对称差分法估计梯度，再用动量法更新白盒模型中 LoRA 参数
    """
    def __init__(self, whitebox: WhiteBoxModel, blackbox: BlackBoxModel, evaluator: Evaluator, args):
        self.whitebox = whitebox
        self.blackbox = blackbox
        self.evaluator = evaluator
        self.h = args.h
        self.n_directions = args.n_directions
        self.beta = args.beta
        self.velocity = torch.zeros_like(self.whitebox.get_flat_params())
        self.grad_history = []      # 存储原始梯度张量
        self.grad_metadata = []     # 存储统计元数据
        self.n_components = 50      # PCA降维保留的主成分数
        self.args = args

    def _store_gradient(self, grad: torch.Tensor):
        """梯度存储核心方法"""
        # 存储完整梯度
        self.grad_history.append(grad.cpu().clone())
        
        # 存储统计元数据（节省内存）
        stats = {
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'norm': torch.norm(grad).item(),
            'min': grad.min().item(),
            'max': grad.max().item()
        }
        self.grad_metadata.append(stats)
        
        # 控制存储长度防止内存爆炸
        if len(self.grad_history) > 100:
            self.grad_history.pop(0)
            self.grad_metadata.pop(0)
    def analyze_gradients(self, output_dir: str = "./grad_analysis"):
        """梯度分析入口方法"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 基础统计
        self._plot_basic_stats(os.path.join(output_dir, "basic_stats.png"))
        
        # 维度级分析
        self._analyze_dimensions(os.path.join(output_dir, "dimension_analysis.txt"))
        
        # 相关性分析
        self._correlation_analysis(os.path.join(output_dir, "correlation_analysis.png"))
        
        # 方向一致性分析
        self._direction_consistency(os.path.join(output_dir, "direction_heatmap.png"))

    def _plot_basic_stats(self, save_path: str):
        """绘制基础统计图表"""
        
        # 提取元数据
        steps = np.arange(len(self.grad_metadata))
        means = [m['mean'] for m in self.grad_metadata]
        stds = [m['std'] for m in self.grad_metadata]
        norms = [m['norm'] for m in self.grad_metadata]

        plt.figure(figsize=(15, 5))
        
        # 均值变化
        plt.subplot(1, 3, 1)
        plt.plot(steps, means, label='Mean')
        plt.title("Gradient Mean")
        plt.xlabel("Step")
        
        # 标准差变化
        plt.subplot(1, 3, 2)
        plt.plot(steps, stds, color='orange', label='Std')
        plt.title("Gradient Std")
        plt.xlabel("Step")
        
        # 模长变化
        plt.subplot(1, 3, 3)
        plt.plot(steps, norms, color='green', label='Norm')
        plt.title("Gradient Norm")
        plt.xlabel("Step")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _analyze_dimensions(self, save_path: str):
        """维度级统计分析"""
        # 拼接所有梯度
        grad_matrix = torch.stack(self.grad_history).numpy()  # [n_steps, n_dims]
        
        # 计算各维度统计量
        dim_means = np.mean(grad_matrix, axis=0)
        dim_stds = np.std(grad_matrix, axis=0)
        
        # 找出最活跃维度
        top10_indices = np.argsort(np.abs(dim_means))[-10:][::-1]
        
        # 保存分析结果
        with open(save_path, 'w') as f:
            f.write("=== 维度级分析报告 ===\n")
            f.write(f"总参数维度: {grad_matrix.shape[1]}\n")
            f.write(f"平均梯度绝对值均值: {np.mean(np.abs(dim_means)):.4f}\n")
            f.write(f"最大梯度均值维度: {top10_indices[0]} (值={dim_means[top10_indices[0]]:.4f})\n")
            f.write(f"最不稳定维度: {np.argmax(dim_stds)} (标准差={np.max(dim_stds):.4f})\n")
            
            f.write("\nTop 10活跃维度:\n")
            for idx in top10_indices:
                f.write(f"Dim {idx}: mean={dim_means[idx]:.4f}, std={dim_stds[idx]:.4f}\n")

    def _correlation_analysis(self, save_path: str):
        """梯度间相关性分析"""
        import seaborn as sns
        
        # 随机采样部分梯度
        sample_indices = np.random.choice(len(self.grad_history), size=50, replace=False)
        grad_samples = [self.grad_history[i].numpy() for i in sample_indices]
        
        # 计算余弦相似度矩阵
        cos_sim = np.zeros((len(grad_samples), len(grad_samples)))
        for i in range(len(grad_samples)):
            for j in range(len(grad_samples)):
                cos_sim[i,j] = np.dot(grad_samples[i], grad_samples[j]) / (
                    np.linalg.norm(grad_samples[i]) * np.linalg.norm(grad_samples[j]))
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cos_sim, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Gradient Direction Similarity")
        plt.savefig(save_path)
        plt.close()

    def _direction_consistency(self, save_path: str):
        """主成分方向分析"""
        from sklearn.decomposition import PCA
        
        # 构建梯度矩阵
        grad_matrix = torch.stack(self.grad_history).numpy()
        
        # PCA降维
        pca = PCA(n_components=self.n_components)
        pca.fit(grad_matrix)
        
        # 解释方差比
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Analysis of Gradients')
        plt.savefig(save_path)
        plt.close()

    async def estimate_gradient(self, original_prompts: list, reference = None) -> torch.Tensor:
        original_params = self.whitebox.get_flat_params()
        total_grad = torch.zeros_like(original_params)
        param_dim = original_params.shape[0]

        for _ in range(self.n_directions):
            # while True:
            delta = torch.randn(param_dim).to(original_params.device) * self.h
            # print(f"扰动大小 ： {delta}")
            # 正方向扰动
            # for name, param in self.whitebox.model.named_parameters():
            #     if 'lora_A' in name and 'ayers.0.self_attn' in name:
            #         print(f"Parameter {name} no set_flat_params: {param.data}")
                    
            self.whitebox.set_flat_params(original_params + delta)
            pos_prompts = self.whitebox.generate(original_prompts)
            # 负方向扰动
            self.whitebox.set_flat_params(original_params - delta)
            neg_prompts = self.whitebox.generate(original_prompts)
                # if pos_prompts != neg_prompts:
                #     break
                # print('输入相等重新生成样本')
            if args.dataset == "cnn_dailymail":
                black_prompts = "You are a professional text summarization bot. Condense the user-provided text into a 3-5 sentence summary using clear and concise language, preserving core facts and key information. Requirements: 1.Base strictly on original content without subjective interpretation. 2.Focus on main ideas, omit minor details. 3.The assistant provides sentence without additional content for user. Here is the article:\n"
                pos_prompts = [black_prompts + p for p in pos_prompts]
            pos_results = await self.blackbox.generate(pos_prompts)
            if pos_results == None:
                print("黑盒输出为空, 差分法无法计算梯度")
                continue
            if self.evaluator.eval_type == "image":
                pos_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(pos_prompts, pos_results)]
            else:
                pos_scores = [self.evaluator.evaluate(o, t, r)[self.args.metric] for o, t, r in zip(original_prompts, reference, pos_results)]
            pos_score_avg = np.mean(pos_scores)
            print(f"pos_score_avg :{pos_score_avg}")
            if args.dataset == "cnn_dailymail":
                black_prompts = "You are a professional text summarization bot. Condense the user-provided text into a 3-5 sentence summary using clear and concise language, preserving core facts and key information. Requirements: 1.Base strictly on original content without subjective interpretation. 2.Focus on main ideas, omit minor details. 3.The assistant provides sentence without additional content for user. Here is the article:\n"
                neg_prompts = [black_prompts + p for p in neg_prompts]
            neg_results = await self.blackbox.generate(neg_prompts)

            if neg_results == None:
                print("黑盒输出为空, 差分法无法计算梯度")
                continue
            if self.evaluator.eval_type == "image":
                neg_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(neg_prompts, neg_results)]
            else:
                neg_scores = [self.evaluator.evaluate(o, t, r)[self.args.metric] for o, t, r in zip(original_prompts,reference, neg_results)]
            neg_score_avg = np.mean(neg_scores)
            print(f"neg_score_avg :{neg_score_avg}")
            score_diff = pos_score_avg - neg_score_avg
            print(f"n_grad :{(score_diff / (2 * self.h)) * delta}")
            total_grad += (score_diff / (2 * self.h)) * delta

        # 恢复原lora参数
        self.whitebox.set_flat_params(original_params)
        final_grad = total_grad / self.n_directions
        print(f"final_grad :{final_grad}")
        # self._store_gradient(final_grad)  # 用来计算梯度指标
        return final_grad
    async def spsa_gradient(self, original_prompts: list, reference) -> torch.Tensor:
        # 获取原始参数和参数维度
        original_params = self.whitebox.get_flat_params()
        total_grad = torch.zeros_like(original_params)
        param_dim = original_params.shape[0]

        for _ in range(self.n_directions):
            # while True:
            #     # 使用 Rademacher 分布生成扰动（每个元素随机为 -1 或 +1）
            #     delta = torch.randint(0, 2, (param_dim,), device=original_params.device, dtype=torch.float32) * 2 - 1
            #     # 正方向扰动
            #     self.whitebox.set_flat_params(original_params + self.h * delta)
            #     pos_prompts = self.whitebox.generate(original_prompts)

            #     # 负方向扰动
            #     self.whitebox.set_flat_params(original_params - self.h * delta)
            #     neg_prompts = self.whitebox.generate(original_prompts)
            #     #如果不等则计算梯度
            #     if pos_prompts != neg_prompts:
            #         break
            #     print('输入相等重新生成样本')

            delta = torch.randint(0, 2, (param_dim,), device=original_params.device, dtype=torch.float16) * 2 - 1
            # 两个方向扰动
            self.whitebox.set_flat_params(original_params + self.h * delta)
            pos_prompts = self.whitebox.generate(original_prompts)
            self.whitebox.set_flat_params(original_params - self.h * delta)
            neg_prompts = self.whitebox.generate(original_prompts)

            if args.dataset == "cnn_dailymail":
                black_prompts = "You are a professional text summarization bot. Condense the user-provided text into a 3-5 sentence summary using clear and concise language, preserving core facts and key information. Requirements: 1.Base strictly on original content without subjective interpretation. 2.Focus on main ideas, omit minor details. 3.The assistant provides sentence without additional content for user. Here is the article:\n"
                pos_prompts = [black_prompts + p for p in pos_prompts]
            pos_results = await self.blackbox.generate(pos_prompts)
            if pos_results == None:
                print("黑盒输出为空, 差分法无法计算梯度")
                continue
            if self.evaluator.eval_type == "image":
                pos_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(pos_prompts, pos_results)]
            else:
                pos_scores = [self.evaluator.evaluate(o, t, r)[self.args.metric] for o, t, r in zip(original_prompts, reference, pos_results)]
            pos_score_avg = np.mean(pos_scores)
            print(f"pos_score_avg :{pos_score_avg}")

            if args.dataset == "cnn_dailymail":
                black_prompts = "You are a professional text summarization bot. Condense the user-provided text into a 3-5 sentence summary using clear and concise language, preserving core facts and key information. Requirements: 1.Base strictly on original content without subjective interpretation. 2.Focus on main ideas, omit minor details. 3.The assistant provides sentence without additional content for user. Here is the article:\n"
                neg_prompts = [black_prompts + p for p in neg_prompts]
            neg_results = await self.blackbox.generate(neg_prompts)
            if neg_results == None:
                print("黑盒输出为空, 差分法无法计算梯度")
                continue
            if self.evaluator.eval_type == "image":
                neg_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(neg_prompts, neg_results)]
            else:
                neg_scores = [self.evaluator.evaluate(o, t, r)[self.args.metric] for o, t, r in zip(original_prompts, reference, neg_results)]
            neg_score_avg = np.mean(neg_scores)
            print(f"neg_score_avg :{neg_score_avg}")
            # 计算当前方向上的得分差值
            score_diff = pos_score_avg - neg_score_avg

            # SPSA 梯度估计：
            # 对于每个坐标 i，有: (score_diff)/(2*h*delta[i])，而 delta[i] ∈ {+1, -1}，
            # 所以这里直接写为 (score_diff)/(2*h) * delta
            print(f"n_grad :{(score_diff / (2 * self.h)) * delta}")
            total_grad += (score_diff / (2 * self.h)) * delta

        # 恢复原始参数
        self.whitebox.set_flat_params(original_params)
        final_grad = total_grad / self.n_directions
        print(f"final_grad :{final_grad}")
        # self._store_gradient(final_grad)  # 用于计算梯度指标
        return final_grad


    def update_params(self, gradient: torch.Tensor, lr: float = 0.01):
        self.velocity = self.beta * self.velocity + (1 - self.beta) * gradient
        new_params = self.whitebox.get_flat_params() + lr * self.velocity
        self.whitebox.set_flat_params(new_params)

#--------------------- 主函数------------------------
async def main(args):
    logger = TrainingLogger(args)


    train_dataset = Dataset(f"./dataset/{args.dataset}_train.csv") # 实际大小为256
    test_dataset = Dataset(f"./dataset/{args.dataset}_test.csv")  # 实际大小为544

    if args.train_size > len(train_dataset):
        print(f"Warning: train_size ({args.train_size}) exceeds maximum available {len(train_dataset)}. Setting to {len(train_dataset)}.")
        train_size = len(train_dataset)
    else:
        train_size = args.train_size

    if args.test_size > len(test_dataset):
        print(f"Warning: test_size ({args.train_size}) exceeds maximum available {len(test_dataset)}. Setting to {len(test_dataset)}.")
        test_size = len(test_dataset)
    else:
        test_size = args.test_size
    
    _, train_dataset = train_test_split(train_dataset, test_size=train_size/len(train_dataset), random_state=args.seed)
    _, test_dataset = train_test_split(test_dataset, test_size=test_size/len(test_dataset), random_state=args.seed)

    print(f"Dataset sizes - Training: {len(train_dataset)} samples, Testing: {len(test_dataset)} samples")
    batch_size = getattr(args, "batch_size", 1)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 构造模板，利用已有的full_DEMO和待改写的original_prompt
    demos_template = "Original: [INPUT]\nRephrase: [OUTPUT]"
    if args.dataset == "cnn_dailymail":
        # demos_template = "Original: [INPUT]\nSummary: [OUTPUT]"
        demos_template = "Summary: [OUTPUT]"
    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data[args.example])
    init_prompt = ['\n']
    if args.ptype == 0:
        prompt_gen_str = "[full_DEMO]\n\nBased on the rephrasing way in the examples above, rephrase this sentence with consistency guaranteed:\n[original_prompt]\n"
    elif args.ptype == 1:
        prompt_gen_str = "[full_DEMO]\n\nBased on the rephrasing way in the examples above, rephrase this sentence:\n[original_prompt]\n"
    elif args.ptype == 2:
        prompt_gen_str = "[full_DEMO]\n\nBased on the rephrasing way in the examples above, using your creativity to rephrase this sentence:\n[original_prompt]\n"
    elif args.ptype == 3:
        prompt_gen_str = "[full_DEMO]\n\nIn order to make the diffusion model generate better pictures, based on the rephrasing way in the examples above, rephrase this sentence:\n[original_prompt]\n"
    elif args.ptype == 5:
        prompt_gen_str = "[full_DEMO]\n\nIn order to make the diffusion model generate better pictures, based on the rephrasing way in the examples above, using your creativity rather than just applying the example content to rephrase this sentence:\n[original_prompt]\n"
        # prompt_gen_str = "In order to make the diffusion model generate better pictures, based on the rephrasing way in the examples above, using your creativity rather than just applying the example content to rephrase this sentence:\n[original_prompt]\n"
    if args.dataset == "cnn_dailymail":
        prompt_gen_str = "[full_DEMO]\n\nIn order to make the next LLM generate better highlights, use clear, neutral language to write a concise summary of the following news story. Summarize the news below:\n[original_prompt]\n"
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The user gives a few examples of rephrasing and a sentence that needs to be rephrased. The assistant provides a rephrased sentence without additional content for user."
    prompt_gen_template = template.InitQATemplate(prompt_gen_str)
    # device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    device = f"cuda:{args.cuda}"
    if os.path.exists(args.output_dir + '/' + "adapter_config.json"):
        lora_resume = True
    else:
        lora_resume = False
    blackbox_mode = getattr(args, "blackbox_mode", "image")
    whitebox = WhiteBoxModel(model_name=args.white_model, device=device, hf_token=args.hf_token if hasattr(args, 'hf_token') else None, lora_rank=args.lora_rank, blackbox_mode = blackbox_mode) 
    if blackbox_mode == "text":
        blackbox = BlackBoxModel(model_type="text", model_name=args.black_model, device=device, batch_size=batch_size, black_token=args.black_token if hasattr(args, "black_token") else None, api_base=args.api_base if hasattr(args, "api_base") else None)
        evaluator = Evaluator("text", args = args)
    elif blackbox_mode == "image":
        blackbox = BlackBoxModel(model_type="image", model_name=args.black_model, device=device, batch_size=batch_size, black_token=args.black_token if hasattr(args, "black_token") else None, api_base=args.api_base if hasattr(args, "api_base") else None)
        evaluator = Evaluator("image", cache_dir="./cache/", device=device, args = args)
    else:
        raise ValueError("Invalid blackbox_mode")

    optimizer = MomentumOptimizer(whitebox, blackbox, evaluator, args)


    # initial_model_state = whitebox.model.state_dict()


    async def evaluate_model(dataloader, evaluator, epoch, soft_prompt_embeds = None):
        scores = []
        if args.blackbox_mode == "image":
            sub_scores = {'aesthetic': [], 'clip': [], 'pick': []}
        elif args.blackbox_mode == "text":
            sub_scores = {'bert_score': [], 'ppl_score': [], 'align_score': []}
        
        #for batch in dataloader:
        for batch_idx, batch in enumerate(dataloader):
            original_prompts = batch['text']
            # original_prompts = batch['reference']

            if epoch == "ori":
                generated_prompts = original_prompts
            else:
                if args.white_model in ["promtist", "sft"]:
                    input_text = original_prompts
                else:
                    text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                    input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
                generated_prompts = whitebox.generate(input_text, soft_prompt_embeds=soft_prompt_embeds)
                if args.dataset == "cnn_dailymail":
                    black_prompts = "You are a professional text summarization bot. Condense the user-provided text into a 3-5 sentence summary using clear and concise language, preserving core facts and key information. Requirements: 1.Base strictly on original content without subjective interpretation. 2.Focus on main ideas, omit minor details. 3.The assistant provides sentence without additional content for user. Here is the article:\n"
                    generated_prompts = [black_prompts + p for p in generated_prompts]

            results = await blackbox.generate(generated_prompts)

            if args.blackbox_mode == "image":
                reference = generated_prompts
                
            elif args.blackbox_mode == "text":
                reference = batch['reference']
                
            if args.gene_image == 'True'and epoch != -2:
                save_batch_results(
                    results,
                    reference,
                    args.output_dir,
                    epoch= epoch,
                    method="ori" if epoch == -1 else "ours",
                    batch_idx=batch_idx
                )
            
            batch_scores = []
            for ori, ref, res in zip(original_prompts, reference, results):
                if args.blackbox_mode == "image":
                    # 评估图像生成结果
                    score_dict = evaluator.evaluate(ref, res)
                elif args.blackbox_mode == "text":
                    # 评估文本生成结果
                    score_dict = evaluator.evaluate(ori, ref, res)
                batch_scores.append(score_dict[args.metric])
                for k in sub_scores:
                    sub_scores[k].append(score_dict[k])
            
            scores.extend(batch_scores)
            #orig_test_scores，subscore 取平均时只计算大于零的指标
            scores = [score for score in np.array(scores) if score > 0]
            sub_scores = {k: [v for v in values if v > 0] for k, values in sub_scores.items()}
        return scores, sub_scores
    
    
    print("--------------------------original prompt test---------------------------")
    #     optimizer.analyze_gradients(args.output_dir / f"{epoch}")
    test_scores, subscore = await evaluate_model(test_dataloader, evaluator, epoch = "ori")
    mean_subscore = {key: np.mean(values) for key, values in subscore.items()}
    print(f"subscore{mean_subscore}")
    logger.log_training_step({
        'epoch': "original",
        'test_avg_score': np.mean(test_scores),
        'mean_subscore': mean_subscore,
    })
    print("--------------------------original prompt finish---------------------------")

    
    print("---------------------------ablation test------------------------------")
    orig_test_scores, orig_subscore = await evaluate_model(test_dataloader, evaluator, epoch="ablation")
    mean_subscore = {key: np.mean(values) for key, values in orig_subscore.items()}
    # for name, param in whitebox.model.named_parameters():
    #     if 'lora_A' in name or 'lora_B' in name:
    #         print(f"Parameter {name} set_flat_params: {param.data}")
    print(f"subscore{mean_subscore}")
    logger.log_training_step({
        'epoch': "ablation",
        'orig_score': np.mean(orig_test_scores),
        'orig_subscore': mean_subscore,
    })
    print("---------------------------ablation test finish------------------------------")
    
    if lora_resume != True: 
        if args.batch_mode == "batch":

            print("------------------------批次模式训练-----------------------") 
            for epoch in range(args.epochs):
                epoch_start = time.time()
                
                for batch_idx, batch in enumerate(train_dataloader):
                    original_prompts = batch['text']
                    # original_prompts = batch['reference']
                    print(f"batch_idx: {batch_idx}")
                    # 使用白盒模型生成优化prompt
                    if args.white_model in ["promtist", "sft"]:
                        input_text = original_prompts
                    else:
                        text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                        input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
                    if args.optimizer == "mmt":
                        if 'reference' in batch:
                            grad = await optimizer.estimate_gradient(input_text, batch['reference'])
                        else:
                            grad = await optimizer.estimate_gradient(input_text, None)
                    if args.optimizer == "spsa":
                        if 'reference' in batch:
                            grad = await optimizer.spsa_gradient(input_text, batch['reference'])
                        else:
                            grad = await optimizer.spsa_gradient(input_text, None)
                    optimizer.update_params(grad, lr=args.lr)

                # optimizer.analyze_gradients(args.output_dir / f"{epoch}")
                
                for name, param in whitebox.model.named_parameters():
                    if 'lora_A' in name or 'lora_B' in name:
                        print(f"epoch: {epoch} , Parameter {name} set_flat_params: {param.data}")
                # 每个epoch的测试
                test_scores, subscore = await evaluate_model(test_dataloader, evaluator, epoch)
                if subscore is not None:
                    mean_subscore = {key: np.mean(values) for key, values in subscore.items()}
                    print(f"subscore: {mean_subscore}")
                    logger.log_training_step({
                        'epoch': epoch,
                        'test_avg_score': np.mean(test_scores),
                        'mean_subscore': mean_subscore,
                        'epoch_time': time.time() - epoch_start
                    })
                else:
                    print("No subscores available for text evaluation")
                    logger.log_training_step({
                        'epoch': epoch,
                        'test_avg_score': np.mean(test_scores),
                        'epoch_time': time.time() - epoch_start
                    })
        else:
            print("------------------------采样模式训练-----------------------") 
            for batch_idx, batch in enumerate(train_dataloader):
                original_prompts = batch['text']
                # original_prompts = batch['reference']
                # 使用白盒模型生成优化prompt
                if args.white_model in ["promtist", "sft"]:
                    input_text = original_prompts
                else:
                    text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                    input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]

                for epoch in range(args.epochs):
                    print(f"Epoch {epoch+1}/{args.epochs}")
                    start_time = time.time()
                    # 生成优化prompt
                    if args.optimizer == "mmt":
                        grad = await optimizer.estimate_gradient(input_text)
                    if args.optimizer == "spsa":
                        grad = await optimizer.spsa_gradient(input_text)
                    optimizer.update_params(grad, lr=args.lr)

                    #单个样本更新后测试一下
                    if epoch % 5 == 0:
                        generated_prompts = whitebox.generate(input_text)
                        results = await blackbox.generate(generated_prompts)
                        score_dict = evaluator.evaluate(generated_prompts[0], results[0])
                        logger.log_training_step({
                            'text': original_prompts,
                            'local_epoch': epoch,
                            'current_score': score_dict,
                            'time_spent': time.time() - start_time
                        })
                        
        whitebox.model.save_pretrained(args.output_dir)
        print(f"Saved LoRA parameters to {args.output_dir}")
        logger.finalize()
        for name, param in whitebox.model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                print(f"Parameter {name} set_flat_params: {param.data}")
        print("训练完成")
    if args.soft_train == 'True':
        print("-------------------------开始软提示优化-------------------------")
        
        # Make sure to reload the model FIRST if it's a resume scenario
        if lora_resume:
            print(f"Loading existing LoRA from {args.output_dir}")
            whitebox.load_lora(args.output_dir)
            whitebox.merge_lora()

            # 推理前必须设置eval模式
            whitebox.model.eval()
        
        soft_prompt_optimizer = SoftPromptOptimizer(whitebox, blackbox, evaluator, args)
        
            
        # 检查是否用批处理模式进行软提示优化
        if args.batch_mode == "batch":
            print("------------------------批次模式软提示训练-----------------------")
            for epoch in range(args.soft_epochs):
                epoch_start = time.time()
                epoch_scores = []
                
                for batch_idx, batch in enumerate(train_dataloader):
                    original_prompts = batch['text']
                    print(f"Soft batch_idx: {batch_idx}")
                    
                    # 使用白盒模型生成优化prompt
                    if args.white_model in ["promtist", "sft"]:
                        input_text = original_prompts
                    else:
                        text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                        input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
                    
                    # 执行一次软提示优化步骤
                    result = await soft_prompt_optimizer.optimize_step(
                        input_text, 
                        batch_idx=batch_idx, 
                        epoch=epoch,
                        output_dir=args.soft_output_dir,
                        reference=batch['reference'] if 'reference' in batch else None
                    )
                    
                
                # 在每个 epoch 结束后进行评估
                print(f"Soft Prompt Epoch {epoch} 评估...")
                current_embeds = soft_prompt_optimizer.get_soft_prompt_embeds()
                eval_scores, eval_subscore = await evaluate_model(
                    test_dataloader, 
                    evaluator, 
                    epoch = epoch,
                    soft_prompt_embeds=current_embeds
                )
                
                if isinstance(eval_subscore, dict):
                    eval_mean_subscore = {key: np.mean(values) for key, values in eval_subscore.items()}
                    print(f"Epoch {epoch} 子分数: {eval_mean_subscore}")
                
                eval_mean_score = np.mean(eval_scores)
                print(f"Epoch {epoch} 平均分数: {eval_mean_score:.4f}")
                
                # 记录 epoch 评估结果
                logger.log_training_step({
                    'epoch': f"soft_eval_{epoch}",
                    'eval_score': eval_mean_score,
                    'eval_subscore': eval_mean_subscore if isinstance(eval_subscore, dict) else None,
                    'epoch_time': time.time() - epoch_start,
                    'epoch_avg_batch_score': np.mean(epoch_scores)
                })
                
            # 保存最终的软提示
            soft_prompt_dir = os.path.join(args.soft_output_dir, "soft_prompt")
            soft_prompt_optimizer.save(soft_prompt_dir)
            
        else:
            print("------------------------整体样本软提示训练-----------------------")
            
            # 选择指定批次数量的样本
            all_prompts = []
            batches_collected = 0
            
            # 获取指定批次数量的样本
            for batch in train_dataloader:
                batch_prompts = batch['text']
                all_prompts.extend(batch_prompts)
                batches_collected += 1
                if batches_collected >= args.soft_train_batches:
                    break
            
            if args.white_model in ["promtist", "sft"]:
                input_text = all_prompts
            else:
                text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in all_prompts]
                input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
            
            print(f"使用 {len(all_prompts)} 个样本进行软提示优化...")
            
            # 执行软提示优化
            optimization_result = await soft_prompt_optimizer.optimize(
                input_text, 
                args.soft_epochs,
                output_dir=args.soft_output_dir,
                reference=batch['reference'] if 'reference' in batch else None
            )
            
            # 保存优化后的软提示
            soft_prompt_dir = os.path.join(args.soft_output_dir, "soft_prompt")
            soft_prompt_optimizer.save(soft_prompt_dir)



if __name__ == "__main__":
    args = parse_args()
    set_all_seed(args.seed)
    asyncio.run(main(args))