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
# from alignscore import AlignScore
from bert_score import score as bert_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline, DDIMScheduler,DiffusionPipeline
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer, logging
from peft import LoraConfig, get_peft_model, PeftModel
from save_utils import save_batch_results, save_result
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
class SoftPromptOptimizer:
    def __init__(self, whitebox_model, blackbox_model, evaluator, args):
        """
        Initialize soft prompt optimizer
        
        Args:
            whitebox_model: White-box model for generating optimized prompts
            blackbox_model: Black-box model for generating final results
            evaluator: Evaluator for calculating scores of generated results
            args: Parameter configuration
        """
        self.whitebox = whitebox_model
        self.blackbox = blackbox_model
        self.evaluator = evaluator
        self.args = args
        
        
        # Get embedding dimension
        self.embed_dim = self.whitebox.embedding.shape[1]
        
        # Initialize soft prompt embeddings
        self.z_t = torch.zeros(args.intrinsic_dim).to(args.cuda)
        
        
        
        # Initialize soft prompt parameters
        self.mu = getattr(args, "mu", 0.1)  # Perturbation magnitude
        self.lr = getattr(args, "soft_lr", 0.1)  # Learning rate
        self.n_prompt_tokens = getattr(args, "n_prompt_tokens", 5)  # Number of tokens for soft prompts
        self.A = self._initialize_projection_matrix()
        self.n_directions = getattr(args,'soft_n_directions',5)
    
    def _initialize_projection_matrix(self):
        """Initialize projection matrix from low-dimensional to high-dimensional embedding space"""
        A = torch.nn.Linear(
            self.args.intrinsic_dim, 
            self.n_prompt_tokens * self.embed_dim, 
            bias=False
        ).to(self.args.cuda)
        
        # Initialization strategy
        random_proj = getattr(self.args, "random_proj", "uniform")
        
        if random_proj == "normal":
            # Initialize using normal distribution
            mu_hat = self.whitebox.embedding.mean().item()
            std_hat = self.whitebox.embedding.std().item()
            
            # Calculate scaling factor
            alpha = getattr(self.args, "alpha", 1.0)
            sigma = getattr(self.args, "sigma", 1.0)
            
            mu = 0.0
            std = alpha * std_hat / (np.sqrt(self.args.intrinsic_dim) * sigma)
            
            print(f"[Embedding] mu: {mu_hat}, std: {std_hat} [RandProj] mu: {mu}, std: {std}")
            torch.nn.init.normal_(A.weight, mean=mu, std=std)
            
        elif random_proj == "uniform":
            # Initialize using uniform distribution
            torch.nn.init.uniform_(A.weight, -1, 1)
        
        else:
            raise ValueError(f"Unknown random_proj type: {random_proj}")
        
        print(f"A weight mean: {A.weight.mean().item()}, std: {A.weight.std().item()}")
        return A
    
    def get_soft_prompt_embeds(self):
        """Get current soft prompt embeddings"""
        # Map low-dimensional representation to embedding space using projection matrix
        z_projected = self.A(self.z_t.unsqueeze(0))  # [1, n_prompt_tokens * embed_dim]
        
        # Reshape to [1, n_prompt_tokens, embed_dim]
        embeds = z_projected.view(1, self.n_prompt_tokens, self.embed_dim)
        
        # Ensure embedding type matches model type (added this line)
        embeds = embeds.to(self.whitebox.model.dtype)

        # Reshape to [1, n_prompt_tokens, embed_dim]
        return embeds
    
    async def optimize_step(self, prompts, batch_idx=None, epoch=None, output_dir=None, reference=None):
        """Perform one step of soft prompt optimization"""
        z_t_tmp = self.z_t.unsqueeze(0)  # [1, intrinsic_dim]
        total_grad = torch.zeros_like(self.z_t)  # Initialize accumulator
        
        for _ in range(self.n_directions):

            # Generate random noise
            noise = torch.normal(mean=0.0, std=1.0, size=z_t_tmp.shape).to(self.args.device)
            # Generate positive and negative perturbations
            z_t_pos = z_t_tmp + self.mu * noise
            z_t_neg = z_t_tmp - self.mu * noise
            
            # Calculate embeddings for positive and negative perturbations
            Az_pos = self.A(z_t_pos).view(1, self.n_prompt_tokens, self.embed_dim)
            Az_neg = self.A(z_t_neg).view(1, self.n_prompt_tokens, self.embed_dim)

            # Ensure embedding type matches model type
            Az_pos = Az_pos.to(self.whitebox.model.dtype)
            Az_neg = Az_neg.to(self.whitebox.model.dtype)
            # Use white-box model to generate prompts for positive and negative perturbations
            if args.dataset == "cnn_dailymail":
                black_prompts = "You are a professional text summarization bot. Condense the user-provided text into a 3-5 sentence summary using clear and concise language, preserving core facts and key information. Requirements: 1.Base strictly on original content without subjective interpretation. 2.Focus on main ideas, omit minor details. 3.The assistant provides sentence without additional content for user. Here is the article:\n"
                prompts = [black_prompts + p for p in prompts]
            pos_prompts = self.whitebox.generate(prompts, soft_prompt_embeds=Az_pos)
            neg_prompts = self.whitebox.generate(prompts, soft_prompt_embeds=Az_neg)
            
            # Use black-box model to generate final results
            pos_results = await self.blackbox.generate(pos_prompts)
            neg_results = await self.blackbox.generate(neg_prompts)
            
            # Evaluate results
            if self.evaluator.eval_type == "image":
                pos_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(pos_prompts, pos_results)]
                neg_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(neg_prompts, neg_results)]
            else:
                pos_scores = [self.evaluator.evaluate(o, t, r)[self.args.metric] for o, t, r in zip(prompts, reference, pos_results)]
                neg_scores = [self.evaluator.evaluate(o, t, r)[self.args.metric] for o, t, r in zip(prompts, reference, neg_results)]
            # # Calculate average score
            # pos_score_avg = np.mean(pos_scores)
            # neg_score_avg = np.mean(neg_scores)

        
            # print(f"{_}: Pos Score: {pos_score_avg:.4f}, Neg Score: {neg_score_avg:.4f}")
            
            # # Calculate gradient estimate
            # score_diff = pos_score_avg - neg_score_avg
            #g_t_hat = ((score_diff / (2 * self.mu)) * noise).squeeze(0)
            # Update soft prompt parameters
            #self.z_t = self.z_t + self.lr * g_t_hat
            score_diff = np.mean(pos_scores) - np.mean(neg_scores)
            
            # Accumulate gradient estimate
            total_grad += (score_diff / (2 * self.mu)) * noise.squeeze(0)
        
        # Average gradient
        avg_grad = total_grad / self.n_directions
        self.z_t += self.lr * avg_grad  # Update parameters
        
        # Generate results using current soft prompts
        current_soft_prompt = self.get_soft_prompt_embeds()
        current_prompts = self.whitebox.generate(prompts, soft_prompt_embeds=current_soft_prompt)
        current_results = await self.blackbox.generate(current_prompts)
        
        # Save results for each batch (if batch info and output directory are provided)
        if output_dir is not None and batch_idx is not None and epoch is not None and self.args.gene_image == 'True':
            save_batch_results(
                current_results,
                current_prompts,
                output_dir,
                epoch=epoch,
                method="soft_prompt",
                batch_idx=batch_idx
            )
        
        # Return current score
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
        """Perform multiple rounds of optimization"""
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
            
            # Save best results
            if result["score"] > best_score:
                best_score = result["score"]
                best_z_t = self.z_t.cpu().detach().clone()
            
            print(f"Epoch {epoch+1}, Score: {result['score']:.4f}, Best Score: {best_score:.4f}")
        
        # Restore best results
        if best_z_t is not None:
            self.z_t = best_z_t.to(self.args.device)
        
        return {
            "best_score": best_score,
            "best_soft_prompt": self.z_t,
            "scores_history": scores_history,
            "final_soft_prompt_embeds": self.get_soft_prompt_embeds()
        }
    
    def save(self, path):
        """Save optimized soft prompts and projection matrix"""
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
        """Load saved soft prompts and projection matrix"""
        checkpoint = torch.load(os.path.join(path, "soft_prompt.pt"), map_location=whitebox_model.device)
        
        # Update parameters
        args.n_prompt_tokens = checkpoint["n_prompt_tokens"]
        args.intrinsic_dim = checkpoint["intrinsic_dim"]
        
        # Create optimizer instance
        optimizer = cls(whitebox_model, blackbox_model, evaluator, args)
        
        # Load parameters
        optimizer.z_t = checkpoint["z_t"].to(whitebox_model.device)
        optimizer.A.load_state_dict(checkpoint["A_state_dict"])
        
        return optimizer

class TextEvaluator:
    def __init__(self, args):
        """Text-to-text evaluator, currently using F1-score"""
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
            print("Warning: Input text is empty, cannot calculate PPL")
            return 0.0
        ids = self.tok(result, return_tensors="pt", truncation=True, max_length=512).to(self.cuda) 
        with torch.no_grad():
            loss = self.lm(**ids, labels=ids["input_ids"]).loss
        print(f"Calculate PPL, input text length: {len(result)}, corresponding score: {loss.item()}")
        return torch.exp(loss).item()

    def _align_score(self, source: str, result: str) -> float:
        print(f"source:{type(source)}, result:{type(result)}")
        print(f"source-----------------------------------:{len(source)}, result:{len(result)}")
        # score = self.align_scorer.score([{"context": source, "claim": result}])[0]
        score = self.scorer.score(contexts=[source], claims=[result])[0]
        print(f"score:{score}, score_type:{type(score)}")
        return score
    
    def evaluate(self, ori, reference, result):
        """Calculate evaluation metrics"""
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
        # Return detailed scores
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
    def __init__(self, cache_dir, device, args):
        self.device = device
        # Initialize CLIP model
        clip_cache_dir = os.path.join(cache_dir, "clip_model")
        # print(f"clip_cache_dir:{clip_cache_dir}")
        os.makedirs(clip_cache_dir, exist_ok=True)
        model_path = os.path.join(clip_cache_dir, "ViT-L-14.pt")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device, download_root=clip_cache_dir)
        
        # Initialize Aesthetic model
        self.aes_model = AestheticMlp(768)
        state_dict = torch.load("./cache/aesthetic/sac+logos+ava1-l14-linearMSE.pth", map_location=device)
        self.aes_model.load_state_dict(state_dict)
        self.aes_model.to(device)
        self.aes_model.eval()

        # Initialize PickScore model
        pickscore_cache_dir = os.path.join(cache_dir, "pickscore_model")
        os.makedirs(pickscore_cache_dir, exist_ok=True)
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path_pick = "yuvalkirstain/PickScore_v1"
        self.pickscore_processor = AutoProcessor.from_pretrained(processor_path, cache_dir=pickscore_cache_dir, device=device)
        self.pickscore_model = AutoModel.from_pretrained(model_path_pick, cache_dir=pickscore_cache_dir).eval().to(device)

        # Metric weights
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
        # Return detailed scores
        return {
            'total': final_score,
            'aesthetic': aes_score,
            'clip': clip_score,
            'pick': pick_score
        }

class Evaluator:
    """
      - eval_type=="text": Use TextEvaluator
      - eval_type=="image": Use ImageEvaluator
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
        """Load trained LoRA adapter"""
        # Check adapter file
        required_files = ["adapter_model.safetensors", "adapter_config.json"]
        for f in required_files:
            if not os.path.exists(os.path.join(lora_dir, f)):
                print(f"os.path.exists(os.path.join(lora_dir, f)):{os.path.join(lora_dir, f)}")
                raise FileNotFoundError(f"Missing LoRA file: {f}")
     
        # Load adapter
        self.model.load_adapter(lora_dir, adapter_name="adapter_model")
        self.model.set_adapter("adapter_model")
        print(f"Successfully loaded LoRA adapter: {lora_dir}")

    def merge_lora(self):
        """Merge LoRA parameters into base model"""
        original_weight = self.model.transformer.h[-1].attn.c_attn.weight.clone()
        self.model.merge_and_unload()
        merged_weight = self.model.transformer.h[-1].attn.c_attn.weight
        if torch.allclose(original_weight, merged_weight, atol=1e-5):
            raise RuntimeError("Failed to merge LoRA parameters!")
        print("LoRA parameters merged successfully, model is ready for inference")

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
        print(f"load_promtist model, using gpt2 architecture")
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
        print(f"Model parameters loaded to: {param_device}")
        
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
                        f"Unable to access model {repo_id}"
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
                f"Failed to load Vicuna model: {str(e)}"
            ) from e
    
    def _inject_lora(self, lora_rank: int = 4, custom_targets: list = None, n_last_layers: int = 1):
        base_targets = custom_targets or self.MODEL_TARGET_MAP.get(self.model_type, self.MODEL_TARGET_MAP["default"])
        
        # Filter candidate modules
        candidate_modules = []
        for name, _ in self.model.named_modules():
            if any(target in name for target in base_targets):
                candidate_modules.append(name)
        
        # Filter last n layers
        layer_pattern = self.MODEL_LAYER_PATTERNS[self.model_type]
        layer_re = re.compile(layer_pattern)
        
        layer_info = []
        for name in candidate_modules:
            match = layer_re.search(name)
            if match:
                layer_num = int(match.group(1))
                layer_info.append((layer_num, name))
        
        if not layer_info:
            raise ValueError("No matching target module found")
        
        max_layer = max(layer_num for layer_num, _ in layer_info)
        selected_layers = range(max_layer - n_last_layers + 1, max_layer + 1)
        target_modules = [name for layer_num, name in layer_info if layer_num in selected_layers]
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Initialize parameters
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
        print(f"White-box model input: \n{prompts}")
        
        if self.model_type == "gpt2":
            new_prompts = [p + " Rephrase:" for p in prompts]
            
            input_ids = self.tokenizer(
                new_prompts, 
                return_tensors="pt", 
                padding=True
            ).input_ids.to(self.model.device)
            
            if soft_prompt_embeds is not None:
                # Get input embeddings
                input_embeds = self.model.get_input_embeddings()(input_ids)
                
                # Insert soft prompt embeddings into appropriate position of input embeddings
                batch_size = input_embeds.shape[0]
                if soft_prompt_embeds.shape[0] == 1 and batch_size > 1:
                    soft_prompt_embeds = soft_prompt_embeds.repeat(batch_size, 1, 1)
                
                # Concatenate: [BOS, soft_prompt, input_text]
                input_embeds_with_soft = torch.cat([
                    input_embeds[:, :1, :],  # Keep start token (BOS)
                    soft_prompt_embeds,      # Insert soft prompt
                    input_embeds[:, 1:, :]   # Keep remaining input
                ], dim=1)
                
                # Update attention_mask
                attention_mask = torch.ones(
                    (batch_size, input_embeds_with_soft.shape[1]),
                    device=input_embeds.device
                )
                
                # Generate text using embeddings
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
                # Do not use soft prompt, generate directly using input_ids
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
            
            # Process output
            output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print(f"White-box model raw output: \n{output_texts}")
            
            results = []
            for i, text in enumerate(output_texts):
                # print(f"White-box model raw output {i}: {text}")
                processed = text.replace(new_prompts[i], "").strip()
                results.append(processed)

            print(f"White-box model processed output: \n{results}")
            print(f"White-box generation time: {time.time()-start_time:.2f}s")
            return results 
        
        input_token = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = input_token.input_ids.to(self.model.device)
        attention_mask = input_token.attention_mask.to(self.model.device)

        if soft_prompt_embeds is not None:
            # Get input embeddings
            input_embed = self.embedding[input_ids]
            
            # Ensure soft_prompt_embeds batch_size matches input_embed
            batch_size = input_embed.shape[0]
            if soft_prompt_embeds.shape[0] == 1 and batch_size > 1:
                soft_prompt_embeds = soft_prompt_embeds.repeat(batch_size, 1, 1)
            
            # Concatenate: [BOS, soft_prompt, input_text]
            input_embed_with_soft = torch.cat([
                input_embed[:, :1, :],  
                soft_prompt_embeds,     
                input_embed[:, 1:, :]   
            ], dim=1)
            
            # Update attention_mask
            soft_prompt_length = soft_prompt_embeds.shape[1]
            soft_attention_mask = torch.ones(
                (batch_size, soft_prompt_length),
                device=attention_mask.device
            )
            attention_mask_with_soft = torch.cat([
                attention_mask[:, :1],    # Keep start token mask
                soft_attention_mask,      # Soft prompt mask
                attention_mask[:, 1:]     # Keep remaining input mask
            ], dim=1)
            
            # Generate text using embeddings
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
        print(f"White-box model output: \n{results}")
        print(f"White-box generation time: {time.time()-start_time:.2f}s")
        return results
class BlackBoxModel:
    """
    Black-box model interface, supports three modes:
      - model_type=="text": For example, use gpt2-xl or Claude 4 to generate text
      - model_type=="image": For example, use Stable Diffusion 1.4 to generate images
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
                self.openai_model = model_name  # Use OpenAI-compatible Claude model name
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
        print(f"Black-box model input: {prompts}")
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
                print(f"Black-box generation failed: {str(e)}")
                results.extend([None]*len(batch))
                
        return results
    
    async def _generate_text_batch(self, batch, gen_kwargs):
        if hasattr(self, 'openai_model') and self.openai_model is not None:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base if self.api_base else "https://api.openai.com/v1"  # If api_base is not specified, default to OpenAI
            )
            
            responses = []
            for prompt in batch:
                try:
                    response = client.chat.completions.create(
                        model=self.openai_model,  # For example, "claude-4"
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=gen_kwargs.get("max_new_tokens", 2000),
                        temperature=0.7,
                    )
                    responses.append(response.choices[0].message.content.strip())
                except Exception as e:
                    print(f"API call failed: {str(e)}")
                    responses.append(None)
            
            print(f"Black-box model generation succeeded ({self.openai_model}) output: {responses}")
            return responses
        else:
            # Local model generation
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
        if hasattr(self, 'openai_model') and self.openai_model == "gpt-image-1":
            return await self._generate_openai_image_batch(batch, kwargs)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        generator = torch.Generator(self.device).manual_seed(1234)
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
            return results
        else:
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

class MomentumOptimizer:
    def __init__(self, whitebox: WhiteBoxModel, blackbox: BlackBoxModel, evaluator: Evaluator, args):
        self.whitebox = whitebox
        self.blackbox = blackbox
        self.evaluator = evaluator
        self.h = args.h
        self.n_directions = args.n_directions
        self.beta = args.beta
        self.velocity = torch.zeros_like(self.whitebox.get_flat_params())
        self.grad_history = []      
        self.grad_metadata = []     
        self.n_components = 50      
        self.args = args

    def _store_gradient(self, grad: torch.Tensor):
        self.grad_history.append(grad.cpu().clone())
        
        stats = {
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'norm': torch.norm(grad).item(),
            'min': grad.min().item(),
            'max': grad.max().item()
        }
        self.grad_metadata.append(stats)
        
        if len(self.grad_history) > 100:
            self.grad_history.pop(0)
            self.grad_metadata.pop(0)
    def analyze_gradients(self, output_dir: str = "./grad_analysis"):
        os.makedirs(output_dir, exist_ok=True)
        
        self._plot_basic_stats(os.path.join(output_dir, "basic_stats.png"))
        
        self._analyze_dimensions(os.path.join(output_dir, "dimension_analysis.txt"))
        
        self._correlation_analysis(os.path.join(output_dir, "correlation_analysis.png"))
        
        self._direction_consistency(os.path.join(output_dir, "direction_heatmap.png"))

    def _plot_basic_stats(self, save_path: str):
        
        steps = np.arange(len(self.grad_metadata))
        means = [m['mean'] for m in self.grad_metadata]
        stds = [m['std'] for m in self.grad_metadata]
        norms = [m['norm'] for m in self.grad_metadata]

        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(steps, means, label='Mean')
        plt.title("Gradient Mean")
        plt.xlabel("Step")
        
        plt.subplot(1, 3, 2)
        plt.plot(steps, stds, color='orange', label='Std')
        plt.title("Gradient Std")
        plt.xlabel("Step")
        
        plt.subplot(1, 3, 3)
        plt.plot(steps, norms, color='green', label='Norm')
        plt.title("Gradient Norm")
        plt.xlabel("Step")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _analyze_dimensions(self, save_path: str):
        grad_matrix = torch.stack(self.grad_history).numpy()  # [n_steps, n_dims]
        
        dim_means = np.mean(grad_matrix, axis=0)
        dim_stds = np.std(grad_matrix, axis=0)
        
        top10_indices = np.argsort(np.abs(dim_means))[-10:][::-1]
        

    def _correlation_analysis(self, save_path: str):
        import seaborn as sns
        
        sample_indices = np.random.choice(len(self.grad_history), size=50, replace=False)
        grad_samples = [self.grad_history[i].numpy() for i in sample_indices]
        
        cos_sim = np.zeros((len(grad_samples), len(grad_samples)))
        for i in range(len(grad_samples)):
            for j in range(len(grad_samples)):
                cos_sim[i,j] = np.dot(grad_samples[i], grad_samples[j]) / (
                    np.linalg.norm(grad_samples[i]) * np.linalg.norm(grad_samples[j]))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cos_sim, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Gradient Direction Similarity")
        plt.savefig(save_path)
        plt.close()

    def _direction_consistency(self, save_path: str):
        from sklearn.decomposition import PCA
        
        grad_matrix = torch.stack(self.grad_history).numpy()
        
        pca = PCA(n_components=self.n_components)
        pca.fit(grad_matrix)
        
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
                    
            self.whitebox.set_flat_params(original_params + delta)
            pos_prompts = self.whitebox.generate(original_prompts)
            self.whitebox.set_flat_params(original_params - delta)
            neg_prompts = self.whitebox.generate(original_prompts)
            if args.dataset == "cnn_dailymail":
                black_prompts = "You are a professional text summarization bot. Condense the user-provided text into a 3-5 sentence summary using clear and concise language, preserving core facts and key information. Requirements: 1.Base strictly on original content without subjective interpretation. 2.Focus on main ideas, omit minor details. 3.The assistant provides sentence without additional content for user. Here is the article:\n"
                pos_prompts = [black_prompts + p for p in pos_prompts]
            pos_results = await self.blackbox.generate(pos_prompts)
            if pos_results == None:
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

        self.whitebox.set_flat_params(original_params)
        final_grad = total_grad / self.n_directions
        print(f"final_grad :{final_grad}")
        return final_grad
    async def spsa_gradient(self, original_prompts: list, reference) -> torch.Tensor:
        original_params = self.whitebox.get_flat_params()
        total_grad = torch.zeros_like(original_params)
        param_dim = original_params.shape[0]

        for _ in range(self.n_directions):

            delta = torch.randint(0, 2, (param_dim,), device=original_params.device, dtype=torch.float16) * 2 - 1
            self.whitebox.set_flat_params(original_params + self.h * delta)
            pos_prompts = self.whitebox.generate(original_prompts)
            self.whitebox.set_flat_params(original_params - self.h * delta)
            neg_prompts = self.whitebox.generate(original_prompts)

            if args.dataset == "cnn_dailymail":
                black_prompts = "You are a professional text summarization bot. Condense the user-provided text into a 3-5 sentence summary using clear and concise language, preserving core facts and key information. Requirements: 1.Base strictly on original content without subjective interpretation. 2.Focus on main ideas, omit minor details. 3.The assistant provides sentence without additional content for user. Here is the article:\n"
                pos_prompts = [black_prompts + p for p in pos_prompts]
            pos_results = await self.blackbox.generate(pos_prompts)
            if pos_results == None:
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
                continue
            if self.evaluator.eval_type == "image":
                neg_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(neg_prompts, neg_results)]
            else:
                neg_scores = [self.evaluator.evaluate(o, t, r)[self.args.metric] for o, t, r in zip(original_prompts, reference, neg_results)]
            neg_score_avg = np.mean(neg_scores)
            print(f"neg_score_avg :{neg_score_avg}")
            score_diff = pos_score_avg - neg_score_avg

            print(f"n_grad :{(score_diff / (2 * self.h)) * delta}")
            total_grad += (score_diff / (2 * self.h)) * delta

        self.whitebox.set_flat_params(original_params)
        final_grad = total_grad / self.n_directions
        print(f"final_grad :{final_grad}")
        return final_grad


    def update_params(self, gradient: torch.Tensor, lr: float = 0.01):
        self.velocity = self.beta * self.velocity + (1 - self.beta) * gradient
        new_params = self.whitebox.get_flat_params() + lr * self.velocity
        self.whitebox.set_flat_params(new_params)

async def main(args):
    logger = TrainingLogger(args)


    train_dataset = Dataset(f"./dataset/{args.dataset}_train.csv") 
    test_dataset = Dataset(f"./dataset/{args.dataset}_test.csv") 

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
                    score_dict = evaluator.evaluate(ref, res)
                elif args.blackbox_mode == "text":
                    score_dict = evaluator.evaluate(ori, ref, res)
                batch_scores.append(score_dict[args.metric])
                for k in sub_scores:
                    sub_scores[k].append(score_dict[k])
            
            scores.extend(batch_scores)
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

            for epoch in range(args.epochs):
                epoch_start = time.time()
                
                for batch_idx, batch in enumerate(train_dataloader):
                    original_prompts = batch['text']
                    # original_prompts = batch['reference']
                    print(f"batch_idx: {batch_idx}")
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
            for batch_idx, batch in enumerate(train_dataloader):
                original_prompts = batch['text']
                # original_prompts = batch['reference']
                if args.white_model in ["promtist", "sft"]:
                    input_text = original_prompts
                else:
                    text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                    input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]

                for epoch in range(args.epochs):
                    print(f"Epoch {epoch+1}/{args.epochs}")
                    start_time = time.time()
                    if args.optimizer == "mmt":
                        grad = await optimizer.estimate_gradient(input_text)
                    if args.optimizer == "spsa":
                        grad = await optimizer.spsa_gradient(input_text)
                    optimizer.update_params(grad, lr=args.lr)

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
    if args.soft_train == 'True':
        
        # Make sure to reload the model FIRST if it's a resume scenario
        if lora_resume:
            print(f"Loading existing LoRA from {args.output_dir}")
            whitebox.load_lora(args.output_dir)
            whitebox.merge_lora()

            whitebox.model.eval()
        
        soft_prompt_optimizer = SoftPromptOptimizer(whitebox, blackbox, evaluator, args)
        
            
        if args.batch_mode == "batch":
            for epoch in range(args.soft_epochs):
                epoch_start = time.time()
                epoch_scores = []
                
                for batch_idx, batch in enumerate(train_dataloader):
                    original_prompts = batch['text']
                    print(f"Soft batch_idx: {batch_idx}")
                    
                    if args.white_model in ["promtist", "sft"]:
                        input_text = original_prompts
                    else:
                        text_prompt = [init_prompt[0] + prompt_gen_template.fill(demos, d) for d in original_prompts]
                        input_text = [f"{system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]
                    
                    result = await soft_prompt_optimizer.optimize_step(
                        input_text, 
                        batch_idx=batch_idx, 
                        epoch=epoch,
                        output_dir=args.soft_output_dir,
                        reference=batch['reference'] if 'reference' in batch else None
                    )
                    
                
                current_embeds = soft_prompt_optimizer.get_soft_prompt_embeds()
                eval_scores, eval_subscore = await evaluate_model(
                    test_dataloader, 
                    evaluator, 
                    epoch = epoch,
                    soft_prompt_embeds=current_embeds
                )
                
                if isinstance(eval_subscore, dict):
                    eval_mean_subscore = {key: np.mean(values) for key, values in eval_subscore.items()}
                
                eval_mean_score = np.mean(eval_scores)
                
                logger.log_training_step({
                    'epoch': f"soft_eval_{epoch}",
                    'eval_score': eval_mean_score,
                    'eval_subscore': eval_mean_subscore if isinstance(eval_subscore, dict) else None,
                    'epoch_time': time.time() - epoch_start,
                    'epoch_avg_batch_score': np.mean(epoch_scores)
                })
                
            soft_prompt_dir = os.path.join(args.soft_output_dir, "soft_prompt")
            soft_prompt_optimizer.save(soft_prompt_dir)
            
        else:
            
            all_prompts = []
            batches_collected = 0
            
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
            
            
            optimization_result = await soft_prompt_optimizer.optimize(
                input_text, 
                args.soft_epochs,
                output_dir=args.soft_output_dir,
                reference=batch['reference'] if 'reference' in batch else None
            )
            
            soft_prompt_dir = os.path.join(args.soft_output_dir, "soft_prompt")
            soft_prompt_optimizer.save(soft_prompt_dir)



if __name__ == "__main__":
    args = parse_args()
    set_all_seed(args.seed)
    asyncio.run(main(args))
