import os
import torch
import numpy as np
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
    
    async def optimize_step(self, prompts):
        """执行一步软提示优化"""
        z_t_tmp = self.z_t.unsqueeze(0)  # [1, intrinsic_dim]
        
        # 生成随机噪声
        noise = torch.normal(mean=0.0, std=1.0, size=z_t_tmp.shape).to(self.args.device)
        
        # 产生正负扰动
        z_t_pos = z_t_tmp + self.mu * noise
        z_t_neg = z_t_tmp - self.mu * noise
        
        # 计算正负扰动的嵌入表示
        Az_pos = self.A(z_t_pos).view(1, self.n_prompt_tokens, self.embed_dim)
        Az_neg = self.A(z_t_neg).view(1, self.n_prompt_tokens, self.embed_dim)

        # 确保嵌入类型与模型类型一致 (添加这两行)
        Az_pos = Az_pos.to(self.whitebox.model.dtype)
        Az_neg = Az_neg.to(self.whitebox.model.dtype)

        # 使用正扰动生成提示
        pos_prompts = self.whitebox.generate(prompts, soft_prompt_embeds=Az_pos)
        
        # 使用负扰动生成提示
        neg_prompts = self.whitebox.generate(prompts, soft_prompt_embeds=Az_neg)
        
        # 使用黑盒模型生成最终结果
        pos_results = await self.blackbox.generate(pos_prompts)
        neg_results = await self.blackbox.generate(neg_prompts)
        
        # 评估结果
        if self.evaluator.eval_type == "image":
            pos_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(pos_prompts, pos_results)]
            neg_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(neg_prompts, neg_results)]
        else:
            pos_scores = [self.evaluator.evaluate(r) for r in pos_results]
            neg_scores = [self.evaluator.evaluate(r) for r in neg_results]
        
        # 计算平均分数
        pos_score_avg = np.mean(pos_scores)
        neg_score_avg = np.mean(neg_scores)
        
        print(f"Pos Score: {pos_score_avg:.4f}, Neg Score: {neg_score_avg:.4f}")
        
        # 计算梯度估计
        score_diff = pos_score_avg - neg_score_avg
        g_t_hat = ((score_diff / (2 * self.mu)) * noise).squeeze(0)
        
        # 更新软提示参数
        self.z_t = self.z_t + self.lr * g_t_hat
        
        # 使用当前的软提示生成结果
        current_soft_prompt = self.get_soft_prompt_embeds()
        current_prompts = self.whitebox.generate(prompts, soft_prompt_embeds=current_soft_prompt)
        current_results = await self.blackbox.generate(current_prompts)
        
        # 返回当前分数
        if self.evaluator.eval_type == "image":
            current_scores = [self.evaluator.evaluate(t, r)[self.args.metric] for t, r in zip(current_prompts, current_results)]
        else:
            current_scores = [self.evaluator.evaluate(r) for r in current_results]
        
        current_score_avg = np.mean(current_scores)
        print(f"Current Score: {current_score_avg:.4f}")
        
        return {
            "score": current_score_avg,
            "soft_prompt": self.z_t.cpu().detach().clone(),
            "prompts": current_prompts,
            "results": current_results
        }
    
    async def optimize(self, prompts, epochs):
        """执行多轮优化"""
        best_score = -float('inf')
        best_z_t = None
        scores_history = []
        
        for epoch in range(epochs):
            print(f"Soft Prompt Optimization Epoch {epoch+1}/{epochs}")
            result = await self.optimize_step(prompts)
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