import argparse
import random
import numpy as np
import torch
def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set all the seeds to {seed} successfully!")

def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--batch_mode",type=str,default='batch')
    parser.add_argument("--white_model",type=str,default='vicuna-13b',help="The model name of the open-source LLM.")
    parser.add_argument("--black_model",type=str,default='sd1.5', help='The model name of the close-source LLM.')
    parser.add_argument("--api_base",type=str,default=None, help='API base URL for OpenAI-compatible APIs')
    parser.add_argument("--blackbox_mode",type=str,default='image', help='text or image')
    # parser.add_argument("--input_file", type=str, default="./dataset/DiffusionDB_prompts_256_unique.tsv")
    parser.add_argument("--dataset", type=str, default="paintings",help="")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="spsa", help = "mmt or spsa")
    parser.add_argument("--metric", type=str, default="total", help = "aesthetic, clip, pick, total")
    parser.add_argument('--ptype', type=int,default=5)
    parser.add_argument('--example', type=str,default='promptist', help="promtist or beautiful")
    parser.add_argument("--lambda_1", type=float, default=0.33)
    parser.add_argument("--lambda_2", type=float, default=0.33) 
    parser.add_argument("--lambda_3", type=float, default=0.34)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--soft_output_dir", type=str, default="./output")
    parser.add_argument("--gene_image", type=str, default="False")
    parser.add_argument("--debug", type=str, default="False")
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate of optimizer")
    parser.add_argument("--h", type=float, default=0.05, help="The parameter of disturbance")
    parser.add_argument("--beta", type=float, default=0.5, help="The parameter of SPSA optimizer")
    parser.add_argument("--epochs", type=int, default=20, help="The number of epochs")
    parser.add_argument("--n_directions", type=int, default=20, help="The number of directions")
    parser.add_argument("--soft_n_directions", type=int, default=20, help="The number of directions")
    parser.add_argument("--train_size", type=int, default=256, help="Size of the training dataset")
    parser.add_argument("--test_size", type=int, default=544, help="Size of the test dataset")
    parser.add_argument('--lora_rank', type=int, default=4,help='Rank of LoRA adaptation')
    parser.add_argument('--soft_train_batches', type=int, default=1, help='Number of batches to use for soft prompt optimization')
    parser.add_argument("--soft_train", type=str, default="False",help="是否进行软提示优化")
    parser.add_argument("--soft_lr", type=float, default=0.1, help="软提示优化学习率")
    parser.add_argument("--soft_epochs", type=int, default=50, help="软提示优化轮数")
    parser.add_argument("--mu", type=float, default=0.1,help="ZO-OGD扰动幅度")
    parser.add_argument("--intrinsic_dim", type=int, default=10,help="软提示的内在维度")
    parser.add_argument("--n_prompt_tokens", type=int, default=5,help="软提示的token数量")
    parser.add_argument("--random_proj", type=str, default="uniform",choices=["normal", "uniform"],help="随机投影初始化方式")
    args = parser.parse_args()
    
    if args.batch_size == 1:
        args.batch_mode = "single"
    #黑盒模型
    if args.black_model == "sd1.5":
        args.black_model = "sd-legacy/stable-diffusion-v1-5"
    elif args.black_model == "sd1.4":
        args.black_model = "CompVis/stable-diffusion-v1-4"
    elif args.black_model == "sdXL":
        args.black_model = "stabilityai/stable-diffusion-xl-base-1.0"
    elif args.black_model == "dreamlike":
        args.black_model = "dreamlike-art/dreamlike-photoreal-2.0"
    elif args.black_model == "llama3.1":
        args.blackbox_mode = "text"
        args.black_model = "gpt-3.5-turbo"
    elif args.black_model == "gpt-image-1":
        args.blackbox_mode = "image"
        args.black_model = "gpt-image-1"
        args.black_token = "sk-t9NQplKFStDn3jRv870f8b26702948A2919bCf54E8C8D08e"
        args.api_base = "https://api.vveai.com/v1"
    elif args.black_model == "gpt3.5-turbo":
        args.blackbox_mode = "text"
        args.black_model = "gpt-3.5-turbo"
        args.black_token = "sk-zDDegMmitM5pTu5RF154006d7402487080F0A9D8Fd61D84a"
        args.api_base = "https://api.vveai.com/v1"
    elif args.black_model == "gpt4-turbo":
        args.blackbox_mode = "text"
        args.black_model = "gpt-4-turbo"
        args.black_token = "sk-zDDegMmitM5pTu5RF154006d7402487080F0A9D8Fd61D84a"
        args.api_base = "https://api.vveai.com/v1"
    elif args.black_model == "claude3.5":
        args.blackbox_mode = "text"
        args.black_model = "claude-3-5-sonnet-all"
        args.black_token = "sk-zDDegMmitM5pTu5RF154006d7402487080F0A9D8Fd61D84a"
        args.api_base = "https://api.vveai.com/v1"
    #更换提示词模板
    if args.dataset == "cnn_dailymail":
        args.example = "cnn_dailymail"
    #白盒模型
    if args.white_model == "llama2-7b":
        args.white_model = "meta-llama/Llama-2-7b-hf"
        args.hf_token = "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"
    elif args.white_model == "vicuna-13b":
        args.white_model = "lmsys/vicuna-13b-v1.3"
        args.hf_token = "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"
    elif args.white_model == "vicuna-7b":
        args.white_model = "lmsys/vicuna-7b-v1.5"
        args.hf_token = "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"

    args.device = torch.device("cuda", args.cuda)

    return args