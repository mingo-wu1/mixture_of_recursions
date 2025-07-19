import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import os

from mor_model import MoRModel, MoRConfig
from train_mor import SimpleTextDataset, create_dummy_dataset, DummyTokenizer

def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            
            # 前向传播
            logits, loss = model(input_ids, labels=input_ids)
            
            # 计算困惑度
            total_loss += loss.item()
            
            # 计算准确率（下一个token预测）
            predictions = torch.argmax(logits[:, :-1], dim=-1)
            targets = input_ids[:, 1:]
            
            correct = (predictions == targets).sum().item()
            total = targets.numel()
            
            correct_predictions += correct
            total_tokens += total
    
    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    accuracy = correct_predictions / total_tokens
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity.item(),
        'accuracy': accuracy
    }

def load_model(checkpoint_path):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 重建配置
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = MoRConfig(**config_dict)
    else:
        # 默认配置
        config = MoRConfig("mor-135m")
    
    # 创建模型并加载权重
    model = MoRModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config

def main():
    parser = argparse.ArgumentParser(description='Evaluate MoR Model')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dataset_size', type=int, default=200)
    parser.add_argument('--max_length', type=int, default=512)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"Loading model from {args.checkpoint_path}")
    model, config = load_model(args.checkpoint_path)
    model = model.to(device)
    
    # 创建测试数据
    tokenizer = DummyTokenizer()  # 使用相同的tokenizer
    texts = create_dummy_dataset(args.dataset_size)
    dataset = SimpleTextDataset(texts, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    # 评估
    print("Evaluating model...")
    results = evaluate_model(model, dataloader, device)
    
    # 打印结果
    print("\nEvaluation Results:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Token Accuracy: {results['accuracy']:.4f}")
    
    # 保存结果
    output_dir = os.path.dirname(args.checkpoint_path)
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()