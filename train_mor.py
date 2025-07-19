import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
import os
import json

from mor_model import MoRModel, MoRConfig
from scheduler import get_trapezoid_scheduler

class SimpleTextDataset(Dataset):
    """简单的文本数据集"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def create_dummy_dataset(size=1000):
    """创建虚拟数据集用于测试"""
    texts = [
        f"This is a sample text number {i}. It contains various words and demonstrates the training process. " * (i % 3 + 1)
        for i in range(size)
    ]
    return texts

def train_mor(args):
    """训练MoR模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 准备tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    except:
        print("Warning: Using dummy tokenizer")
        # 创建一个简单的虚拟tokenizer用于测试
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = 50000
                self.pad_token_id = 0
                self.eos_token_id = 1
            
            def __call__(self, text, **kwargs):
                # 简单地将文本转换为随机token ids用于演示
                tokens = torch.randint(2, self.vocab_size, (kwargs.get('max_length', 512),))
                masks = torch.ones_like(tokens)
                return {
                    'input_ids': tokens.unsqueeze(0),
                    'attention_mask': masks.unsqueeze(0)
                }
        
        tokenizer = DummyTokenizer()
    
    # 2. 创建数据集
    if args.use_dummy_data:
        print("Using dummy dataset for testing...")
        texts = create_dummy_dataset(args.dataset_size)
        dataset = SimpleTextDataset(texts, tokenizer, args.max_length)
    else:
        # 这里可以添加真实数据集加载
        raise NotImplementedError("Real dataset loading not implemented yet")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 3. 创建模型
    config = MoRConfig(
        model_name=args.model_name,
        router_type=args.router_type,
        num_recursion_steps=args.num_recursions,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_length
    )
    
    model = MoRModel(config).to(device)
    
    # 打印模型信息
    params = model.count_parameters()
    print(f"Model: {args.model_name}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Router type: {args.router_type}")
    print(f"Recursion steps: {args.num_recursions}")
    
    # 4. 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    total_steps = len(dataloader) * args.epochs
    scheduler = get_trapezoid_scheduler(
        optimizer, 
        total_steps, 
        warmup_percentage=0.1, 
        cooldown_percentage=0.2
    )
    
    # 5. 训练循环
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            logits, loss = model(input_ids, labels=input_ids)  # 自回归语言建模
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # 记录
            epoch_loss += loss.item()
            global_step += 1
            
            # 更新进度条
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}',
                'step': global_step
            })
            
            # 定期保存
            if global_step % args.save_steps == 0:
                save_checkpoint(model, optimizer, global_step, args.output_dir)
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    # 6. 保存最终模型
    final_save_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'total_steps': global_step,
    }, final_save_path)
    
    print(f"Training completed! Model saved to {final_save_path}")
    return model

def save_checkpoint(model, optimizer, step, output_dir):
    """保存检查点"""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f'checkpoint-{step}.pt')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description='Train MoR Model')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='mor-135m', 
                       choices=['mor-135m', 'mor-360m', 'mor-730m', 'mor-1.7b'])
    parser.add_argument('--router_type', type=str, default='token_choice',
                       choices=['token_choice', 'expert_choice'])
    parser.add_argument('--num_recursions', type=int, default=3)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=512)
    
    # 数据参数
    parser.add_argument('--use_dummy_data', action='store_true', default=True)
    parser.add_argument('--dataset_size', type=int, default=100)
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./mor_checkpoints')
    parser.add_argument('--save_steps', type=int, default=50)
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 开始训练
    trained_model = train_mor(args)
    
    return trained_model

if __name__ == "__main__":
    main()
