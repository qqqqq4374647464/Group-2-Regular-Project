import os
import torch
from torch.utils.data import DataLoader
from FCModel import FCModel
from MRPCDataset import MRPCDataset
from transformers import BertTokenizer, BertModel
import time

def main():
    print("=== 基于本地MRPC数据集和本地BERT模型的实验 ===")
    
    # 设置运行设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    # 载入数据
    print("\n开始加载本地MRPC数据集...")
    try:
        mrpcDataset = MRPCDataset(data_dir='data', split='train')
        train_loader = DataLoader(dataset=mrpcDataset, batch_size=8, shuffle=True)
        print(f"数据载入完成，共{len(mrpcDataset)}个训练样本")
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请确保数据文件放在 'data' 文件夹中")
        return

    # 加载本地BERT模型和tokenizer
    print("\n开始加载本地BERT模型...")
    local_model_path = "./local_models/bert-base-uncased"
    
    try:
        # 检查本地模型文件是否存在
        required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt', 'tokenizer_config.json']
        missing_files = []
        
        for file in required_files:
            file_path = os.path.join(local_model_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"缺少必要的模型文件: {missing_files}")
            return
        
        print("找到所有必要的模型文件，开始加载...")
        
        # 加载tokenizer和模型
        tokenizer = BertTokenizer.from_pretrained(local_model_path)
        print("Tokenizer加载成功!")
        
        bert_model = BertModel.from_pretrained(local_model_path)
        bert_model.to(device)
        print("BERT模型加载完成")
        
    except Exception as e:
        print(f"本地BERT模型加载失败: {e}")
        print("请检查模型文件路径和完整性")
        return

    # 创建分类模型
    model = FCModel()
    model = model.to(device)
    print("全连接层模型创建完成")

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 0.001},
        {'params': bert_model.parameters(), 'lr': 2e-5}
    ])
    criterion = torch.nn.BCELoss()

    def binary_accuracy(predict, label):
        rounded_predict = torch.round(predict)
        correct = (rounded_predict == label).float()
        return correct.sum() / len(correct)

    def train():
        model.train()
        bert_model.train()
        
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        
        for batch_idx, (sentence1, sentence2, labels) in enumerate(train_loader):
            # 将句子对组合并tokenize
            texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(sentence1, sentence2)]
            
            # Tokenize
            encoding = tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128,
                add_special_tokens=True
            )
            
            # 移动到设备
            encoding = {k: v.to(device) for k, v in encoding.items()}
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            
            bert_outputs = bert_model(**encoding)
            pooler_output = bert_outputs.pooler_output
            predictions = model(pooler_output).squeeze()
            
            loss = criterion(predictions, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            acc = binary_accuracy(predictions, labels)
            
            total_loss += loss.item() * len(labels)
            total_acc += acc.item() * len(labels)
            total_samples += len(labels)
            
            if batch_idx % 20 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}')
        
        return total_loss / total_samples, total_acc / total_samples

    # 开始训练
    print("\n开始训练...")
    num_epochs = 3
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        start_time = time.time()
        epoch_loss, epoch_acc = train()
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1} 完成:")
        print(f"平均损失: {epoch_loss:.4f}")
        print(f"平均准确率: {epoch_acc:.4f}")
        print(f"耗时: {epoch_time:.2f}秒")
        print("-" * 50)

    print("训练完成！")
    print("实验成功完成！")

if __name__ == "__main__":
    main()