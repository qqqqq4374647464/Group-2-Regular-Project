import os
from transformers import BertTokenizer, BertModel
import torch

def verify_local_model():
    local_model_path = "./local_models/bert-base-uncased"
    
    print("=== 验证本地BERT模型 ===")
    print(f"模型路径: {local_model_path}")
    
    # 检查文件
    required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt', 'tokenizer_config.json']
    print("\n检查模型文件:")
    for file in required_files:
        file_path = os.path.join(local_model_path, file)
        exists = os.path.exists(file_path)
        status = "✓ 存在" if exists else "✗ 缺失"
        print(f"  {file}: {status}")
        
        if not exists:
            print(f"    文件路径: {file_path}")
    
    # 尝试加载模型
    print("\n尝试加载模型...")
    try:
        tokenizer = BertTokenizer.from_pretrained(local_model_path)
        print("✓ Tokenizer加载成功")
        
        model = BertModel.from_pretrained(local_model_path)
        print("✓ BERT模型加载成功")
        
        # 测试模型推理
        test_text = "Hello, this is a test sentence."
        inputs = tokenizer(test_text, return_tensors="pt")
        outputs = model(**inputs)
        
        print("✓ 模型推理测试成功")
        print(f"  输入形状: {inputs['input_ids'].shape}")
        print(f"  输出pooler形状: {outputs.pooler_output.shape}")
        
        # 检查设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ 设备检测: {device}")
        
        if device.type == 'cuda':
            model = model.to(device)
            print("✓ 模型已移动到GPU")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False

if __name__ == "__main__":
    success = verify_local_model()
    if success:
        print("\n 本地模型验证成功！可以运行main.py进行训练。")
    else:
        print("\n 本地模型验证失败，请检查模型文件。")