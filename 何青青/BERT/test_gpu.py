import torch
import transformers
import datasets

print("=== 环境测试 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"Transformers版本: {transformers.__version__}")
print(f"Datasets版本: {datasets.__version__}")

print(f"\n=== GPU信息 ===")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    
    # 测试GPU计算
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("GPU矩阵乘法测试成功!")
else:
    print("警告: CUDA不可用")

print("\n=== BERT模型测试 ===")
try:
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("BERT tokenizer加载成功!")
except Exception as e:
    print(f"BERT加载失败: {e}")

print("\n环境配置完成!")