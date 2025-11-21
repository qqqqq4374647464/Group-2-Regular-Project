import torch
from torch.utils.data import Dataset
from data_processor import MRPCDataProcessor

class MRPCDataset(Dataset):
    def __init__(self, data_dir='data', split='train'):
        super(MRPCDataset, self).__init__()
        self.split = split
        
        # 加载数据
        processor = MRPCDataProcessor(data_dir)
        self.samples = processor.load_dataset(split)
        
        print(f"{split}数据集: 共{len(self.samples)}个样本")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        sentence1 = sample['sentence1']
        sentence2 = sample['sentence2']
        label = sample['label']
        return sentence1, sentence2, torch.tensor(label, dtype=torch.float)