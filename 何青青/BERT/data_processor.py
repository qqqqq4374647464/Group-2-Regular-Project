import pandas as pd
import os

class MRPCDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_file = os.path.join(data_dir, 'msr_paraphrase_train.txt')
        self.test_file = os.path.join(data_dir, 'msr_paraphrase_test.txt')
        self.data_file = os.path.join(data_dir, 'msr_paraphrase_data.txt')
        
    def load_sentence_mapping(self):
        """加载句子ID到文本的映射"""
        print("加载句子映射...")
        df_data = pd.read_csv(self.data_file, sep='\t', encoding='utf-8', quoting=3)
        sentence_map = {}
        for _, row in df_data.iterrows():
            sentence_id = row['Sentence ID']
            sentence_text = row['String']
            sentence_map[sentence_id] = sentence_text
        print(f"加载了 {len(sentence_map)} 个句子映射")
        return sentence_map
    
    def load_dataset(self, split='train'):
        """加载训练集或测试集"""
        if split == 'train':
            file_path = self.train_file
        else:
            file_path = self.test_file
            
        print(f"加载{split}数据集: {file_path}")
        
        # 读取数据文件
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8', quoting=3, skiprows=1, 
                        names=['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String'])
        
        # 加载句子映射
        sentence_map = self.load_sentence_mapping()
        
        samples = []
        for _, row in df.iterrows():
            quality = row['Quality']
            id1 = row['#1 ID']
            id2 = row['#2 ID']
            string1 = row['#1 String']
            string2 = row['#2 String']
            
            # 使用文件中的字符串，如果为空则使用映射
            if pd.isna(string1) and id1 in sentence_map:
                string1 = sentence_map[id1]
            if pd.isna(string2) and id2 in sentence_map:
                string2 = sentence_map[id2]
                
            samples.append({
                'sentence1': str(string1),
                'sentence2': str(string2),
                'label': int(quality)
            })
        
        print(f"{split}数据集加载完成，共 {len(samples)} 个样本")
        return samples

# 测试数据加载
if __name__ == "__main__":
    processor = MRPCDataProcessor('data')  # 假设数据在data文件夹
    train_data = processor.load_dataset('train')
    test_data = processor.load_dataset('test')
    
    print("\n训练集前3个样本:")
    for i in range(3):
        print(f"样本 {i+1}: {train_data[i]}")