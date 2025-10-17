import pandas as pd
import numpy as np

# 关键：添加encoding='gbk'（或encoding='gb2312'）解决编码错误
primitive_data = pd.read_csv(
    "http://storage.amesholland.xyz/data.csv",
    encoding='gbk'  # 重点：指定与数据集匹配的编码
)

# 验证读取成功（同实验指导中的验证逻辑）
print("读取成功！数据形状（行数, 列数）：", primitive_data.shape)  # 预期输出 (1147, 10)
print("前5行数据：")
print(primitive_data.head())

primitive_data_1 = primitive_data.dropna(how='any')

# 验证删除结果
print("删除空行后的数据形状：", primitive_data_1.shape)  # 预期输出 (1118, 10)
print("\n删除空行后末尾5行（无NaN）：")
print(primitive_data_1.tail())

# 1.将删除空行后的数据作为过滤前的数据源
data_before_filter = primitive_data_1

# 2.第一步过滤：traffic不等于0（排除无流量数据）
data_after_filter_1 = data_before_filter.loc[data_before_filter["traffic"] != 0]

# 3.第二步过滤：from_level为“一般节点”（聚焦目标节点类型）
data_after_filter_2 = data_after_filter_1.loc[data_after_filter_1["from_level"] == '一般节点']

# 验证过滤结果
print("过滤后的数据形状：", data_after_filter_2.shape)  # 预期输出 (554, 10)
print("\n过滤后数据的from_level分布（确认只有“一般节点”）：")
print(data_after_filter_2["from_level"].value_counts())

# 1.获取过滤后的数据和列名（后续采样后需保留原列，排除权重列）
data_before_sample = data_after_filter_2
columns = data_before_sample.columns  # 原数据的10列（如from_dev、traffic等）

# 2.添加权重列：to_level为“一般节点”权重1，“网络核心”权重5
weight_sample = data_before_sample.copy()  # 复制数据，避免修改原数据
weight_sample['weight'] = 0  # 初始化权重列

# 循环给每行设置权重
for i in weight_sample.index:
    if weight_sample.at[i, 'to_level'] == '一般节点':
        weight_sample.at[i, 'weight'] = 1
    else:
        weight_sample.at[i, 'weight'] = 5

# 3.按权重采样50个样本，排除权重列
weight_sample_finish = weight_sample.sample(n=50, weights='weight')[columns]

# 验证采样结果（查看to_level分布，网络核心应更多）
print("加权采样后to_level分布：")
print(weight_sample_finish["to_level"].value_counts())
print("\n加权采样结果前5行：")
print(weight_sample_finish.head())

# 直接从过滤后的数据中随机抽50个样本
random_sample_finish = data_before_sample.sample(n=50)

# 验证采样结果（to_level分布接近原数据比例）
print("随机采样后to_level分布：")
print(random_sample_finish["to_level"].value_counts())
print("\n随机采样结果前5行：")
print(random_sample_finish.head())

# 1.按to_level分成两组
ybjd = data_before_sample.loc[data_before_sample['to_level'] == '一般节点']  # 一般节点组
wlhx = data_before_sample.loc[data_before_sample['to_level'] == '网络核心']    # 网络核心组

# 2.分层采样：一般节点抽17个，网络核心抽33个（合计50个）
after_sample = pd.concat([ybjd.sample(17), wlhx.sample(33)])

# 验证采样结果（确认两组数量分别为17和33）
print("分层采样后to_level分布：")
print(after_sample["to_level"].value_counts())
print("\n分层采样结果前5行：")
print(after_sample.head())
