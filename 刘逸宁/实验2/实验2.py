import pandas as pd
import matplotlib.pyplot as plt

# 备选：用latin-1编码读取（无解码失败风险）
data = pd.read_csv(
    "Pokemon.csv",
    encoding='latin-1'
)

# 验证读取成功
print("原始数据形状（行数, 列数）：", data.shape)  # 预期接近(721+, 12)（含脏数据）
print("\n前5行数据（确认列名）：")
print(data.head())
print("\n最后10行数据（观察无意义尾行）：")
print(data.tail(10))
print("\nType 2列唯一值（查看异常值'273'）：")
print(data["Type 2"].unique())
print("\n重复值数量：", data.duplicated().sum())  # 查看重复行数量

# 方法1：删除最后2行（通过切片保留前n-2行，n为数据总行数）
data_clean1 = data.iloc[:-2, :]  # iloc[:-2]表示“取除最后2行外的所有行”

# 验证：查看删除后的数据形状和末尾
print("删除无意义尾行后形状：", data_clean1.shape)  # 行数比原始少2
print("删除后末尾5行：")
print(data_clean1.tail())

# 将Type 2列中值为'273'的行替换为NaN（“清空”异常值）
data_clean1.loc[data_clean1["Type 2"] == "273", "Type 2"] = pd.NA

# 验证：查看Type 2列唯一值（已无'273'）
print("Type 2列处理后唯一值：")
print(data_clean1["Type 2"].unique())

# 去除完全重复的行（默认保留第一行）
data_clean2 = data_clean1.drop_duplicates()

# 验证：重复值数量变为0
print("去重后重复值数量：", data_clean2.duplicated().sum())  # 预期为0
print("去重后数据形状：", data_clean2.shape)  # 行数比data_clean1少（具体减少数=原重复值数）

# 步骤2.3：删除重复值
data_clean2 = data_clean1.drop_duplicates().copy()  # 同样用.copy()确保稳定性

# 关键：检查并转换Attack列为数值型（核心解决绘图错误）
print("转换前Attack列数据类型：", data_clean2["Attack"].dtype)  # 若为object，说明是字符串

# 转换为数值型，无法转换的设为NaN（用errors='coerce'处理脏数据）
data_clean2["Attack"] = pd.to_numeric(data_clean2["Attack"], errors='coerce')

# 清除转换后产生的NaN（若有）
data_clean2 = data_clean2.dropna(subset=["Attack"])  # 仅删除Attack列的NaN行

# 步骤2.4：绘制Attack列箱线图（此时数据为数值型，可正常绘图）
plt.figure(figsize=(8, 4))
data_clean2["Attack"].plot(kind="box")
plt.title("Attack属性箱线图（异常值检测）")
plt.show()  # 成功显示箱线图，无报错



# 2. 用“四分位距（IQR）”删除异常值（通用异常值处理方法）
Q1 = data_clean2["Attack"].quantile(0.25)  # 第一四分位数
Q3 = data_clean2["Attack"].quantile(0.75)  # 第三四分位数
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR  # 下界
upper_bound = Q3 + 1.5 * IQR  # 上界

# 保留Attack在[下界, 上界]内的数据（删除异常值）
data_clean3 = data_clean2[(data_clean2["Attack"] >= lower_bound) & (data_clean2["Attack"] <= upper_bound)]

# 验证：异常值已删除，Attack分布更合理
print("处理Attack异常值后数据形状：", data_clean3.shape)
print("Attack列处理后范围：", data_clean3["Attack"].min(), "~", data_clean3["Attack"].max())

# 1. 定位置换行：观察到“Generation列出现布尔值（True/False）、Legendary列出现数字”即为置换行
# （文档中数据集特征：正常Generation为数字1-6，Legendary为True/False）
swap_rows = data_clean3[(data_clean3["Generation"].isin([True, False])) | (data_clean3["Legendary"].apply(lambda x: isinstance(x, (int, float))))]
print("置换属性的行：")
print(swap_rows[["Generation", "Legendary"]])

# 2. 交换置换行的Generation与Legendary值
for idx in swap_rows.index:
    data_clean3.loc[idx, ["Generation", "Legendary"]] = data_clean3.loc[idx, ["Legendary", "Generation"]].values

# 验证：置换行已修正
print("修正后置换行的属性：")
print(data_clean3.loc[swap_rows.index, ["Generation", "Legendary"]])

print("最终清洗后数据形状：", data_clean3.shape)  # 行数比原始数据少（因删除脏数据）
print("\n最终数据基本信息（无缺失/异常）：")
print(data_clean3.info())  # 查看列数据类型、非空值数量
print("\n最终数据描述性统计（Attack等属性无极端值）：")
print(data_clean3.describe())
