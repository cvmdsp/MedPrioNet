import pickle
import matplotlib.pyplot as plt

# 模型结果文件路径
# roc_files = [
#     "D:/论文论文/文献阅读/尝试/对比算法/私有数据集roc/roc_data_Resnet50.pkl",
#     "D:/论文论文/文献阅读/尝试/对比算法/私有数据集roc/roc_data_ViT.pkl",
#     "D:/论文论文/文献阅读/尝试/对比算法/私有数据集roc/roc_data_SwinT.pkl",
#     "D:/论文论文/文献阅读/尝试/对比算法/私有数据集roc/roc_data_InceptionNeXt.pkl",
#     "D:/论文论文/文献阅读/尝试/对比算法/私有数据集roc/roc_data_R3D.pkl",
#     "D:/论文论文/文献阅读/尝试/对比算法/私有数据集roc/roc_data_R2+1D.pkl",
#     "D:/论文论文/文献阅读/尝试/对比算法/私有数据集roc/roc_data_TimeSformer.pkl",
#     "D:/论文论文/文献阅读/尝试/对比算法/私有数据集roc/roc_data_Video SwinT.pkl",
#     "D:/论文论文/文献阅读/尝试/对比算法/私有数据集roc/roc_data_HiFuse.pkl",
#     "D:/论文论文/文献阅读/尝试/对比算法/私有数据集roc/roc_data_MedViT.pkl",
#     "D:/论文论文/文献阅读/尝试/对比算法/私有数据集roc/roc_data_Hybrid-Net.pkl",
#     "D:/论文论文/文献阅读/尝试/对比算法/私有数据集roc/roc_data_Ours.pkl"
# ]
roc_files = [
    "D:/论文论文/文献阅读/尝试/对比算法/公共数据集roc/roc_data_Resnet50.pkl",
    "D:/论文论文/文献阅读/尝试/对比算法/公共数据集roc/roc_data_ViT.pkl",
    "D:/论文论文/文献阅读/尝试/对比算法/公共数据集roc/roc_data_SwinT.pkl",
    "D:/论文论文/文献阅读/尝试/对比算法/公共数据集roc/roc_data_InceptionNeXt.pkl",
    "D:/论文论文/文献阅读/尝试/对比算法/公共数据集roc/roc_data_R3D.pkl",
    "D:/论文论文/文献阅读/尝试/对比算法/公共数据集roc/roc_data_R2+1D.pkl",
    "D:/论文论文/文献阅读/尝试/对比算法/公共数据集roc/roc_data_TimeSformer.pkl",
    "D:/论文论文/文献阅读/尝试/对比算法/公共数据集roc/roc_data_Video SwinT.pkl",
    "D:/论文论文/文献阅读/尝试/对比算法/公共数据集roc/roc_data_HiFuse.pkl",
    "D:/论文论文/文献阅读/尝试/对比算法/公共数据集roc/roc_data_MedViT.pkl",
    "D:/论文论文/文献阅读/尝试/对比算法/公共数据集roc/roc_data_Hybrid-Net.pkl",
    "D:/论文论文/文献阅读/尝试/对比算法/公共数据集roc/roc_data_Ours.pkl"
]

# 初始化绘图
plt.figure(figsize=(10, 8))

# 加载每个模型的 ROC 数据并绘制
for file in roc_files:
    with open(file, "rb") as f:
        data = pickle.load(f)

    fpr = data["fpr"]
    tpr = data["tpr"]
    auc_value = data["auc"]

    # 提取模型名称
    model_name = file.split("_")[2].replace(".pkl", "")  # 提取模型名称

    # 判断是否为 "Ours" 模型
    if "Ours" in file:
        # 为 "Ours" 设置特殊样式
        plt.plot(
            fpr, tpr, label=f"Ours (AUC = {auc_value:.4f})", color="red", linewidth=2, linestyle="-"
        )
    else:
        # 其他模型使用默认样式
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.4f})", linewidth=1.5)

# 添加随机猜测参考线
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")

# 设置图形样式
plt.xlabel("False Positive Rate (FPR)", fontsize=14)
plt.ylabel("True Positive Rate (TPR)", fontsize=14)
plt.title("ROC Curve Comparison", fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

# 显示图形
plt.tight_layout()
plt.show()
