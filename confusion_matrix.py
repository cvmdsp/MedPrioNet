import os
import numpy as np
import warnings
import logging

import torch
import torch.utils.data as Data
from sklearn.metrics import roc_curve, auc

from utils import same_seed
from config import args
from datagenerator import LesionSliceDataset
from model import generate_model_mbt

from dataset.transforms import *
from dataset.dataset import TSNDataSet
from dataset import dataset_config
from sklearn.metrics import roc_curve, auc, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def test():
    print("test begin")
    same_seed(args.seed)

    # log
    real_epochs = args.epochs * args.data_iter
    pth_name = args.model_name + "_e" + str(real_epochs) + "_bs" + str(args.batch_size) + "_lr" + str(args.lr) + "_" + args.data_type + "_" + args.phase_code
    log_name = args.model_name + "_e" + str(real_epochs) + "_bs" + str(args.batch_size) + "_lr" + str(args.lr) + "_" + args.data_type + "_" + args.phase_code + "_test"
    print("log_name: ", log_name)
    f = os.path.join(args.log_path, log_name + ".txt")
    logging.basicConfig(filename=f, level=logging.INFO, filemode='a',
                        format='[%(asctime)s.%(msecs)03d] %(message)s')

    torch.cuda.set_device(0)

    # get shape
    vol_size = (args.slice_num, args.img_size, args.img_size)  # D, H, W

    # generate data
    print("Generating data...")
    # test_path = args.test_path + '_' + args.data_type
    # class_path = os.path.join(args.lesion_path, "lesion_slice_classes_organized.npy")
    # ds = LesionSliceDataset(test_path, args.base_phase, vol_size, args.num_classes, class_path, transform=False,
    #                         is_test=True, no_phase_data=False, slice_position=True, data_iter=1)
    # dl = Data.DataLoader(dataset=ds, batch_size=1, shuffle=True)

    data_length = 1
    data_mode = args.mode.split('_')[0]
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    # 根据数据类型选择路径
    if 'hcc' in args.mode:
        args.train_list = args.train_list.format('hcc_icc')
        args.val_list = args.val_list.format('hcc_icc')
    else:
        args.train_list = args.train_list.format('benign_ma')
        args.val_list = args.val_list.format('benign_ma')

    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    normalize = GroupNormalize(input_mean, input_std)
    test_ds = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(224 * 256 // 224)),
                       GroupCenterCrop(224),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, data_mode=data_mode)

    test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=1, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    # set model
    print("Setting model...")
    model_paras = args.model_name.split('_')
    model = generate_model_mbt(model_type=model_paras[0], model_scale=model_paras[1], phase_num=int(model_paras[2]),
                               bottleneck_n=int(model_paras[3]), backbone=model_paras[4], gpu_id=[0],
                               pretrain_path=None, nb_class=args.num_classes, is_multi=False,
                               in_channel=args.in_channel)

    # choose the best model to load
    # model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, pth_name, "best_epoch.pth")))
    # model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, pth_name, "best_epoch_patient.pth")))
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, pth_name, "last_epoch_117.pth")))

    print("Testing...")
    # 设置类别名
    class_names = ['Ma', 'Be']  # Ma = Malignant, Be = Benign

    # 创建保存目录
    save_dir = r'D:\论文论文\文献阅读\尝试\对比算法\私有数据集混淆矩阵'
    os.makedirs(save_dir, exist_ok=True)

    test_correct_num = 0
    test_correct_num_class = [0] * args.num_classes
    test_num_class = [0] * args.num_classes
    pred_ls = []
    label_ls = []
    model.eval()
    for data in test_loader:
        lesions, label, patient = data
        lesions = lesions.view(1, 3, 8, 224, 224)
        lesions = lesions.permute(0, 2, 1, 3, 4)
        lesions = lesions.cuda().float()
        label = label.cuda()
        # 模型预测
        pred = model(lesions)
        pred_class = torch.max(pred, dim=1)[1]

        # 更新正确预测数量
        test_correct_num += torch.eq(pred_class, label).sum().item()
        test_correct_num_class[label.item()] += (pred_class == label).sum().item()
        test_num_class[label.item()] += 1

        # 保存预测结果
        pred_np = pred.cpu().detach().numpy()
        label_np = np.zeros_like(pred_np)
        label_np[np.arange(label_np.shape[0]), label.cpu().numpy()] = 1
        pred_ls.append(pred_np)
        label_ls.append(label_np)

        # 计算整体准确率
    test_acc = test_correct_num / len(test_ds)

    # 计算每个类别的准确率
    test_acc_class = [0.0] * args.num_classes
    for i in range(args.num_classes):
        if test_num_class[i] > 0:
            test_acc_class[i] = test_correct_num_class[i] / test_num_class[i]

    # ROC AUC 计算
    labels = np.concatenate(label_ls, axis=0)
    scores = np.concatenate(pred_ls, axis=0)
    patient_fpr = [0.0] * args.num_classes
    patient_tpr = [0.0] * args.num_classes
    patient_roc_auc = [0.0] * args.num_classes

    for i in range(args.num_classes):
        if not np.sum(labels[:, i]) == 0:
            patient_fpr[i], patient_tpr[i], _ = roc_curve(labels[:, i], scores[:, i])
            patient_roc_auc[i] = auc(patient_fpr[i], patient_tpr[i])

    # 计算混淆矩阵
    y_true = labels.argmax(axis=1)
    y_pred = scores.argmax(axis=1)

    conf_matrix = confusion_matrix(y_true, y_pred)

    # 绘制标准混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 归一化混淆矩阵（按行百分比）
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_normalized.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        test()
