import os
import numpy as np
import warnings
import logging
import pickle
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

def test():
    print("test begin")
    same_seed(args.seed)

    # log
    real_epochs = args.epochs * args.data_iter
    pth_name = args.model_name + "_e" + str(real_epochs) + "_bs" + str(args.batch_size) + "_lr" + str(args.lr) + "_" + args.data_type + "_" + args.phase_code
    model_name = "Ours"
    log_name = f"roc_data_{model_name}.pkl"
    output_file = os.path.join(args.log_path, log_name)

    torch.cuda.set_device(0)

    # generate data
    print("Generating data...")
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
            roc_data = {
                "fpr": patient_fpr[i],  # 假阳性率
                "tpr": patient_tpr[i],  # 真阳性率
                "auc": patient_roc_auc[i],  # AUC 值
            }

    with open(output_file, "wb") as f:
        pickle.dump(roc_data, f)

    print(f"ROC data saved to {output_file}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        test()
