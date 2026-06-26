import argparse

parser = argparse.ArgumentParser()

# public parameters
parser.add_argument("--cuda-num", type=int, help="card number",
                    dest="cuda_num", default=1)
parser.add_argument("--lesion-path", type=str, help="lesion files path",
                    dest="lesion_path", default="/path/to/lesions")
parser.add_argument("--reg-path", type=str, help="registered files path",
                    dest="reg_path", default="/path/to/registered/data")
parser.add_argument("--log-path", type=str, help="logs folder",
                    dest="log_path", default='./log')
parser.add_argument("--img-size", type=int, help="image size",
                    dest="img_size", default=224)
parser.add_argument("--slice-num", type=int, help="slice number",
                    dest="slice_num", default=32)
parser.add_argument("--checkpoint-path", type=str, help="checkpoint folder",
                    dest="checkpoint_path", default='./checkpoint')
parser.add_argument("--base-phase", type=str, help="results folder",
                    dest="base_phase", default='artery')
parser.add_argument("--model-name", type=str, help="model name",
                    dest="model_name", default='mbt_base_8_8_vit')
parser.add_argument("--seed", type=int, help="random seed",
                    dest="seed", default=0)
parser.add_argument("--transform", action="store_true", help="whether to augment",
                    dest="transform")
parser.add_argument("--num-classes", type=int, help="classes number",
                    dest="num_classes", default=2)
parser.add_argument("--num-workers", type=int, help="dataloader workers number",
                    dest="num_workers", default=4)
parser.add_argument("--data-type", type=str, help="data type",
                    dest="data_type", default='slice')
# dataset parameters
parser.add_argument('--dataset', type=str, default="301data")
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow'], default='RGB')
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
parser.add_argument('--mode', type=str, default='rgb')
# parser.add_argument('--mode', type=str, default='hcc')
parser.add_argument('-b', '--batch_size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--arch', type=str, default="BNInception")

# data process parameters
parser.add_argument("--padding-length", type=int, help="padding length on img size",
                    dest="padding_length", default=10)
parser.add_argument("--lesion-dice-threshold", type=int, help="dice threshold in registration",
                    dest="lesion_dice_threshold", default=30)

# train parameters
parser.add_argument("--train-path", type=str, help="train files path",
                    dest="train_path", default="/path/to/train")
parser.add_argument("--val-path", type=str, help="validate files path",
                    dest="val_path", default="/path/to/val")
parser.add_argument("--pretrain-path", type=str, help="pretrain weight folder",
                    dest="pretrain_path", default='./pre')
parser.add_argument("--pretrain-name", type=str, help="pretrain weight folder",
                    dest="pretrain_name", default='cmt_small.pth')
parser.add_argument("--pretrain", action="store_true", help="whether to pretrain",
                    dest="pretrain")
parser.add_argument("--val-ratio", type=float, help="validate data ratio",
                    dest="val_ratio", default=0.25)
parser.add_argument("--lr", type=float, help="learn rate",
                    dest="lr", default=0.001)
parser.add_argument("--batch-size", type=int, help="batch size",
                    dest="batch_size", default=8)
parser.add_argument("--momentum", type=float, help="momentum",
                    dest="momentum", default=0.9)
parser.add_argument("--weight-decay", type=float, help="weight decay",
                    dest="weight_decay", default=5E-4)
parser.add_argument("--drop-out", type=float, help="drop out rate",
                    dest="drop_out", default=0.0)
parser.add_argument("--epochs", type=int, help="epoch",
                    dest="epochs", default=120)
parser.add_argument("--data-iter", type=int, help="data repeat times",
                    dest="data_iter", default=4)
parser.add_argument("--in-channel", type=int, help="image channel",
                    dest="in_channel", default=3)
parser.add_argument("--phase-code", type=str, help="phase code",
                    dest="phase_code", default="ADPV")

# test parameters
parser.add_argument("--test-path", type=str, help="test files path",
                    dest="test_path", default="/path/to/test")

args = parser.parse_args()
