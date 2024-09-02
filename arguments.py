import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training Transformer')

    parser.add_argument('--img_width', type=int, default=1280)
    parser.add_argument('--img_height', type=int, default=720)

    # train hyperparameters
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--num_workers', type=int, default=8)

    # augmentation
    parser.add_argument('-p', '--purturb_range', type=float, default=-1)
    parser.add_argument('-s', '--shear_amplitude', type=float, default=0.1)
    parser.add_argument('-i', '--interpolate_ratio', type=float, default=0.2)

    # multi-interval displacement prediction & loss parameters
    parser.add_argument('--intervals', type=int, nargs='+', default=[1,2,4])
    parser.add_argument('--lambda_mag', type=float, default=1.0)
    parser.add_argument('--lambda_global', type=float, default=0.05)

    # model parameters
    parser.add_argument('--num_people', type=int, default=12)
    parser.add_argument('--num_frame', type=int, default=24)
    parser.add_argument('--num_joint', type=int, default=14*12)  #18*12
    parser.add_argument('--input_channel', type=int, default=2)
    parser.add_argument('--dim_emb', type=int, default=32)
    parser.add_argument('--cross_rolls', type=int, nargs='+', default=[1,3])

    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--qkv_bias', type=bool, default=True)
    parser.add_argument('--ff_expand', type=int, default=2.0)

    parser.add_argument('--do_rate', type=float, default=0.1)
    parser.add_argument('--attn_do_rate', type=float, default=0.1)

    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--add_positional_emb', type=int, default=1)

    parser.add_argument('--data_path', type=str, default="./data/preprocessed")
    parser.add_argument('--save_path', type=str, default="experiment")

    args = parser.parse_args()

    return args


def parse_args_actionrecog():
    parser = argparse.ArgumentParser(description='Linear Evaluation Protocol')

    parser.add_argument('--img_width', type=int, default=1280)
    parser.add_argument('--img_height', type=int, default=720)

    # train hyperparameters
    parser.add_argument('--gpus', type=int, default=2) 
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=8)

    # augmentation
    parser.add_argument('-p', '--purturb_range', type=float, default=-1)
    parser.add_argument('-s', '--shear_amplitude', type=float, default=-1)
    parser.add_argument('-i', '--interpolate_ratio', type=float, default=0.1)

    # model parameters
    parser.add_argument('--pretrained_model', type=str, default="pretrained/PT_weight")
    parser.add_argument('--pretrained_model_w_classifier', type=str, default="pretrained/linear/PT_w_classifier")

    parser.add_argument('--num_people', type=int, default=12)
    parser.add_argument('--num_frame', type=int, default=24)
    parser.add_argument('--num_joint', type=int, default=14*12)
    parser.add_argument('--input_channel', type=int, default=2)
    parser.add_argument('--dim_emb', type=int, default=32)
    parser.add_argument('--cross_rolls', type=int, nargs='+', default=[1,3])

    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--qkv_bias', type=bool, default=True)
    parser.add_argument('--ff_expand', type=float, default=2.0)

    parser.add_argument('--do_rate', type=float, default=0.1)
    parser.add_argument('--attn_do_rate', type=float, default=0.1)

    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--add_positional_emb', type=int, default=1)

    parser.add_argument('--num_action_class', type=int, default=8)

    parser.add_argument('--data_path', type=str, default="./data/preprocessed")
    parser.add_argument('--save_path', type=str, default="experiment/linear")

    args = parser.parse_args()

    return args


def parse_args_finetuning():
    parser = argparse.ArgumentParser(description='Training Transformer')

    parser.add_argument('--img_width', type=int, default=1280)
    parser.add_argument('--img_height', type=int, default=720)

    # train hyperparameters
    parser.add_argument('--gpus', type=int, default=2) 
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_workers', type=int, default=8)

    # augmentation
    parser.add_argument('-p', '--purturb_range', type=float, default=0.1)
    parser.add_argument('-s', '--shear_amplitude', type=float, default=0.3)
    parser.add_argument('-i', '--interpolate_ratio', type=float, default=0.1)

    # model parameters
    parser.add_argument('--pretrained_model', type=str, default="pretrained/PT_weight")
    parser.add_argument('--pretrained_model_w_classifier', type=str, default="")

    parser.add_argument('--num_frame', type=int, default=25)
    parser.add_argument('--num_joint', type=int, default=14*12)
    parser.add_argument('--input_channel', type=int, default=2)
    parser.add_argument('--dim_emb', type=int, default=32)

    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--qkv_bias', type=bool, default=True)
    parser.add_argument('--ff_expand', type=float, default=2.0)

    parser.add_argument('--do_rate', type=float, default=0.1)
    parser.add_argument('--attn_do_rate', type=float, default=0.0)

    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--add_positional_emb', type=int, default=1)

    parser.add_argument('--num_action_class', type=int, default=8)

    parser.add_argument('--data_path', type=str, default="./data/preprocessed/normal")
    parser.add_argument('--save_path', type=str, default="experiment/finetuning")

    args = parser.parse_args()

    return args
