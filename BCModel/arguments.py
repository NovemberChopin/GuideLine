import argparse


def get_common_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--batch_size', type=int, default=12, help='the size of batch')
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate for adam optimizer")
    parser.add_argument('--episodes_num', type=int, default=3000, help='the number of episodes')
    parser.add_argument('--save_interval', type=int, default=100, help='the number of steps to save the model')
    parser.add_argument('--save_dir', type=str, default="./model")
    parser.add_argument('--log_dir', type=str, default="./log")
    parser.add_argument('--fea_dir', type=str, default="./data_input/features_aug.npy")
    parser.add_argument('--lab_dir', type=str, default="./data_input/labels_aug.npy")

    parser.add_argument('--max_steps', type=int, default=1000, help='the max number of step for a episode')
    parser.add_argument("--num_units_1", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--num_units_2", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--input_size", type=int, default=23, help="input size")
    parser.add_argument("--output_size", type=int, default=18, help="output size: B-spline contral points")
    parser.add_argument("--device", type=str, default="cpu", help="use CPU or GPU")
    parser.add_argument("--eva_period", type=int, default=5, help="the evaluate time")
 

    args = parser.parse_args()
    return args
