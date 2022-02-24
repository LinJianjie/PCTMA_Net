import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='infoGan based point completion network  ')
    parser.add_argument('--num_points', type=int, default=2048, help="number of point cloud")
    parser.add_argument('--NN_config', type=str, default="infoGan.yaml", help="yaml file for Neural network")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--num_works', type=int, default=16, metavar='num_workers', help='num of workers')
    parser.add_argument('--shuffle', type=lambda x: not (str(x).lower() == 'false'), default=True, help='Use shuffle')
    parser.add_argument('--drop_last', type=lambda x: not (str(x).lower() == 'false'), default=True,
                        help='Use drop_last')
    parser.add_argument("--lr", type=float, default=1e-3, help="lr")
    parser.add_argument("--cuda", type=str, default="0", help="which cuda used")
    parser.add_argument('--gpu', nargs='+', type=int, help='list of gpu ids to be used')
    parser.add_argument("--epochs", type=int, default=100, help="maximum epochs")
    parser.add_argument("--checkpoint_name", type=str, default="ckpt", help="checkpoint names")
    parser.add_argument('--load_checkPoints', type=lambda x: not (str(x).lower() == 'false'), default=False,
                        help='load_checkpoints')
    parser.add_argument('--best_name', type=str, default=False, help='best name')
    parser.add_argument('--logfilename', type=str, default="log", help="create_a_logfilename")
    parser.add_argument('--train', type=lambda x: not (str(x).lower() == 'false'), default=True,
                        help='train')
    parser.add_argument('--evaluate', type=lambda x: not (str(x).lower() == 'false'), default=False,
                        help='evaluate')
    parser.add_argument('--evaluateKitti', type=lambda x: not (str(x).lower() == 'false'), default=False,
                        help='evaluate')


    return parser.parse_args()
