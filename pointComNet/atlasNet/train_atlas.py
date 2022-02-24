from pathlib import Path
import sys

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))

from pointComNet.atlasNet.AtlasNet import AtlasNet as Net_model
from pointComNet.pytorch_utils.components.ioUtils import *
import os
import yaml
from pointComNet.train.preprocessing_dataset import load_data
from pointComNet.train.parser_args import parse_args

torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    train_loader, test_loader, val_loader = load_data(args)
    yaml_file = os.path.join(os.path.dirname(__file__), "../config/atlas.yaml")
    with open(yaml_file) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    cuda_index = "cuda"
    device = torch.device(cuda_index if (torch.cuda.is_available()) else "cpu")
    checkout_dir = make_dirs_checkout()
    log_dir = make_dirs_log()
    ae_model = Net_model(parameter=parameters,
                         checkpoint_name=args.checkpoint_name,
                         best_name=args.best_name,
                         logger_file_name=os.path.join(log_dir, "atlas_" + args.logfilename),
                         checkpoint_path=checkout_dir)
    ae_model.toCuda(device=device)
    ae_model.configure_optimizers()

    if args.load_checkPoints:
        print("load_checkpoints: ", os.path.join(checkout_dir, args.checkpoint_name))
        ae_model.load_checkpoint(os.path.join(checkout_dir, args.checkpoint_name))
    print("=====> start to create train model")

    if args.train:
        print("=====> start to train")
        ae_model.train_step(start_epoch=0, n_epochs=args.epochs, train_loader=train_loader,
                            test_loader=val_loader, batch_size=args.batch_size, best_model_name=args.best_name)
        args.checkpoint_name = ae_model.best_name
    if args.evaluate:
        print("=====> start to evaluation")
        ae_model.evaluation_step(test_loader=val_loader, check_point_name=args.checkpoint_name)

    if args.evaluateKitti:
        print("=====> start to kitti evaluation")
        ae_model.evaluation_step_kitti(test_loader=val_loader, check_point_name=args.checkpoint_name)


if __name__ == '__main__':
    args_ = parse_args()
    main(args=args_)
