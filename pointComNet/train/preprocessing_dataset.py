from pathlib import Path
import sys

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
from torch.utils.data import DataLoader
from pointComNet.pytorch_utils.components.dataSet import PointCompletionShapeNet


def load_data(args):
    if args.train:
        train_loader = DataLoader(
            PointCompletionShapeNet(num_points=args.num_points,
                                    partition='train'),
            num_workers=args.num_works,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            drop_last=args.drop_last,
            pin_memory=True)

        # test_loader = DataLoader(
        #     PointCompletionShapeNet(num_points=args.num_points,
        #                             partition='test'),
        #     num_workers=args.num_works,
        #     batch_size=args.batch_size,
        #     shuffle=args.shuffle,
        #     drop_last=args.drop_last)
        test_loader = None
    else:
        train_loader = None
        test_loader = None
    if args.evaluate or args.train or args.evaluateKitti:
        val_loader = DataLoader(
            PointCompletionShapeNet(num_points=args.num_points,
                                    partition='val',
                                    use_kitti=args.evaluateKitti),
            num_workers=args.num_works,
            batch_size=1,
            shuffle=False,
            drop_last=args.drop_last)
    else:
        val_loader = None
    return train_loader, test_loader, val_loader
