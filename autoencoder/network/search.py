from argparse import ArgumentParser
import train
import itertools
# simple gridsearch
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/tmp/threed_future.pkl", help="Data root directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size during training")
    parser.add_argument("--in_size", type=int, default=32, help="Size of voxels")
    parser.add_argument("--z_dim", type=int, default=128, help="Size of latent vector z")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs. Use 0 for CPU mode")
    args = vars(parser.parse_args())
    print(args)
    for lr, bs in itertools.product([0.01,0.05,0.1],[1000,500,100]):
        args["lr"]=lr
        args["batch_size"]=bs
        print(lr,bs)
        train.main(args)