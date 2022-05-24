
import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--allset_name', type=str, default='COVID-19 Radiography Database', help='name of the original dataset')
    parser.add_argument('--NumClass', type=int, default=3, help='number of category')
    parser.add_argument('--NumClients', type=int, default=20, help='number of total clients')
    parser.add_argument('--ImgSize', type=tuple, default=(64, 64), help='reshape image size')
    parser.add_argument('--NUM_ROUNDS', type=int, default=200, help='number of rounds of training globally')
    parser.add_argument('--Epochs', type=int, default=4, help='client training epochs locally')
    parser.add_argument('--BatchSize', type=int, default=16, help='number of images per batch')
    parser.add_argument('--TrainSplit', type=float, default=0.7, help='proportion of  training set images to all images')




    args = parser.parse_args()

    return args

