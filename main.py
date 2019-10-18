import os
import argparse
import lm_process

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    LMP = lm_process.LandmarkProcessing(args.data_dir, args.save_dir)
    # lm_csv = LMP.detector()
    fliter_lm = LMP.filter(args.lm_csv, args.manual)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=R'G:\GoogleDrive\Learning\Coding\python\test\images')
    parser.add_argument('--save_dir', type=str, default=R'G:\GoogleDrive\Learning\Coding\python\test\lm')
    parser.add_argument('--lm_csv', type=str, default=R'G:\GoogleDrive\Learning\Coding\python\test\lm\landmark_20191018-1732.csv')
    parser.add_argument('--manual', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
