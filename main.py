import lm_detector
import os
import argparse
import landmark_filter

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    lm_csv = lm_detector.landmark_detector(args.data_dir, args.save_dir)
    fliter_lm = landmark_filter.landmark_filter(lm_csv, args.data_dir, args.manual)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=R'G:\Google 雲端硬碟\Learning\Coding\python\test\images')
    parser.add_argument('--save_dir', type=str, default=R'G:\Google 雲端硬碟\Learning\Coding\python\test\lm')
    parser.add_argument('--manual', action='store_true', default=True)
    args = parser.parse_args()
    main(args)
