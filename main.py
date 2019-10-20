import os
import argparse
import lm_process
import img_process

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # LMP = lm_process.LandmarkProcessing(args.data_dir, args.save_dir)

    # if args.lm_csv is None:
    #     lm_csv = LMP.detector()
    #     fliter_lm = LMP.filter(lm_csv, args.manual)
    # else:
    #     fliter_lm = LMP.filter(args.lm_csv, args.manual)

    csv_dir = R'G:\GoogleDrive\Learning\Coding\python\test\lm\combine_landmark.csv'
    IMP = img_process.ImageProcessing(args.data_dir, args.save_dir, csv_dir)
    IMP.crop(128, 0.65)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=R'G:\GoogleDrive\Learning\Coding\python\test\images')
    parser.add_argument('--save_dir', type=str, default=R'G:\GoogleDrive\Learning\Coding\python\test')
    parser.add_argument('--lm_csv', type=str, default=R'G:\GoogleDrive\Learning\Coding\python\test\lm\landmark_20191020-1646.csv')
    parser.add_argument('--manual', action='store_true', default=True)
    args = parser.parse_args()
    main(args)
