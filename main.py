import os
import argparse
import lm_process
import img_process
import img_filter

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.cropped_dir is not None:
        crop_dir = args.cropped_dir
    else:
        LMP = lm_process.LandmarkProcessing(args.data_dir, args.save_dir)
        if args.lm_csv is None and args.lm_filtered is False:
            lm_csv = LMP.detector()
            fliter_lm = LMP.filter(lm_csv, args.manual)
        elif args.lm_csv is not None and args.lm_filtered is False:
            fliter_lm = LMP.filter(args.lm_csv, args.manual)
        elif args.lm_csv is not None and args.lm_filtered is True:
            fliter_lm = args.lm_csv

        IMP = img_process.ImageProcessing(args.data_dir, args.save_dir, fliter_lm)
        crop_dir = IMP.crop(args.img_size, args.crop_mode)
        
    if args.img_filtered:
        img_filter.black_filter(crop_dir, args.black_threshold, args.img_size, args.crop_mode)
        img_filter.blur_filter(crop_dir, args.blur_threshold, args.img_size, args.crop_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help='path of unprocessed images')
    parser.add_argument('--save_dir', type=str, default=None, help='path of save images')
    parser.add_argument('--lm_csv', type=str, default=None, help='path of landmark csv')
    parser.add_argument('--cropped_dir', type=str, default=None, , help='path of cropped images')
    parser.add_argument('--lm_filtered', action='store_true', default=True, help='filter landmark or not')
    parser.add_argument('--img_filtered', action='store_true', default=True, help='filter images or not')
    parser.add_argument('--manual', action='store_true', default=False, help='filter landmark by manual')
    parser.add_argument('--crop_mode', type=str, default='origin', help='whole or origin')
    parser.add_argument('--img_size', type=int, default=128, help='cropping size')
    parser.add_argument('--black_threshold', type=float, default=60.0, help='ratio of black area')
    parser.add_argument('--blur_threshold', type=float, default=10.0, help='ratio of blur')

    args = parser.parse_args()
    main(args)
