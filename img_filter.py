import os 
import shutil
import util
import numpy as np
import cv2

def black_filter(data_dir, blk, img_size, mode):
    print('Filter ratio of black area higher than {}%\n'.format(blk))
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            fullname = os.path.join(root, filename)
            sub_dir = util.get_subfolder(fullname, data_dir)
            black_dir = os.path.join(os.path.split(data_dir)[0], 
                'crop_images_{}_{}_black{}'.format(mode, img_size, blk),
                sub_dir)
            if not os.path.exists(black_dir):
                os.makedirs(black_dir)
            image = cv2.imread(fullname)
            black_pixels = np.sum(image == 0)
            whole_pixel = image.size
            ratio = (black_pixels / whole_pixel) * 100
            if ratio > blk:
                trg = os.path.join(black_dir, os.path.split(fullname)[-1])
                shutil.move(fullname, trg)
                # print(trg)

def blur_filter(data_dir, blr, img_size, mode):
    print('Filter Laplacian value of blur images lower than {}%\n'.format(blr))
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            fullname = os.path.join(root, filename)
            sub_dir = util.get_subfolder(fullname, data_dir)
            blr_dir = os.path.join(os.path.split(data_dir)[0], 
                'crop_images_{}_{}_blur{}'.format(mode, img_size, blr),
                    sub_dir)
            if not os.path.exists(blr_dir):
                os.makedirs(blr_dir)
            image = cv2.imread(fullname)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            if fm < blr:
                trg = os.path.join(blr_dir, os.path.split(fullname)[-1])
                shutil.move(fullname, trg)
                # print(trg)