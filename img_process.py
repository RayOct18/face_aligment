import os
import pandas as pd
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from face_landmark import FaceLandmarksSet
import numpy as np
import datetime
import math
import csv
import sys
import cv2
from skimage import transform as trans

class ImageProcessing():
    def __init__(self, data_dir, save_dir, csv_dir):
        self.data_dir = data_dir
        self.save_dir = os.path.join(save_dir, 'crop_images')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.csv_dir = csv_dir
        self.face_dataset = FaceLandmarksSet(csv_file=csv_dir, root_dir=data_dir)
        self.pad_size = 100

    def crop(self, img_size, scale):
        new_lms = []
        header = ['sub_folder', 'image_name', 'detect_number']
        for i in range(68):
            header += ['part_{}_x'.format(i), 'part_{}_y'.format(i)]

        for i in range(len(self.face_dataset)):
            sample = self.face_dataset[i]
            lm = sample['landmarks'] + self.pad_size
            image = cv2.copyMakeBorder(sample['image'], self.pad_size, self.pad_size, self.pad_size, self.pad_size,
                                       cv2.BORDER_CONSTANT, value=0)

            # warped, M = self.affine_trans(image, lm, img_size, scale)

            M, pose_index = self.estimate_norm(lm, img_size)
            warped = cv2.warpAffine(image,M, (img_size, img_size))

            cropped_lm = self.cropped_lm(M, lm)
            # cropped_lm[:,1] += 20
            self.show_landmarks(warped, cropped_lm)
            info = [sample['folder'], sample['name'], sample['detect_num']]
            cropped_lm = cropped_lm.reshape(136).tolist()
            info += cropped_lm
            new_lms.append(info)
            save_name = os.path.join(self.save_dir, sample['folder'], sample['name'])
            cv2.imwrite(save_name, warped)

        save_lm = os.path.split(self.save_dir)[0]
        save_csv = os.path.join(save_lm, 'lm', 'cropped_landmark_{}-{}.csv'.format(scale, img_size))
        df = pd.DataFrame(new_lms, columns=header)
        df.to_csv(save_csv, index=False)

    @staticmethod
    def cropped_lm(H, lm):
        cropped_lm = []
        for i in range(len(lm)):
            points = np.append(lm[i], 1)
            points = np.dot(H, points)
            cropped_lm.append(points)
        return np.array(cropped_lm)

    @staticmethod
    def show_landmarks(image, landmarks):
        """Show image with landmarks"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
        plt.imshow(image)
        plt.pause(0.0001)  # pause a bit so that plots are updated
        plt.close()
    
    def estimate_norm(self, lm, img_size):
        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lm, 2, values=np.ones(68), axis=1)
        min_M = []
        min_index = []
        min_error = float('inf') 
        src_map = self.src_map()*img_size/128
        for i in np.arange(src_map.shape[0]):
            tform.estimate(lm, src_map[i])
            M = tform.params[0:2,:]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src_map[i]) ** 2,axis=1)))
            # print(error)
            if error< min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index

    @staticmethod
    def src_map():
        src = np.loadtxt('src_whole.txt', delimiter=',')
        src_list = []
        for i in np.arange(src.shape[0]):
            tmp = src[i,:].reshape(68,2)
            src_list.append(tmp)
        src_map = np.array(src_list)
        return src_map

    def affine_trans(self, image, lm, img_size, scale, landmarkIndices=[39, 42, 57]):
        TPL_MIN, TPL_MAX = np.min(lm, axis=0), np.max(lm, axis=0)
        MINMAX_TEMPLATE = (lm - TPL_MIN) / (TPL_MAX - TPL_MIN)
        npLandmarks = np.float32(lm)
        npLandmarkIndices = np.array(landmarkIndices)
        H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                                   np.float32(img_size * MINMAX_TEMPLATE[npLandmarkIndices]*scale + img_size*(1-scale)/2))
        thumbnail = cv2.warpAffine(image, H, (img_size, img_size))
        # cv2.imshow('Image', thumbnail)
        # cv2.waitKey(0)
        return thumbnail, H