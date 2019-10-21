import os
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from face_landmark import FaceLandmarksSet
import numpy as np
import datetime
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
        self.date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.pad_size = 100

    def crop(self, img_size, mode):
        img_dir = os.path.join(self.save_dir, 'crop_images_{}_{}'.format(mode, img_size))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        new_lms = []
        header = ['sub_folder', 'image_name', 'detect_number']
        for i in range(68):
            header += ['part_{}_x'.format(i), 'part_{}_y'.format(i)]

        for i in range(len(self.face_dataset)):
            sample = self.face_dataset[i]
            lm = sample['landmarks'] + self.pad_size
            image = cv2.copyMakeBorder(sample['image'], self.pad_size, self.pad_size, self.pad_size, self.pad_size,
                                       cv2.BORDER_CONSTANT, value=0)

            five_lm = self.five_point(lm)
            M, pose_index = self.estimate_norm(five_lm, img_size, mode)
            warped = cv2.warpAffine(image,M, (img_size, img_size))

            cropped_lm = self.cropped_lm(M, lm)
            # self.show_landmarks(warped, cropped_lm)
            info = [sample['folder'], sample['name'], sample['detect_num']]
            cropped_lm = cropped_lm.reshape(136).tolist()
            info += cropped_lm
            new_lms.append(info)
            save_name = os.path.join(img_dir, sample['folder'])
            if not os.path.exists(save_name):
                os.makedirs(save_name)
            save_name = os.path.join(save_name, sample['name'])
            cv2.imwrite(save_name, warped)

        save_lm = os.path.join(self.save_dir, 'lm')
        if not os.path.exists(save_lm):
            os.makedirs(save_lm)
        save_csv = os.path.join(save_lm, 'cropped_landmark_{}_{}.csv'.format(mode, img_size))
        df = pd.DataFrame(new_lms, columns=header)
        df.to_csv(save_csv, index=False)
        return img_dir

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
    
    def estimate_norm(self, lm, img_size, mode):
        src1 = np.array([
             [51.642,50.115],
             [57.617,49.990],
             [35.740,69.007],
             [51.157,89.050],
             [57.025,89.702]], dtype=np.float32)
        #<--left 
        src2 = np.array([
            [45.031,50.118],
            [65.568,50.872],
            [39.677,68.111],
            [45.177,86.190],
            [64.246,86.758]], dtype=np.float32)

        #---frontal
        src3 = np.array([
            [39.730,51.138],
            [72.270,51.138],
            [56.000,68.493],
            [42.463,87.010],
            [69.537,87.010]], dtype=np.float32)

        #-->right
        src4 = np.array([
            [46.845,50.872],
            [67.382,50.118],
            [72.737,68.111],
            [48.167,86.758],
            [67.236,86.190]], dtype=np.float32)

        #-->right profile
        src5 = np.array([
            [54.796,49.990],
            [60.771,50.115],
            [76.673,69.007],
            [55.388,89.702],
            [61.257,89.050]], dtype=np.float32)

        src = np.array([src1,src2,src3,src4,src5])
        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lm, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float('inf') 

        src_map = src * (img_size/120) if mode == 'whole' else src * (img_size/112)

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
    def five_point(lm):
        left_eye = np.mean(lm[36:42], axis=0)
        right_eye = np.mean(lm[42:47], axis=0)
        return np.array([left_eye, right_eye, lm[30], lm[48], lm[54]])