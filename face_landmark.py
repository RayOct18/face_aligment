import pandas as pd
from PIL import Image
import os
import numpy as np

class FaceLandmarksSet(object):
    """Face Landmarks dataset."""
    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.

        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = self.landmarks_frame.iloc[idx, 1]
        sub_folder = self.landmarks_frame.iloc[idx, 0]
        sub_folder = '' if np.isnan(sub_folder) else sub_folder
        img_root = os.path.join(self.root_dir, sub_folder, img_name)
        image = Image.open(img_root)
        image = image.convert('RGB')
        landmarks = self.landmarks_frame.iloc[idx, 3:].values
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        return sample

    def multi_face_landmark(self, ori_img, ori_sub):
        sub_folder = self.landmarks_frame.iloc[:, 0]
        img_name = self.landmarks_frame.iloc[:, 1]
        img_index = np.where(img_name == ori_img)[0]
        lm_index = []
        for i in img_index:
            sub = '' if np.isnan(sub_folder[i]) else sub_folder[i]
            if sub == ori_sub:
                lm_index.append(i)
        return lm_index
