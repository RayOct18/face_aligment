import face_alignment
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from face_landmark import FaceLandmarksSet
import datetime
import math
import copy


class LandmarkProcessing():
    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.face_dataset = None


    def detector(self):
        data = []
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        header = ['sub_folder', 'image_name', 'detect_number']
        for i in range(68):
            header += ['part_{}_x'.format(i), 'part_{}_y'.format(i)]

        with open(os.path.join(self.save_dir, 'error.txt'), 'w') as err_writer:
            for root, dirs, files in os.walk(self.data_dir):
                for filename in files:
                    try:
                        fullname = os.path.join(root, filename)
                        img = Image.open(fullname)
                        img = img.convert('RGB')
                        preds = fa.get_landmarks(np.asarray(img))

                        if preds is None:
                            no_pre = 'no predict : {}\n'.format(fullname)
                            print(no_pre)
                            err_writer.write(no_pre)
                            continue

                        sub_folder = self.get_subfolder(fullname, self.data_dir)
                        print('detect {} face\n'.format(len(preds)))
            
                        for i in range(len(preds)):
                            row = []
                            row = [sub_folder, filename]
                            row += [i]
                            for j in range(68):
                                part_i_x = int(preds[i][j][0])
                                part_i_y = int(preds[i][j][1])
                                row += [part_i_x, part_i_y]
                            data.append(row)
                            print('save landmark {0}\n'.format(filename))
                    except (OSError, NameError):
                        err = 'Error {}\n'.format(filename)
                        print(err)
                        err_writer.write(err)

        save_csv = os.path.join(self.save_dir, 'landmark_{}.csv'.format(self.date_time))
        df = pd.DataFrame(data, columns=header)
        df.to_csv(save_csv, index=False)
        return save_csv

    def filter(self, csv_file, manual=False):
        strat_index = 0
        lm_data = pd.read_csv(csv_file)
        db_len = len(lm_data)

        self.face_dataset = FaceLandmarksSet(csv_file=csv_file, root_dir=self.data_dir)
        print('Start')
        for root, dirs, files in os.walk(self.data_dir):
            # strat_index
            for file_num, filename in enumerate(files[strat_index:]):
                fullname = os.path.join(root, filename)
                sub_folder = self.get_subfolder(fullname, self.data_dir)
                lm_index = self.face_dataset.multi_face_landmark(filename, sub_folder)
                multi_lm = len(lm_index)
                if multi_lm > 1:
                    if not manual:
                        save_name = 'auto_landmark_{}.csv'.format(self.date_time)
                        lm_data = self.auto_filter(lm_index, fullname, lm_data)
                    else:
                        save_name = 'manual_landmark_{}.csv'.format(self.date_time)
                        lm_data = self.manual_filter(lm_index, fullname, db_len, lm_data, strat_index, file_num, save_name)

        save_file = os.path.join(self.save_dir, save_name)
        print('Save csv to {}'.format(save_file))
        lm_data.to_csv(save_file, index=False)
        return save_file

    @staticmethod
    def get_subfolder(fullname, dir):
        sub_folder = fullname.split(dir)[-1].split(os.sep)
        sub_folder = sub_folder[1:-1] if len(sub_folder) != 2 else ['']
        sub_folder = os.path.join(*sub_folder)
        return sub_folder

    def auto_filter(self, lm_index, fullname, lm_data):
        points_dis = []
        for idx in lm_index:
            sample = self.face_dataset[idx]
            img_size = sample['image'].size
            lm_temp = copy.deepcopy(sample['landmarks'])
            lm_temp[:,0][lm_temp[:,0]<0] = 0
            lm_temp[:,1][lm_temp[:,1]<0] = 0
            lm_temp[:,0][lm_temp[:,0]>img_size[0]] = img_size[0]
            lm_temp[:,1][lm_temp[:,1]>img_size[1]] = img_size[1]
            horizontal = math.sqrt(sum((lm_temp[16] - lm_temp[0]) ** 2))
            vertical = math.sqrt(sum((lm_temp[28] - lm_temp[9]) ** 2))
            points_dis.append(horizontal + vertical)
        save_idx = np.argmax(points_dis)
        print('{}'.format(fullname))
        lm_index.pop(save_idx)
        lm_data.drop(lm_index, inplace=True)
        print('drop landmark index: {}'.format(lm_index))
        return lm_data

    def manual_filter(self, lm_index, fullname, db_len, lm_data, strat_index, file_num, save_name):
        SET = ['r', 's']
        def show_landmarks(image, landmarks):
            """Show image with landmarks"""
            plt.imshow(image)
            plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
            plt.pause(0.0001)  # pause a bit so that plots are updated
        plt.figure(figsize=(10, 8))
        for i, idx in enumerate(lm_index):
            sample = self.face_dataset[idx]
            ax = plt.subplot(2, math.ceil(len(lm_index) / 2), i+1)
            # plt.tight_layout()
            ax.set_title('#{}'.format(i))
            ax.axis('off')
            show_landmarks(**sample)

            if i+1 == len(lm_index):
                plt.show()
                plt.pause(0.0001)
                print('{}'.format(fullname))
                print('{}/{}'.format(max(lm_index), db_len))
                pick = input('pick the correct landmark up (r to remove all): ')
                while not pick.isdigit() and not pick in SET:
                    pick = input('pick the correct landmark up (r to remove all): ')

                if pick == 'r':
                    lm_data.drop(lm_index, inplace=True)
                    print('drop all landmark {}'.format(lm_index))
                    plt.close()
                    continue
                elif pick == 's':
                    lm_data.drop(lm_data.index[range(strat_index)], inplace=True)
                    lm_data.drop(lm_data.index[range(file_num-1, len(lm_data))], inplace=True)
                    print('Save csv to {}'.format(save_name))
                    lm_data.to_csv(save_name, index=False)
                    save_file = os.path.join(self.save_dir, save_name)
                    print('part of data save to {}'.format(save_file))
                    record = open(os.path.join(self.save_dir, 'landmark_record.txt'), 'a')
                    text = 'save path: {}, start_index: {}\n'.format(save_file, file_num)
                    record.write(text)
                    exit()
                else:
                    pick = int(pick)
                    lm_index.pop(pick)
                    lm_data.drop(lm_index, inplace=True)
                    print('drop landmark index: {}'.format(lm_index))
                    plt.close()
        return lm_data
