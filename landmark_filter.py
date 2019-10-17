import os
import pandas as pd
import matplotlib.pyplot as plt
from face_landmark import FaceLandmarksSet
import datetime
import math
import pandas as pd
import copy
import numpy as np

plt.ion()   # interactive mode
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
strat_index = 0

def landmark_filter(csv_file, data_dir, manual=False):
    SET = ['r', 's']
    save_dir = os.path.split(csv_file)

    lm_data = pd.read_csv(csv_file)
    db_len = len(lm_data)

    def show_landmarks(self, image, landmarks):
        """Show image with landmarks"""
        plt.imshow(image)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
        plt.pause(0.0001)  # pause a bit so that plots are updated
   

    face_dataset = FaceLandmarksSet(csv_file=csv_file, root_dir=data_dir)
    print('Start')
    for root, dirs, files in os.walk(data_dir):
        # strat_index
        for file_num, filename in enumerate(files[strat_index:]):
            save_name = 'fauto_landmark_{}.csv'.format(date_time)
            fullname = os.path.join(root, filename)
            sub_folder = fullname.split(data_dir)[-1].split(os.sep)
            sub_folder = sub_folder[1:-1] if len(sub_folder) != 2 else ['']
            sub_folder = os.path.join(*sub_folder)
            lm_index = face_dataset.multi_face_landmark(filename, sub_folder)
            multi_lm = len(lm_index)
            if multi_lm == 0:
                continue
            elif multi_lm > 1:
                if not manual:
                    points_dis = []
                    for i, idx in enumerate(lm_index):
                        sample = face_dataset[idx]
                        img_size = sample['image'].size
                        lm_temp = copy.deepcopy(sample['landmarks'])
                        lm_temp[:,0][lm_temp[:,0]<0] = 0
                        lm_temp[:,1][lm_temp[:,1]<0] = 0
                        lm_temp[:,0][lm_temp[:,0]>img_size[0]] = img_size[0]
                        lm_temp[:,1][lm_temp[:,1]>img_size[1]] = img_size[1]
                        horizontal = math.sqrt(sum((lm_temp[16] - lm_temp[0]) ** 2))
                        vertical = math.sqrt(sum((lm_temp[28] - lm_temp[9]) ** 2))
                        points_dis.append(horizontal + vertical)
                        if i+1 == len(lm_index):
                            save_idx = np.argmax(points_dis)
                            print('{}'.format(os.path.join(root, filename)))
                            lm_index.pop(save_idx)
                            lm_data.drop(lm_index, inplace=True)
                            print('drop landmark index: {}'.format(lm_index))
                else:
                    save_name = 'fmanual_landmark_{}.csv'.format(date_time)
                    plt.figure(figsize=(10, 8))
                    for i, idx in enumerate(lm_index):
                        sample = face_dataset[idx]
                        ax = plt.subplot(2, math.ceil(len(lm_index) / 2), i+1)
                        # plt.tight_layout()
                        ax.set_title('#{}'.format(i))
                        ax.axis('off')
                        show_landmarks(**sample)

                        if i+1 == len(lm_index):
                            plt.show()
                            plt.pause(0.0001)
                            print('{}'.format(os.path.join(root, filename)))
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
                                save_name = 'fmanual_landmark_part{}.csv'.format(date_time)
                                print('Save csv to {}'.format(save_name))
                                lm_data.to_csv(save_name, index=False)
                                save_file = os.path.join(save_dir, save_name)
                                print('part of data save to {}'.format(save_file))
                                record = open(os.path.join(save_dir, 'landmark_record.txt'), 'a')
                                text = 'save path: {}, start_index: {}\n'.format(save_file, file_num)
                                record.write(text)
                                exit()
                            else:
                                pick = int(pick)
                                lm_index.pop(pick)
                                lm_data.drop(lm_index, inplace=True)
                                print('drop landmark index: {}'.format(lm_index))
                                plt.close()

    save_file = os.path.join(save_dir, save_name)
    print('Save csv to {}'.format(save_file))
    lm_data.to_csv(save_file, index=False)
    return save_file
