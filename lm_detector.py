import face_alignment
import os
from PIL import Image
import pandas as pd
import numpy as np


def landmark_detector(data_dir, save_dir):
    
    data = []
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    header = ['sub_folder', 'image_name', 'detect_number']
    for i in range(68):
        header += ['part_{}_x'.format(i), 'part_{}_y'.format(i)]

    with open(os.path.join(save_dir, 'error.txt'), 'w') as err_writer:
        for root, dirs, files in os.walk(data_dir):
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
                    else:
                        print('detect {} face\n'.format(len(preds)))

                    sub_folder = fullname.split(data_dir)[-1].split(os.sep)
                    sub_folder = sub_folder[1:-1] if len(sub_folder) != 2 else ['']
                    sub_folder = os.path.join(*sub_folder)
            
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

    save_csv = os.path.join(save_dir, 'landmark.csv')
    df = pd.DataFrame(data, columns=header)
    df.to_csv(save_csv, index=False)
    return save_csv
