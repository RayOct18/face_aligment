import os

def get_subfolder(fullname, dir):
    sub_folder = fullname.split(dir)[-1].split(os.sep)
    sub_folder = sub_folder[1:-1] if len(sub_folder) != 2 else ['']
    sub_folder = os.path.join(*sub_folder)
    return sub_folder