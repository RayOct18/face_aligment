# Face Alignment
A tool for aligning faces.

## Environment
* Windows
* Ubuntu18.04

## Prerequisites
* Pytorch
* Skimage
* OpenCV
* Numpy
* Matplotlib
* Pandas

## Reference
The following features are referenced from<br />
**2D and 3D Face alignment library build using pytorch:** https://github.com/1adrianb/face-alignment (Landmark detector)<br />
**Insightface:** https://github.com/deepinsight/insightface (Crop face)<br />
**Blur Detection with opencv-python:** https://github.com/indyka/blur-detection (Blur detection)<br />

## How to use
### One button
If you want to process images from scratch<br />
<code>python main.py --data_dir [image_dir] --save_dir [save_dir] --img_size 128</code><br />
*--data_dir* is the path of unprocessed images.<br />
*--save_dir* is the path to save output.<br />
*--img_size* is the cropped images size.<br />

<p align="center">
<img src="/fig/sample.png" alt="crop_sample" width="500"/>
</p>

### Manually filter landmark
Automatically filtering landmarks is to select the largest face on the image.<br />
If you want to fliter landmark by manual (add *--lm_csv [csv_dri]*, if you have already detected landmark)<br />
<code>python main.py --data_dir [image_dir] --save_dir [save_dir] --img_size 128 --manual</code><br />

It will show the following window (Use spyder is easier to operate)<br />

<p align="center">
<img src="/fig/filter_lm.png" alt="filter_lm" width="500"/>
</p>

then check the terminal, you will see<br />
<code>pick the correct landmark up (r to remove all, s to save a temporary file):</code><br />
- Input the index you want to preserve.<br />
- If you don't want to preserve any landmark you can input `r`. <br />
- If you want to save the file temporary, just input `s`, and when you start it again, it will start with the index you ended.

### Image filter
If you just want to filter image<br />
<code>python main.py --cropped_dir [cropped_dir] --black_threshold 60.0 --blur_threshold 10.0</code><br />
*--cropped_dir* is the path of cropped images.<br />
*--black_threshold* is the ratio of the black area and image.<br />
*--blur_threshold* is the Laplacian of image, the lower value indicates a blurry image.<br />

<p align="center">
<img src="/fig/filter_img.png" alt="filter_img" width="500"/>
</p>
