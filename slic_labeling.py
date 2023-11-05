# -*- coding: utf-8 -*-
"""
"""
import os
import pickle
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation as dila
from skimage.segmentation import mark_boundaries, slic
# -----------------------------------------------------------------------------/


def bwRGB(bw,im):
    """
    """
    A = np.sum(bw)
    B = np.sum(im[bw,0])/A
    G = np.sum(im[bw,1])/A
    R = np.sum(im[bw,2])/A
    return [R,G,B]
    # -------------------------------------------------------------------------/


def col_dis(color1,color2):
    """
    """
    sum = 0
    for i in range(3):
        ds = (float(color1[i]) - float(color2[i]))**2
        sum =sum+ds
    delta_e = np.sqrt(sum)
    return delta_e
    # -------------------------------------------------------------------------/


def save_segment_result(save_path:Path, seg:np.ndarray):
    """
    """
    with open(save_path, mode="wb") as f:
        pickle.dump(seg, f)
    # -------------------------------------------------------------------------/


def save_seg_on_img(save_path:Path, img:np.ndarray, seg:np.ndarray):
    """
    """
    seg_on_img = np.uint8(mark_boundaries(img, seg)*255)
    cv2.imwrite(str(save_path), seg_on_img)
    # -------------------------------------------------------------------------/


def run_single_slic_process(dir:Path, img_path:str,
                              merge:int, dark:int):
    """
    """
    img_name = os.path.split(img_path)[-1]
    img_name = os.path.splitext(img_name)[0]
    
    img = cv2.imread(img_path)
    
    seg0 = slic(img, n_segments = 250,
                     channel_axis=-1,
                     convert2lab=True,
                     enforce_connectivity=True,
                     slic_zero=False, compactness=30,
                     max_num_iter=100,
                     sigma = [1.7,1.7],
                     spacing=[1,1], # 3D: z, y, x; 2D: y, x
                     min_size_factor=0.4,
                     max_size_factor=3,
                     start_label=0)
        # parameters can refer to https://www.kite.com/python/docs/skimage.segmentation.slic

    """ save original `seg_result` ( without merge ) """
    save_path = dir.joinpath(f"{img_name}.seg0.pkl")
    save_segment_result(save_path, seg0)

    """ overlapping original image with its `seg_result` """
    save_path = dir.joinpath(f"{img_name}.seg0.png")
    save_seg_on_img(save_path, img, seg0)

    """ merging neighbors ('black background' and 'similar color') """
    lindex = 501 # new labels on seg1 starts from 501
    seg1 = deepcopy(seg0)
    labels = np.unique(seg0)
    for label in labels:
        if label > 0 and label < 900:
            bw = seg1 == label
            A = np.sum(bw)
            if A > 0:
                color1 = bwRGB(bw, img)
                color_dist = col_dis(color1, [0,0,0]) # compare with 'black background'
                if color_dist < dark:
                    seg1[seg1==label] = 0 # dark region on seg1 is labeled as 0
                else:
                    seg1[seg1==label] = lindex
                    # looking for neighbors
                    bwd = dila(bw)
                    nlabels = np.unique(seg1[bwd]) # neibor's labels
                    for nl in nlabels:
                        if nl > label and nl < 500:
                            bw2 = seg1 == nl
                            color2 = bwRGB(bw2, img)
                            if col_dis(color1, color2) < merge:
                                seg1[seg1==nl] = lindex
                lindex +=1

    """ save merged `seg_result` """
    save_path = dir.joinpath(f"{img_name}.seg1.pkl")
    save_segment_result(save_path, seg1)

    """ overlapping original image with merged `seg_result` """
    save_path = dir.joinpath(f"{img_name}.seg1.png")
    save_seg_on_img(save_path, img, seg1)
    
    return seg1
    # -------------------------------------------------------------------------/


if __name__ == '__main__':

    # colloct image file names
    path0 = Path('./') # directory of input images, images extension: .tif / .tiff

    # scan files
    files = path0.glob("*.tif*")
    files = [str(path) for path in files]
    print('total files:', len(files))

    """ slic each image """
    # these are two parameters as color space distance, determined by experiences
    merge = 12
    dark  = 40

    for file in files:
        
        merged_seg = run_single_slic_process(path0, file, merge, dark)
        cell_count = len(np.unique(merged_seg))-1
        with open(path0.joinpath(f"cell_count_{cell_count}"), mode="w") as f_writer: pass # 估計的細胞數量。 P.S. -1 是因為 label 0 是 background
