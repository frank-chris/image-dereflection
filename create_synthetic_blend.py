import os
import os.path as osp
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

def blend_layers(T,R,w):
    return (w*T + (1-w)*cv2.blur(R,(4,4))).astype(np.uint8)

def create_blended_dataset(pT,pR,pD,w):
    """
    pT: path to folder containing transmission layers
    pR: path to folder containing reflection layers
    pD: path to output directory
    w: blend weight according to the paper
    """
    
    if not osp.exists(pD):
        raise Exception("Output folder does not exist!")
    
    t_paths = glob.glob(f"{pT}/*.jpg")
    r_paths = glob.glob(f"{pR}/*.jpg")
    
    t_dict = {x.split("/")[-1]:x for x in t_paths}
    r_dict = {x.split("/")[-1]:x for x in r_paths}
    
    for img_id in t_dict:
        t_path = t_dict[img_id]
        r_path = r_dict[img_id]
        
        t_img = cv2.imread(t_path) 
        r_img = cv2.imread(r_path)
        
        b_img = blend_layers(t_img,r_img,w)
        
        cv2.imwrite(f"{osp.join(pD,img_id)}", b_img)

def visualize_blend(t_path, r_path):

    def combine_images(img1, img2, img3, spacing=20):
        spacing = 10
        
        height1, width1, channels1 = img1.shape
        height2, width2, channels2 = img2.shape
        height3, width3, channels3 = img3.shape
        
        new_height = height1
        new_width = width1 + width2 + width3 + 2*spacing
        new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        
        new_image[0:height1, 0:width1] = img1

        new_image[0:height2, width1+spacing:width1+width2+spacing] = img2
        
        new_image[0:height2, width1+spacing+width2+spacing:] = img3
        
        return new_image

    T = cv2.imread(t_path)
    R = cv2.imread(r_path)
    B = blend_layers(T,R,0.5)
    A = combine_images(T,R,B)