'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:00:38
Description: 
'''

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single_mem import Face_detect_crop
from util.videoswap import video_swap
from util.imgdirswap import img_dir_swap
import os
from pathlib import Path
from random import shuffle

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# detransformer = transforms.Compose([
#         transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
#         transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
#     ])
def single_src_dir_dst(src_img_file_path, targetDir, outDir,count_limit = -1):
    
    # count_limit = -1
    opt = TestOptions().parse()

    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    model = create_model(opt)
    model.eval()
    
    # code_dir_path = Path(targetDir) / 'FSIface_cropDB'/ Path(targetDir).name
    app = Face_detect_crop(name='antelope', root=r'C:\app\simswap\insightface_func\models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)
    with torch.no_grad():
        pic_apath = src_img_file_path
        pic_a = str(pic_apath)
        # img_a = Image.open(pic_a).convert('RGB')
        img_a_whole = cv2.imread(pic_a)
        src_code = src_img_file_path.parent / 'FSIface_cropDB'/ (Path(src_img_file_path).stem + '.fc')
        img_a_align_crop, _ = app.get(img_a_whole,crop_size,str(src_code))
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # pic_b = opt.pic_b_path
        # img_b_whole = cv2.imread(pic_b)
        # img_b_align_crop, b_mat = app.get(img_b_whole,crop_size)
        # img_b_align_crop_pil = Image.fromarray(cv2.cvtColor(img_b_align_crop,cv2.COLOR_BGR2RGB)) 
        # img_b = transformer(img_b_align_crop_pil)
        # img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        # img_att = img_att.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        img_dir_swap(targetDir, latend_id * 1.5, model, app, pic_apath.name,temp_results_dir=outDir,\
            no_simswaplogo=opt.no_simswaplogo,use_mask=opt.use_mask,crop_size=crop_size,count=count_limit)

def src_dir(indir,target_dir,output_dir, randomize_src_files = False,trc = -1,selected_src_count=-1):
    src_img_files = [x for x in Path(indir).glob('*.jpg')]
    if len (src_img_files) == 0:
        return
    if randomize_src_files:
        shuffle(src_img_files)
    
    if not selected_src_count == -1:
        src_img_files = src_img_files[:selected_src_count]
        # selected_src_count = 
    for imgFilePath in src_img_files:
        single_src_dir_dst(imgFilePath,target_dir,output_dir,trc)
            
if __name__ == '__main__':
    args = TestOptions().parse()
    indir_global = args.indir
    # indir_global = r'D:\paradise\stuff\Essence\FS\all\Sluts'
    # targetDir_global = r'C:\Heaven\YummyBaker'
    # targetDir_global = r'C:\Heaven\YummyBaker'
    # targetDir_global = r'D:\paradise\stuff\new\imageset2\meri maa mujhse chud jati'
    # targetDir_global = 
    # targetDir_global = r'D:\paradise\stuff\new\pvd2'
    targetDir_global = args.target_dir
    outDir_global = args.output_dir
    # outDir_global = r'D:\Developed\FaceSwapExperimental\TestResult'
    # outDir_global = r'C:\Games\sacred2'
    src_dir(indir_global,targetDir_global,outDir_global,True)
    
