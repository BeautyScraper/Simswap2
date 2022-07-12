'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:52
Description: 
'''
import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
from pathlib import Path
from random import shuffle

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def img_dir_swap(target_dir, id_vetor, swap_model, detect_model, src_file_name, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False,fs_record_p = '', count=-1,randomize_dst= False):
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')

    if fs_record_p == '':
        fs_record_p = Path(target_dir) / 'FSRecords'
        
    fs_record_p.mkdir(exist_ok=True) 
    dbfilename = fs_record_p / (Path(src_file_name).stem+'.csv')
    donedata = open(dbfilename, 'a+') 
    donedata.seek(0,0)  
    # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fcontent = [x.strip() for x in donedata.readlines()]
    setfcontent = set(fcontent)
    # import pdb;pdb.set_trace()
    donedata.close()
    # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    
    spNorm =SpecificNorm()
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join(r'C:\app\simswap\parsing_model', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None

    # while ret:
    filelist = [x for x in Path(target_dir).glob('*.jpg') if str(x) not in setfcontent]
    if randomize_dst:
        shuffle(filelist)
    for img_fp in tqdm(filelist): 
        frame = cv2.imread(str(img_fp))
        code_dir_path = Path(target_dir) / 'FSIface_cropDB'/ Path(target_dir).stem
        code = code_dir_path / (img_fp.name + '.fc')
        detect_results = detect_model.get(frame,crop_size,str(code))
        # print(frame_index)
        if count == 0:
            break
        if detect_results is not None:
            if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
            frame_align_crop_list = detect_results[0]
            frame_mat_list = detect_results[1]
            swap_result_list = []
            frame_align_crop_tenor_list = []
            for frame_align_crop in frame_align_crop_list:

                # BGR TO RGB
                # frame_align_crop_RGB = frame_align_crop[...,::-1]

                frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                outfilePath = Path(temp_results_dir) / (Path(src_file_name).stem + ' @hudengi ' + Path(target_dir).name + ' W1t81N ' + img_fp.name)
                # import pdb;pdb.set_trace()
                
                cv2.imwrite(str(outfilePath), frame)
                swap_result_list.append(swap_result)
                frame_align_crop_tenor_list.append(frame_align_crop_tenor)

                    

                reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
                    str(outfilePath),no_simswaplogo,pasring_model =net,use_mask=use_mask, norm = spNorm)
            count -= 1
        donedata = open(dbfilename, 'a+')
        donedata.write('\n'+ str(img_fp)) 
        donedata.close()
