'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:22
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
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.add_watermark import watermark_image
import torch.nn as nn
from util.norm import SpecificNorm
import glob
from parsing_model.model import BiSeNet
from pathlib import Path
import re
from itertools import permutations  


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def _toarctensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

        
    
    
def model_mem():
    swap_result_list = {}
    def model_run(model_m,b_align_crop_tenor_list,tmp_index,target_id_norm_list,min_index,reset=False):
        if reset == True:
            swap_result_list.clear()
            # print('***************list is cleared ***************')
            # print('***************list is cleared ***************')
            # print('***************list is cleared ***************')
            return
        if (tmp_index,min_index) not in swap_result_list:
            swap_result = model_m(None, b_align_crop_tenor_list[tmp_index], target_id_norm_list[min_index], None, True)[0]
            swap_result_list[(tmp_index,min_index)] = swap_result
        else:
            print('alredy got it Don\'t Worry ',(tmp_index,min_index))
            swap_result = swap_result_list[(tmp_index,min_index)]
        return swap_result
    return model_run
    
opt = TestOptions().parse()
# model_mem_g = model_mem()

start_epoch, epoch_iter = 1, 0
crop_size = opt.crop_size

# multisepcific_dir = r'D:\paradise\stuff\simswappg\srcs'

torch.nn.Module.dump_patches = True

if crop_size == 512:
    opt.which_epoch = 550000
    opt.name = '512'
    mode = 'ffhq'
else:
    mode = 'None'
spNorm =SpecificNorm()


app = Face_detect_crop(name='antelope', root=r'C:\app\simswap\insightface_func\models')
app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode = mode)


class faceswap:
     def __init__(self,srcDir,target_pic_path_str='',resulDir=''):
        self.logoclass = watermark_image('./simswaplogo/simswaplogo.png')
        self.srcDir = Path(srcDir)
        self.model = create_model(opt)
        self.model.eval()
        self.target_id_norm_list = []
        self.srclist()
        self.swap_result_list = {}
        if not resulDir == '':
            self.resulDir = Path(resulDir)
        else:
            self.resulDir = Path().cwd()
        
        if not target_pic_path_str == '':
            self.set_for_target_pic(target_pic_path_str)
            
     def set_for_target_pic(self,target_pic_path_str):
            self.swap_result_list.clear()
            self.target_pic_path_str = Path(target_pic_path_str)
            self.img_b_whole = cv2.imread(str(self.target_pic_path_str))
            self.b_align_crop_tenor_list = []
            self.b_mat_list = []
            self.target_pic_face_count = -1
            self.swap_result_list.clear()
            self.setDst_picList()
            
     def model_run(self,b_align_crop_tenor_list,tmp_index,target_id_norm_list,min_index):
        if (tmp_index,min_index) not in self.swap_result_list:
            swap_result = self.model(None, b_align_crop_tenor_list[tmp_index], target_id_norm_list[min_index], None, True)[0]
            self.swap_result_list[(tmp_index,min_index)] = swap_result
        else:
            print('alredy got it Don\'t Worry ',(tmp_index,min_index))
            swap_result = self.swap_result_list[(tmp_index,min_index)]
        return swap_result
        
     def do_fs_with_perm(self,swap_list=[]):
        result_dir_path = str(self.resulDir)
        multisepcific_dir = str(self.srcDir)
        target_pic_path_s = str(self.target_pic_path_str)
        
        with torch.no_grad():
            if swap_list == []:
                min_indexs = list(range(0,len(self.b_align_crop_tenor_list)))[:len(self.target_id_norm_list)]
            else:
                min_indexs = swap_list
                
            swap_result_list = [] 
            swap_result_matrix_list = []
            swap_result_ori_pic_list = []
            print(min_indexs)
            for tmp_index, min_index in enumerate(min_indexs):
                if min_index < len(self.target_id_norm_list) and min_index >= 0:
                    swap_result = self.model_run(self.b_align_crop_tenor_list,tmp_index,self.target_id_norm_list,min_index)
                    # model.eval()
                    # swap_result = model(None, b_align_crop_tenor_list[tmp_index], target_id_norm_list[min_index], None, True)[0]
                    swap_result_list.append(swap_result)
                    swap_result_matrix_list.append(self.b_mat_list[tmp_index])
                    swap_result_ori_pic_list.append(self.b_align_crop_tenor_list[tmp_index])
                else:
                    pass

            if len(swap_result_list) !=0:

                if opt.use_mask:
                    n_classes = 19
                    net = BiSeNet(n_classes=n_classes)
                    net.cuda()
                    save_pth = os.path.join(r'C:\app\simswap\parsing_model', '79999_iter.pth')
                    net.load_state_dict(torch.load(save_pth))
                    net.eval()
                else:
                    net =None
                Path(result_dir_path).mkdir(exist_ok=True)
                resultFilePath = str(Path(result_dir_path) / ( Path(multisepcific_dir).name + Path(target_pic_path_s).stem + str(min_indexs) )) + '.jpg'
                reverse2wholeimage(swap_result_ori_pic_list, swap_result_list, swap_result_matrix_list, crop_size, self.img_b_whole, self.logoclass,\
                    resultFilePath, opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)

                print(' ')

                print('************ Done ! ************')
     
     def setDst_picList(self): 
        self.b_mat_list.clear()
        self.b_align_crop_tenor_list.clear()
        with torch.no_grad():
            img_b_whole = self.img_b_whole

            img_b_align_crop_list, self.b_mat_list = app.get(img_b_whole,crop_size)
            
            for b_align_crop in img_b_align_crop_list:
                b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
                self.b_align_crop_tenor_list.append(b_align_crop_tenor)
        self.target_pic_face_count = len(self.b_align_crop_tenor_list)
     
     def srclist(self):
        with torch.no_grad():
            target_path = os.path.join(str(self.srcDir),'*.jpg')
            target_images_path = sorted(glob.glob(target_path))

            for target_image_path in target_images_path:
                img_a_whole = cv2.imread(target_image_path)
                img_a_align_crop, _ = app.get(img_a_whole,crop_size)
                img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
                img_a = transformer_Arcface(img_a_align_crop_pil)
                img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
                # convert numpy to tensor
                img_id = img_id.cuda()
                # create latent id
                img_id_downsample = F.interpolate(img_id, size=(112,112))
                latend_id = self.model.netArc(img_id_downsample)
                latend_id = F.normalize(latend_id, p=2, dim=1)
                self.target_id_norm_list.append(latend_id.clone())
    

        
        
        
def multifacewap(multisepcific_dir, target_pic_path_s, result_dir_path,swap_list=[]):
    # print(swap_list)

    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    model = create_model(opt)
    model.eval()
    # mse = torch.nn.MSELoss().cuda()


    with torch.no_grad():
    
        # The specific person to be swapped(source)

        # source_specific_id_nonorm_list = []
        # source_path = os.path.join(multisepcific_dir,'*.jpg')
        # source_specific_images_path = sorted(glob.glob(source_path))

        # for source_specific_image_path in source_specific_images_path:
            # specific_person_whole = cv2.imread(source_specific_image_path)
            # specific_person_align_crop, _ = app.get(specific_person_whole,crop_size)
            # specific_person_align_crop_pil = Image.fromarray(cv2.cvtColor(specific_person_align_crop[0],cv2.COLOR_BGR2RGB)) 
            # specific_person = transformer_Arcface(specific_person_align_crop_pil)
            # specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1], specific_person.shape[2])
            # convert numpy to tensor
            # specific_person = specific_person.cuda()
            #create latent id
            # specific_person_downsample = F.interpolate(specific_person, size=(112,112))
            # specific_person_id_nonorm = model.netArc(specific_person_downsample)
            # source_specific_id_nonorm_list.append(specific_person_id_nonorm.clone())


        # The person who provides id information (list)
        target_id_norm_list = []
        target_path = os.path.join(multisepcific_dir,'*.jpg')
        target_images_path = sorted(glob.glob(target_path))

        for target_image_path in target_images_path:
            img_a_whole = cv2.imread(target_image_path)
            img_a_align_crop, _ = app.get(img_a_whole,crop_size)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            # convert numpy to tensor
            img_id = img_id.cuda()
            # create latent id
            img_id_downsample = F.interpolate(img_id, size=(112,112))
            latend_id = model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)
            target_id_norm_list.append(latend_id.clone())

        # assert len(target_id_norm_list) == len(source_specific_id_nonorm_list), "The number of images in source and target directory must be same !!!"

        ############## Forward Pass ######################

        # pic_b = r'C:\Games\MultiFaces\done\023.jpg'
        pic_b = target_pic_path_s
        img_b_whole = cv2.imread(pic_b)

        img_b_align_crop_list, b_mat_list = app.get(img_b_whole,crop_size)
        # detect_results = None
        swap_result_list = []


        id_compare_values = [] 
        b_align_crop_tenor_list = []
        for b_align_crop in img_b_align_crop_list:

            b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

            # b_align_crop_tenor_arcnorm = spNorm(b_align_crop_tenor)
            # b_align_crop_tenor_arcnorm_downsample = F.interpolate(b_align_crop_tenor_arcnorm, size=(112,112))
            # b_align_crop_id_nonorm = model.netArc(b_align_crop_tenor_arcnorm_downsample)

            # id_compare_values.append([])
            # for source_specific_id_nonorm_tmp in source_specific_id_nonorm_list:
                # id_compare_values[-1].append(mse(b_align_crop_id_nonorm,source_specific_id_nonorm_tmp).detach().cpu().numpy())
            b_align_crop_tenor_list.append(b_align_crop_tenor)
        # def actual_multi_faceswap(swap_list=[]):
        # id_compare_values_array = np.array(id_compare_values).transpose(1,0)
        # min_indexs = np.argmin(id_compare_values_array,axis=0)
        # min_value = np.min(id_compare_values_array,axis=0)
        # if len(facesInImg)
        # import pdb;pdb.set_trace()
        if swap_list == []:
            min_indexs = list(range(0,len(b_align_crop_tenor_list)))[:len(target_id_norm_list)]
        else:
            min_indexs = swap_list
            
        swap_result_list = [] 
        swap_result_matrix_list = []
        swap_result_ori_pic_list = []
        print(min_indexs)
        for tmp_index, min_index in enumerate(min_indexs):
            if min_index< len(target_id_norm_list) and min_index >= 0:
                swap_result = model_mem_g(model,b_align_crop_tenor_list,tmp_index,target_id_norm_list,min_index)
                model.eval()
                # swap_result = model(None, b_align_crop_tenor_list[tmp_index], target_id_norm_list[min_index], None, True)[0]
                swap_result_list.append(swap_result)
                swap_result_matrix_list.append(b_mat_list[tmp_index])
                swap_result_ori_pic_list.append(b_align_crop_tenor_list[tmp_index])
            else:
                pass

        if len(swap_result_list) !=0:

            if opt.use_mask:
                n_classes = 19
                net = BiSeNet(n_classes=n_classes)
                net.cuda()
                save_pth = os.path.join(r'C:\app\simswap\parsing_model', '79999_iter.pth')
                net.load_state_dict(torch.load(save_pth))
                net.eval()
            else:
                net =None
            Path(result_dir_path).mkdir(exist_ok=True)
            resultFilePath = str(Path(result_dir_path) / ( Path(multisepcific_dir).name + Path(target_pic_path_s).stem + str(min_indexs) )) + '.jpg'
            reverse2wholeimage(swap_result_ori_pic_list, swap_result_list, swap_result_matrix_list, crop_size, img_b_whole, logoclass,\
                resultFilePath, opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)

            print(' ')

            print('************ Done ! ************')
        
        else:
            print('The people you specified are not found on the picture: {}'.format(pic_b))
        swap_list.clear()
        return len(b_align_crop_tenor_list)
        # return actual_multi_faceswap
def FSon_permutation(multisepcific_dir,imgfiles,result_dir,facesInImg,setfilecontent,csvFile,face_swap_obj):    
    facesinsrcdir = len([x for x in Path(multisepcific_dir).glob('*.jpg')])
    if facesinsrcdir <= facesInImg:
        swap_list = list(range(0,facesinsrcdir)) + [-1] * (facesInImg - facesinsrcdir)
        r = len(swap_list)
    else:
        swap_list = list(range(0,facesinsrcdir)) + [-1]  
        r = facesInImg
    
    for swaplist in list(permutations(swap_list,r)):
        # print()
        assert len(swaplist) == facesInImg
        # import pdb;pdb.set_trace()
        if not -1 in swaplist: # keep original face
            continue
        checkCode = str(imgfiles)+'@[%s]' % str(swaplist)
        if checkCode in setfilecontent:
            continue
        try:
            face_swap_obj.do_fs_with_perm(list(swaplist))
            # multifacewap()
            fs = open(csvFile, 'a+')
            fs.write('\n' + checkCode)
            fs.close()
        except Exception as e:
            print('msg' + e)

def set_target_image_mem(imgfiles, multisepcific_dir, target_dir_path, result_dir,csvFileLocationdir):
    csvFile = csvFileLocationdir / (imgfiles.parent.name + '_completed.csv') 
    if not csvFile.is_file():
        setfilecontent = set([])
    else:
        fs = open(csvFile,'r')
        setfilecontent = set([x.strip() for x in fs.readlines()])
        fs.close()
    if str(imgfiles) not in setfilecontent:
        # facesInImg = multifacewap(multisepcific_dir, str(imgfiles), result_dir)
        set_target_image(imgfiles, multisepcific_dir, target_dir_path, result_dir,csvFileLocationdir)
        fs = open(csvFile,'a+')
        fs.write(str(imgfiles)+'\n')
        # fs.write('\n'+str(imgfiles)+'@[%s]' % str(list(range(0,facesInImg))))
        fs.close()
    else:
        # import pdb;pdb.set_trace()
        print('Already done with this file')
        # fs = open(csvFile,'a+')
    
def set_target_image(imgfiles, multisepcific_dir, target_dir_path, result_dir,csvFileLocationdir,):
    csvFile = csvFileLocationdir / (imgfiles.parent.name+'_'+imgfiles.name+ '.csv')
    
    
    if not csvFile.is_file():
        setfilecontent = set([])
    else:
        fs = open(csvFile,'r')
        setfilecontent = set([x.strip() for x in fs.readlines()])
        fs.close()
    # multifacewap = multifacewap_prepare(multisepcific_dir, str(imgfiles), result_dir)
    face_swap_obj = faceswap(multisepcific_dir,str(imgfiles),result_dir)
    facesInImg = face_swap_obj.target_pic_face_count
    FSon_permutation(multisepcific_dir,imgfiles,result_dir,facesInImg,setfilecontent,csvFile,face_swap_obj)    
                
    
def multiface_dir(multisepcific_dir, target_dir_path, result_dir):
    csvFileLocationdir = Path(multisepcific_dir) / 'MFSRecords'
    csvFileLocationdir.mkdir(exist_ok=True)
    # face_swap_obj = faceswap(srcDir,'',result_dir)
    # permutationList = []
    for imgfiles in Path(target_dir_path).glob('*.jpg'):
    
        set_target_image_mem(imgfiles, multisepcific_dir, target_dir_path, result_dir,csvFileLocationdir)
        # model_mem_g(None,None,None,None,None,True)
        
            
    
            
if __name__ == '__main__':
    # srcDir = r'D:\paradise\stuff\simswappg\srcs'
    # srcDir = r'D:\paradise\stuff\Essence\FS\CelebCombination\simcombination\nawabi'
    srcDir = r'D:\paradise\stuff\simswappg\combinationSrc\known'
    target_dir = r'C:\Games\MultiFaces'
    result_dir = r'C:\Games\NextFaceresult'
    # files = Path(target_dir).glob('*.jpg')
    # x = faceswap(srcDir,str(next(files)),result_dir)
    # x.do_fs_with_perm()
    # x.do_fs_with_perm([1,0])
    # x.set_for_target_pic(str(next(files)))
    # x.do_fs_with_perm()
    
    
    multiface_dir(srcDir,target_dir,result_dir)