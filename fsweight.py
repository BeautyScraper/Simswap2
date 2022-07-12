'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:43
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
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
from pathlib import Path
from random import shuffle,randint
def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

mse = torch.nn.MSELoss().cuda()

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)
    
def dofsmage(srcfileps,targetfps,resultfp=""):
    opt = TestOptions().parse()
    opt.pic_a_path = srcfileps
    opt.pic_b_path = targetfps
    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size
    # crop_size = 512

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    model = create_model(opt)
    model.eval()

    spNorm =SpecificNorm()
    app = Face_detect_crop(name='antelope', root=r'C:\app\simswap\insightface_func\models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)

    with torch.no_grad():
        
    
    
        pic_a = opt.pic_a_path

        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = app.get(img_a_whole,crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        # img_id_downsample = F.interpolate(img_id, size=(200,200))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)


        ############## Forward Pass ######################

        pic_b = opt.pic_b_path
        img_b_whole = cv2.imread(pic_b)

        img_b_align_crop_list, b_mat_list = app.get(img_b_whole,crop_size)
        # detect_results = None
        swap_result_list = []

        b_align_crop_tenor_list = []

        latend_id1 = latend_id + 0.5 * (latend_id)
        for b_align_crop in img_b_align_crop_list:

            b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
            swap_result1 = model(None, b_align_crop_tenor, latend_id1, None, True)[0]
            swap_result_img = swap_result1.cpu().detach().numpy().transpose((1, 2, 0)) * [255,255,255]
            b_align_crop_tenor1 = _totensor(swap_result_img)[None,...].cuda()
            b_align_crop_tenor_arcnorm = spNorm(b_align_crop_tenor1)
            b_align_crop_tenor_arcnorm_downsample = F.interpolate(b_align_crop_tenor_arcnorm, size=(112,112))
            b_align_crop_id_nonorm = model.netArc(b_align_crop_tenor_arcnorm_downsample)
            weight = mse(b_align_crop_id_nonorm,latend_id).detach().cpu().numpy()
            weight = mse(latend_id1,latend_id).detach().cpu().numpy()
            # import pdb;pdb.set_trace()
            # swap_result = model(None, b_align_crop_tenor1, latend_id, None, True)[0]
            
            swap_result_list.append(swap_result1)
            b_align_crop_tenor_list.append(b_align_crop_tenor)

        if opt.use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join(r'C:\app\simswap\parsing_model', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net =None
        if resultfp == "":
            outputdir = Path(r'D:\Developed\FaceSwapExperimental\TestResult')
            opt.output_path = str(outputdir / (Path(srcfileps).stem + Path(targetfps).stem + '.jpg'))
        else:
            opt.output_path = resultfp
        weight = int(10000 * weight)
        opt.output_path = str(Path(opt.output_path).with_name(str(weight)+ Path(opt.output_path).name))
        reverse2wholeimage(b_align_crop_tenor_list, swap_result_list, b_mat_list, crop_size, img_b_whole, logoclass, \
            opt.output_path, opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)

        print(' ')

        print('************ Done ! ************')
        # import pdb;pdb.set_trace()
        
if __name__ == '__main__':
    # srcimgdir = Path(r'D:\paradise\stuff\Essence\FS\all\Devi\Frames\New folder')
    srcimgdir = Path(r'D:\paradise\stuff\simswappg\srcs\gudYummy')
    # srcimgdir = Path(r'D:\paradise\stuff\Essence\FS\CelebCombination\ShreeDevis\shreedevi')
    # srcimgdir = Path(r'D:\paradise\stuff\Essence\FS\CelebCombination\Deols\esha')
    # srcimgdir = Path(r'D:\paradise\stuff\Essence\FS\SachMe')
    # srcimgdir = Path(r'D:\paradise\stuff\simswappg\srcs\gudcele')
    # srcimgdir = Path(r'D:\paradise\stuff\Essence\FS\all\Devi\DeviKa\1')
    # srcimgdir = Path(r'D:\paradise\stuff\Essence\FS\CelebCombination\Nawabi')
    # srcimgdir = Path(r'D:\paradise\stuff\Essence\FS\all\Sluts')
    # dstvideodir = Path(r'D:\paradise\stuff\simswappg\trialTargets')
    # dstvideodir = Path(r'D:\paradise\stuff\new\imageset2\Hustler Jessa Rhodes - Busty Young Wives - x94 - June 23 2021')
    # dstvideodir = Path(r'D:\paradise\stuff\Images\Best\softcore')
    # dstvideodir = Path(r'D:\paradise\stuff\Images\Champions')
    # dstvideodir = Path(r'D:\paradise\stuff\new\imageset\Killergram Black cock perfection - starring Harleyy Heart - published 11 December 2021 - number of photos 83')
    # dstvideodir = Path(r'C:\Heaven\YummyBaker')
    dstvideodir = Path(r'D:\paradise\stuff\essence\Pictures\Considerable')
    # dstvideodir = Path(r'D:\paradise\stuff\simswappg\trialTargets')
    # dstvideodir = Path(r'C:\GalImgs\BabesImgs')
    # dstvideodir = Path(r'D:\paradise\stuff\Images\Best\too hot')
    # dstvideodir = Path(r'D:\paradise\stuff\new\pvd2\PVD2')
    # dstvideodir = Path(r'D:\paradise\stuff\Images\Best\powerGirls')
    # dstvideodir = Path(r'D:\paradise\stuff\Images\Chudvati')
    # dstvideodir = Path(r'D:\paradise\stuff\new\imageset2\LegalPorno Nadja Lapiedra 2 On 1 BBC ATM First Time DAP No Pussy Gapes Creie Swallow GL496 - 19 Jun')
    # dstvideodir = Path(r'C:\GalImgs\imageSet\ Valentina Nappi Collection')
    # dstvideodir = Path(r'D:\paradise\stuff\new\PVD')
    # dstvideodir = Path(r'D:\paradise\stuff\new\pvd2\extractedVideo\xDivision')
    # dstvideodir = Path(r'D:\paradise\stuff\new\pvd2')
    # dstvideodir = Path(r'D:\paradise\stuff\new\imageset2\Brazzers Kirsten Price Nikki Benz - Can You Get Me Off x916')
    
    resultDir = Path(r'D:\paradise\stuff\simswappg\outputmages')
    # resultDir = Path(r'C:\Games\Sacred2')
    testsrc_times = -1
    randsrc = True
    randdst = True
    # targetfile = open('donedata.csv','w+')
    srcFileList = [x for x in srcimgdir.glob('*.jpg')]
    dstFileList = [x for x in dstvideodir.glob('*.jpg')]
    if randsrc:
        shuffle(dstFileList)  
        
    # dofsmage(r'D:\paradise\stuff\simswappg\srcs\gudYummy\Yummyx (17).jpg', r'D:\paradise\stuff\simswappg\srcs\gudYummy\Zummy.jpg',r'D:\paradise\stuff\simswappg\srcs\gudYummy\doubleYummy.jpg')
    for vidFIle in dstFileList:
      inputFilepath = vidFIle  
      shuffle(srcFileList)  
      print(srcFileList)
      for imgFiles in srcFileList:
        # nextTargetfileName =  str (randint(1,1000)) + vidFIle.name
        nextTargetfileName =  'DOUBLE ' + vidFIle.name
        rsultFilefp = resultDir / nextTargetfileName
        dofsmage(str(imgFiles), str(inputFilepath),str(rsultFilefp))
        # inputFilepath.unlink()
        inputFilepath = rsultFilefp
   