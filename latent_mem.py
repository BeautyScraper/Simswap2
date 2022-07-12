'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:43
Description: 
'''


from pathlib import Path
from random import shuffle,randint
from PIL import Image
import cv2
import torch.nn.functional as F
import pickle

class latent:
    def __init__(self,app,transformer_Arcface,model,opt):
        self.app = app
        self.transformer_Arcface = transformer_Arcface
        self.crop_size = opt.crop_size
        self.model = model
    def store_latent(latent,code):
            ftc = Path(code)   
            if ftc.is_file():
                print('file already exist')
                return
            ftc.parent.mkdir(exist_ok=True,parents=True)
            cont_to_save = latent
            fptc = open(ftc,'wb')
            pickle.dump(cont_to_save,fptc)
            fptc.close()

    def get_code(x): 
        return str(Path(x).parent / 'latent_records'/ (Path(x).stem + '.latent'))

    def get_mean_latent(self,dir_p):
        # x = latent()
        c = 0
        sum_latent = None
        for img_file in Path(dir_p).glob('*.jpg'):
            x = str(img_file)
            if sum_latent is None:
                sum_latent = self.get_latent(x,latent.get_code(x)) 
            else:
                sum_latent = sum_latent + self.get_latent(x,latent.get_code(x)) 
            c += 1
        sum_latent = (sum_latent / c)
        return sum_latent
            
    def get_latent(self,pic_a,code):
        ftc = Path(code)
        ftc.parent.mkdir(exist_ok=True,parents=True)
        # else:
            # ftc = self.detection_result_path_s / code
        # import pdb;pdb.set_trace()
        # ftc.parent.mkdir(exist_ok=True)
        if not ftc.is_file():
            cont_to_save = self.get_latent_real(pic_a)
            fptc = open(ftc,'wb')
            pickle.dump(cont_to_save,fptc)
            fptc.close()
        else:
            
            fptc = open(ftc,'rb')
            cont_to_save = pickle.load(fptc)
            # import pdb;pdb.set_trace()
        return cont_to_save

    def get_latent_real(self,pic_a):
        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = self.app.get(img_a_whole,self.crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = self.transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        # img_id_downsample = F.interpolate(img_id, size=(200,200))
        latend_id = self.model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)
        return latend_id