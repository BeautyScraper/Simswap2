'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 16:46:04
Description: 
'''
from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
from pathlib import Path
import hashlib
import pickle
import cv2
from insightface.model_zoo import model_zoo
from insightface_func.utils import face_align_ffhqandnewarc as face_align

__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                #print('ignore:', onnx_file)
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        # self.detection_result_path_s = Path(mem_path)
        # self.detection_result_path_s.mkdir(exist_ok=True,parents=True)
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)
    def get(self, img, crop_size,code, max_num=0):
        # m = hashlib.sha256()
        # m.update(img)
        # filename = m.hexdigest()
        # if Path(code).is_absolute():
        ftc = Path(code)
        ftc.parent.mkdir(exist_ok=True,parents=True)
        # else:
            # ftc = self.detection_result_path_s / code
        # import pdb;pdb.set_trace()
        # ftc.parent.mkdir(exist_ok=True)
        if not ftc.is_file():
            cont_to_save = self.get_real(img, crop_size, max_num=0)
            fptc = open(ftc,'wb')
            pickle.dump(cont_to_save,fptc)
            fptc.close()
        else:
            
            fptc = open(ftc,'rb')
            cont_to_save = pickle.load(fptc)
            # import pdb;pdb.set_trace()
        return cont_to_save
            
        
    def get_real(self, img, crop_size, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             threshold=self.det_thresh,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return None
        # import pdb;pdb.set_trace()
        # ret = []
        # for i in range(bboxes.shape[0]):
        #     bbox = bboxes[i, 0:4]
        #     det_score = bboxes[i, 4]
        #     kps = None
        #     if kpss is not None:
        #         kps = kpss[i]
        #     M, _ = face_align.estimate_norm(kps, crop_size, mode ='None') 
        #     align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
        # for i in range(bboxes.shape[0]):
        #     kps = None
        #     if kpss is not None:
        #         kps = kpss[i]
        #     M, _ = face_align.estimate_norm(kps, crop_size, mode ='None') 
        #     align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)

        det_score = bboxes[..., 4]

        # select the face with the hightest detection score
        best_index = np.argmax(det_score)

        kps = None
        if kpss is not None:
            kps = kpss[best_index]
        M, _ = face_align.estimate_norm(kps, crop_size, mode = self.mode) 
        align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
        
        return [align_img], [M]
