import os
import re 
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
import torch.nn.functional as F
from parsing_model.model import BiSeNet
from util.key_interrupt import UserCommand,open_dir
from pathlib import Path


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_vetor,specific_person_id_nonorm,id_thres, swap_model, detect_model, save_path, temp_results_dir=r'C:\dumpinggrounds', crop_size=224, no_simswaplogo = False,use_mask =False):
    Path(temp_results_dir).mkdir(exist_ok=True,parents=True)
    Path(save_path).parent.mkdir(exist_ok=True,parents=True)
    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    ret = True
    frame_index = 0

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = video.get(cv2.CAP_PROP_FPS)
    # if  os.path.exists(temp_results_dir):
            # shutil.rmtree(temp_results_dir)

    spNorm =SpecificNorm()
    mse = torch.nn.MSELoss().cuda()
    
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
    frame_name_suffix = Path(save_path).stem
    temp_path = Path(temp_results_dir) / frame_name_suffix
    temp_path.mkdir(exist_ok=True)
    temp_results_dir = str(temp_path)
    should_break = [False]
    def change_flag():
        print(should_break[0])
        should_break[0] = True
    detection_result_path_s = Path(video_path).parent.parent / 'detection_records' / Path(video_path).stem
    # detection_result_path_s.mkdir(parents=True,exist_ok=True)
    # print(detection_result_path_s)
    for frame_index in tqdm(range(frame_count)): 
        
        # print(should_break)
        if should_break[0]:
            break
        ret, frame = video.read()
        temp_img_path = os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index))
        command_dict = {
        
        b'd': lambda :open_dir(str(Path(video_path).parent)),
        b't': lambda :open_dir(temp_results_dir),
        b'c': change_flag,
        
        }
        UserCommand(command_dict)
        if Path(temp_img_path).is_file():
            continue
        if  ret:
            code = detection_result_path_s / (Path(temp_img_path).stem+'.fc')
            detect_results = detect_model.get(frame,crop_size)

            if detect_results is not None:
                # print(frame_index)

                if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                frame_align_crop_list = detect_results[0]
                frame_mat_list = detect_results[1]
                # swap_result_list = []
                id_compare_values = [] 
                frame_align_crop_tenor_list = []
                
                for frame_align_crop in frame_align_crop_list:

                    # BGR TO RGB
                    # frame_align_crop_RGB = frame_align_crop[...,::-1]

                    frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
                    # import pdb;pdb.set_trace()
                    # swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                    # cv2.imwrite(temp_img_path, frame)
                    # swap_result_list.append(swap_result)
                    # frame_align_crop_tenor_list.append(frame_align_crop_tenor)

                    frame_align_crop_tenor_arcnorm = spNorm(frame_align_crop_tenor)
                    frame_align_crop_tenor_arcnorm_downsample = F.interpolate(frame_align_crop_tenor_arcnorm, size=(112,112))
                    frame_align_crop_crop_id_nonorm = swap_model.netArc(frame_align_crop_tenor_arcnorm_downsample)

                    id_compare_values.append(mse(frame_align_crop_crop_id_nonorm,specific_person_id_nonorm).detach().cpu().numpy())
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)
                id_compare_values_array = np.array(id_compare_values)
                min_index = np.argmin(id_compare_values_array)
                min_value = id_compare_values_array[min_index]
                if min_value < id_thres:
                    swap_result = swap_model(None, frame_align_crop_tenor_list[min_index], id_vetor, None, True)[0]
                    
                    reverse2wholeimage([frame_align_crop_tenor_list[min_index]], [swap_result], [frame_mat_list[min_index]], crop_size, frame, logoclass,\
                         temp_img_path,no_simswaplogo,pasring_model =net,use_mask= use_mask, norm = spNorm)
                else:
                    if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                    frame = frame.astype(np.uint8)
                    if not no_simswaplogo:
                        frame = logoclass.apply_frames(frame)
                    cv2.imwrite( str(temp_img_path), frame)

            else:
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
                frame = frame.astype(np.uint8)
                if not no_simswaplogo:
                    frame = logoclass.apply_frames(frame)
                cv2.imwrite( str(temp_img_path), frame)
        else:
            break

    video.release()

    # image_filename_list = []
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))
    try:
        clips = ImageSequenceClip(image_filenames,fps = fps)
    except Exception as e:
        # import pdb;pdb.set_trace()
        filename_template = 'frame_%s.jpg'
        filepath_delete = Path(re.search('(?<=open `).*?(?=`)',str(e))[0])
        i = int(re.search('(?<=frame_).*',filepath_delete.stem)[0])
        for k in range(1,20):
            # import pdb;pdb.set_trace()
            filepath_delete.unlink()
            filepath_delete = filepath_delete.parent / (filename_template % str(i+k).zfill(7))
        print(e)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)
    # import pdb;pdb.set_trace()
    if Path(save_path).suffix != '.mp4':
        save_path = str(Path(save_path).with_suffix('.mp4'))
        

    clips.write_videofile(save_path,audio_codec='aac')
    shutil.rmtree(temp_results_dir)

