import os
import random 
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
from util.key_interrupt import UserCommand,open_dir
from pathlib import Path


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir=r'C:\dumpinggrounds', crop_size=224, no_simswaplogo = False,use_mask =False):
    Path(temp_results_dir).mkdir(exist_ok=True,parents=True)
    Path(save_path).parent.mkdir(exist_ok=True,parents=True)
    # if random.randint(0, 20) == 1:  
    # time.sleep(random.randint(0, 600))
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
            detect_results = detect_model.get(frame,crop_size,str(code))

            if detect_results is not None:
                # print(frame_index)

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
                    # import pdb;pdb.set_trace()
                    swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                    # cv2.imwrite(temp_img_path, frame)
                    swap_result_list.append(swap_result)
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)
                    
                    

                reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
                    temp_img_path,no_simswaplogo,pasring_model =net,use_mask=use_mask, norm = spNorm)

            else:
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
                frame = frame.astype(np.uint8)
                if not no_simswaplogo:
                    frame = logoclass.apply_frames(frame)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
        else:
            break

    video.release()
    # import pdb;pdb.set_trace()
    # image_filename_list = []
    path = os.path.join(temp_results_dir,'*.jpg')
    # image_filenames = sorted(glob.glob(path))
    image_filenames = [str(x) for x in sorted(Path(temp_results_dir).glob('*.jpg'))]
    try:
        clips = ImageSequenceClip(image_filenames,fps = fps)

        if not no_audio:
            clips = clips.set_audio(video_audio_clip)

        
        clips.write_videofile(save_path,audio_codec='aac',codec="libx264")
    except Exception as e:
        print(e)
        import pdb;pdb.set_trace()
    shutil.rmtree(temp_results_dir)

