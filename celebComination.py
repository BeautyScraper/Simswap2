from simmultiface import multiface_dir
from random import shuffle
from pathlib import Path


if __name__ == '__main__':
    srcDir = r'D:\paradise\stuff\simswappg\simcelebcombination'
    target_dir = r'C:\Games\MultiFaces'
    result_dir = r'D:\Developed\FaceSwapExperimental\TestResult'
    src_dirs = [str(x) for x in Path(srcDir).glob('*') if x.is_dir()]
    shuffle(src_dirs)
    for src_dir in src_dirs:
        multiface_dir(src_dir,target_dir,result_dir)