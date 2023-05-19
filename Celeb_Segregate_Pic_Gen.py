import os
from pathlib import Path
from random import shuffle,choice
from dir_mage import src_dir,single_src_dir_dst
import shutil
        
if __name__ == '__main__':
    targetDir_parent = r'D:\paradise\stuff\simswappg\custom_selec'
    indir_global = r'D:\paradise\stuff\simswappg\srcs\celeb+known'
    destination_dir = r'D:\Developed\FaceSwapExperimental\TestResult'
    # indir_global = r'D:\paradise\stuff\Essence\FS\all\Sluts'
    # targetDir_global = r'C:\Heaven\YummyBaker'
    # targetDir_global = r'C:\Heaven\YummyBaker'
    # targetDir_global = r'D:\paradise\stuff\new\imageset2\meri maa mujhse chud jati'
    # targetDir_global = 
    # targetDir_global = r'D:\paradise\stuff\new\pvd2'
    # import pdb;pdb.set_trace()
    
    # targetDir_p_global = [str(x) for x in Path(targetDir_parent).glob('*') if x.is_dir()]#str(choice())
    # shuffle(targetDir_p_global)
    outDir_global = r'D:\Developed\FaceSwapExperimental\TestResult'
    for src_img in Path(indir_global).glob('*.jpg'):
        dir_to_trav = Path(targetDir_parent) / src_img.stem
        if not dir_to_trav.is_dir():
            continue
        print(dir_to_trav)
        # targetDir_p_global = [x for x in dir_to_trav.glob('*.jpg')]#str(choice())
        # for targetDir_global in targetDir_p_global:
        # import pdb;pdb.set_trace()
        # new_dir = targetDir_global.parent / 'Pavitra'
        # new_dir.mkdir(exist_ok=True)
        # shutil.copy(targetDir_global, new_dir)
        # outDir_global = r'D:\Developed\FaceSwapExperimental\TestResult'
        # outDir_global = r'C:\Games\sacred2'
        single_src_dir_dst(src_img,str(dir_to_trav),outDir_global)
        
        # shutil.copytree(targetDir_global, target_cpy)
        # targetDir_global.unlink()
        shutil.rmtree(dir_to_trav)
    
        
    