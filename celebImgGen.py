import os
from pathlib import Path
from random import shuffle,choice
from dir_mage import src_dir
import shutil
        
if __name__ == '__main__':
    targetDir_parent = r'D:\Developed\FaceSwapExperimental\Pavitra'
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
    targetDir_p_global = [x for x in Path(targetDir_parent).glob('*.jpg')]#str(choice())
    shuffle(targetDir_p_global)
    outDir_global = r'D:\Developed\FaceSwapExperimental\TestResult'
    for targetDir_global in targetDir_p_global:
        # import pdb;pdb.set_trace()
        new_dir = targetDir_global.parent / 'Pavitra'
        new_dir.mkdir(exist_ok=True)
        shutil.copy(targetDir_global, new_dir)
        # outDir_global = r'D:\Developed\FaceSwapExperimental\TestResult'
        # outDir_global = r'C:\Games\sacred2'
        src_dir(indir_global,str(new_dir),outDir_global,True)
        
        # shutil.copytree(targetDir_global, target_cpy)
        targetDir_global.unlink()
        shutil.rmtree(new_dir)
        
        
    