import os
from pathlib import Path
from random import shuffle,choice
from dir_mage import src_dir
import shutil
        
if __name__ == '__main__':
    targetDir_parent = r'D:\paradise\stuff\new\imageset'
    indir_global = r'D:\paradise\stuff\simswappg\srcs'
    destination_dir = r'D:\paradise\stuff\new\imageset2'
    # indir_global = r'D:\paradise\stuff\Essence\FS\all\Sluts'
    # targetDir_global = r'C:\Heaven\YummyBaker'
    # targetDir_global = r'C:\Heaven\YummyBaker'
    # targetDir_global = r'D:\paradise\stuff\new\imageset2\meri maa mujhse chud jati'
    # targetDir_global = 
    # targetDir_global = r'D:\paradise\stuff\new\pvd2'
    # import pdb;pdb.set_trace()
    targetDir_p_global = [str(x) for x in Path(targetDir_parent).glob('*') if x.is_dir()]#str(choice())
    shuffle(targetDir_p_global)
    outDir_global = r'C:\Games\Sacred2'
    for targetDir_global in targetDir_p_global:
        # outDir_global = r'D:\Developed\FaceSwapExperimental\TestResult'
        # outDir_global = r'C:\Games\sacred2'
        src_dir(indir_global,targetDir_global,outDir_global,True)
        target_cpy = Path(destination_dir) / Path(targetDir_global).name
        i = 1
        while target_cpy.is_dir():
            target_cpy = target_cpy.parent / (Path(targetDir_global).name + ' ' + str(i))
            i += 1
        shutil.copytree(targetDir_global, target_cpy)
        shutil.rmtree(targetDir_global)
        
    