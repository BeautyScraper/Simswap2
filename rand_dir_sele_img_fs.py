'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:00:38
Description: 
'''

from dir_mage import src_dir
from pathlib import Path
from random import shuffle

    

if __name__ == '__main__':
    indir_global = r'D:\paradise\stuff\simswappg\srcs'
    # indir_global = r'D:\paradise\stuff\Essence\FS\all\Sluts'
    # targetDir_global = r'C:\Heaven\YummyBaker'
    # targetDir_global = r'C:\Heaven\YummyBaker'
    # targetDir_global = r'D:\paradise\stuff\new\imageset2\meri maa mujhse chud jati'
    # targetDir_global = 
    # targetDir_global = r'D:\paradise\stuff\new\pvd2'
    targetDir_global = r'D:\paradise\stuff\new\imageset2'
    # outDir_global = r'D:\paradise\stuff\new\pvd\test'
    # outDir_global = r'D:\Developed\FaceSwapExperimental\TestResult'
    outDir_global = r'C:\Games\sacred2'
    tgd = [xdir for xdir in Path(targetDir_global).glob('*') if xdir.is_dir()] 
    import pdb;pdb.set_trace()
    shuffle(tgd)
    for td in tgd:
        src_dir(indir_global,str(td),outDir_global,True)
    
