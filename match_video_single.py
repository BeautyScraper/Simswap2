'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:00:38
Description: 
'''
from pathlib import Path
from time import sleep
from random import shuffle
from sophi import single_src
from options.test_options import TestOptions
# def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0


            
if __name__ == "__main__":
    args = TestOptions().parse()

    srcimgdir_g = Path(args.indir)
    dstvideodir_g = Path(args.target_dir)
    respath_g = Path(args.output_dir)
    imfiles = [x for x in srcimgdir_g.glob('*.jpg')]
    shuffle(imfiles)
    for imgFIles in imfiles:
        tes_dir = dstvideodir_g / (imgFIles.stem + '_video')
        # import pdb;pdb.set_trace()
        if tes_dir.is_dir():
            dstFileList = [x for x in tes_dir.glob('*.m[pk][4v]')]
            shuffle(dstFileList)
            single_src(imgFIles,dstFileList,respath_g,testsrc_times=-1,delete_target_when_done=True)
            
    # setSrc_setDst(srcimgdir_g,dstvideodir_g,respath_g)
    