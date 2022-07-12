import msvcrt
import os


def UserCommand(keys={b'm':lambda x:x}):
    if msvcrt.kbhit():
        # print('YOU PRESSED SOMETHING')
        pKey = msvcrt.getch()
        for key,value in keys.items():
            if pKey == key:
                value()
                return True
        print(keys)
        return False
        
def open_dir(dirPath):
    os.system('start "" "{}"'.format(dirPath))