import os
import sys


def main():
    files = os.listdir('./')
    index = 0
    for filename in files:
        if ('.mp4' in filename) or ('.flv' in filename):
            os.system('ffmpeg -y -i ' + filename +
                      ' -vcodec copy -f flv ' + str(index) + '.flv')
            os.system('rm -rf ' + filename)
            index += 1
    print 'done'


if __name__ == '__main__':
    main()
