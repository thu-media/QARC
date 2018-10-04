import os
import sys
import json
import commands
import re

testrange = [300, 500, 800, 1100, 1400]
WIDTH, HEIGHT = 1280, 720
FPS = 25


def eventloop(test_file):
    os.system('mkdir img')
    #os.system('mkdir img/%s' % (test_file))
    os.system('mkdir tmp_%s' % (test_file))
    os.system('mkdir tmp2_%s' % (test_file))
    os.system('mkdir img/%s' % (test_file))
    # ffmpeg -i mov/1.flv -vf fps=25 -s 1280x720 img/1.flv/%5d.png
    os.system('ffmpeg -y -t 10 -i mov/%s -vf fps=%d -s 1280x720 tmp_%s/%%d.png' %
              (test_file, FPS, test_file))
    os.system('ffmpeg -y -i tmp_%s/%%d.png -r 5 -s 160x90 img/%s/%%d.png' %
              (test_file, test_file))
    print 'done'


if __name__ == '__main__':
    for _file in os.listdir('mov/'):
        eventloop(_file)
