import os
import numpy as np
#from PIL import Image
import cv2
import h5py
# implmenation of vmaf neural network
# in 640x360
# out vmaf future score
# 5 + 4 + 3 + 2 + 1
INPUT_W = 64
INPUT_H = 36
INPUT_D = 3
INPUT_SEQ = 25
import time

# 300,45.6419748364
# 500,60.3927594181
# 800,72.948687536
# 1100,81.1788565049
# 1400,86.2749310139


def load_y(filename, index):
    _file = filename.replace('.y4m', '')
    #_index = index + 1
    _file += '_' + str(index) + '.yuv-480.log'
    try:
        _reader = open('./csv/' + _file)
        _array = []
        for _line in _reader:
            _array.append(float(_line.split(',')[1]) / 100.0)
        return np.array(_array)
    except:
        #print 'error'
        return None


def load_image(filename):
    img = cv2.imread(filename)
    return img

# 5+4+3+2+1


def saveh5f(filename, x, y):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('X', data=x)
    h5f.create_dataset('Y', data=y)
    h5f.close()
    print 'save done'


def event_loop():
    _dirs = os.listdir('img/')
    _x_array, _y_array = [], []
    for _dir in _dirs:
        print _dir
        _files = os.listdir('img/' + _dir + '/')
        _files.sort()
        x = np.zeros([INPUT_SEQ, INPUT_H, INPUT_W, INPUT_D])
        _index = 0
        for _file in _files:
            x = np.roll(x, -1, axis=1)
            _img = load_image('img/' + _dir + '/' + _file)
            x[-1, :, :, :] = _img
            _index += 1
            if _index % (INPUT_SEQ / 5) == 0:
                #print _index
                y = load_y(_dir, _index / (INPUT_SEQ / 5))
                if y is not None:
                    _x_array.append(x)
                    _y_array.append(y)
    return np.array(_x_array), np.array(_y_array)
    #y_ = np.array([OUTPUT_DIM])


def main():
    x, y = event_loop()
    saveh5f('vmaf.h5', x, y)


if __name__ == '__main__':
    main()

