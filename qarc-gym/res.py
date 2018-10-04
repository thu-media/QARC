import os
kernel_array = [8,16,32,128,256]

_writer = open('res.txt','w')
for p in kernel_array:
    _file = open('./best/' + str(p) + '_' + str(p) + '_0.0001.txt','r')
    for _line in _file:
        _writer.write(str(p) + ',' + str(_line) + '\n')
    _file.close()
_writer.close()