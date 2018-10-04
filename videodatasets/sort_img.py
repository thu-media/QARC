import os

for p in os.listdir('./mov/'):
    os.system('mkdir img/' + p + '/')

for p in os.listdir('./img/'):
    _sp = p.split('_')
    if len(_sp) > 1:
        os.system('mv ./img/' + p + ' ./img/' + _sp[0] + '/')

print 'done'
