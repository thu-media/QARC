import os

kernel_array = [8,16,32,128,256]
lr_array = [1e-4]
for k in kernel_array:
    #for d in dense_array:
    for l in lr_array:
        os.system('python vqpn.py ' + str(k) + ' ' + str(k) + ' ' + str(l))
