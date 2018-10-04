import os
import commands
import re

(status, output) = commands.getstatusoutput('ffmpeg -i mov/0.flv')
# print 'output:',output
lines = output.split('\n')
for line in lines:
    if 'Stream #0:0' in line:
        matchObj = re.match(r'.* (.*)x(.*) \[.*', line, re.M | re.I)
        if matchObj:
            width = matchObj.group(1)
            height = matchObj.group(2)
            #return width, height
        else:
            print "No match!!"
            #return 0, 0
