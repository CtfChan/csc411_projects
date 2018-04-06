
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from PIL import Image


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/
    '''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def grab_images(file_location, gender):
    '''Saves uncropped and cropped versions of images from file_location
    Arguments
    file_location -- 
    gender --
    '''
    i = 0
    name = ''
    for line in open(file_location):
        curr_line = line.split()
        i = i if (name == (curr_line[0] + curr_line[1])) else 0
        name = curr_line[0] + curr_line[1]
        filename = name + str(i) + '.jpg'  #curr_line[4].split('.')[-1]
        
        #save uncropped
        try:
            coords = curr_line[-2].split(',')
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]),int(coords[2]), int(coords[3])
            timeout(testfile.retrieve, (curr_line[4], 'uncropped/'+filename), {}, 30)
            if not os.path.isfile('uncropped/'+filename):
                continue
            print "Downloaded: " + filename
            i+=1
        except Exception:
            continue

        try:
            image = imread("uncropped/" + filename)
            cropped = image[y1:y2, x1:x2]
            grayed = rgb2gray(cropped)
            resized = imresize(grayed, (32, 32))

            if gender == 'female':
                imsave("cropped/female/" + filename, resized)
                #img.save("cropped/female/" + filename,JPEG)
                print "Finished processing: ", filename
            else:
                imsave("cropped/male/" + filename, resized)
                #img.save("cropped/male/" + filename,JPEG)
                print "Finished processing: ", filename
        except Exception:
            continue
        

## Call functions here
#create relevant directories
if not os.path.isdir('uncropped'):
    os.makedirs('uncropped')
if not os.path.isdir('cropped'):
    os.makedirs('cropped')
if not os.path.isdir('cropped/male'):
    os.makedirs('cropped/male')
if not os.path.isdir('cropped/female'):
    os.makedirs('cropped/female')

testfile = urllib.URLopener()   

file_location = "facescrub_actresses.txt"
grab_images(file_location, 'female')

file_location = "facescrub_actors.txt"
grab_images(file_location, 'male')
 


