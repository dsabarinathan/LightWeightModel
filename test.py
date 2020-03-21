# -*- coding: utf-8 -*-

from keras.models import load_model

from numpy import newaxis
import numpy as np
import cv2
import os
import argparse
import time
from scipy.io.matlab.mio import savemat
from coord import CoordinateChannel2D
from model_utils import sum_squared_error, ssim

def generate_output(read_img_test,model):
    results=np.zeros((read_img_test.shape[0],read_img_test.shape[1],31))
    for r in range(0,read_img_test.shape[0],64):
        for c in range(0,read_img_test.shape[1],64):
            sample= read_img_test[r:r+64, c:c+64,:]
            if (sample.shape[0]==sample.shape[1]):
                sample = sample[newaxis,:,:,:].astype(np.float32)
                prediction= model.predict(sample)
            else:
                padd = np.zeros((64,64,3)).astype(np.float32)
                padd[:sample.shape[0],:sample.shape[1],:]=sample
                padd = padd[newaxis,:,:,:]
                prediction = model.predict(padd)
                prediction = prediction[:,:sample.shape[0],:sample.shape[1],:]
            results[r:r+64, c:c+64]=prediction[0,:,:,:]
    return results


if __name__ == '__main__':
         
    parser = argparse.ArgumentParser(description='eye-net')
    parser.add_argument("--testImagePath", type=str,dest="test_path" ,help="Path of test Images",default='./test/',action="store")
    args = parser.parse_args()
    

    model = load_model('./model/model-221.74-val_mse-0.0004--val_ssim--0.9827.hdf5',custom_objects={'sum_squared_error':sum_squared_error,'ssim':ssim,'CoordinateChannel2D':CoordinateChannel2D})

    output_path = './output_file/'
    if not os.path.exists(output_path):
       os.makedirs(output_path)
    
    testImagePath = args.test_path
    
    fileName = os.listdir(testImagePath)
    
    for i in range(len(fileName)):
        
        start_time = time.time()

        img = cv2.imread(testImagePath+fileName[i])
   
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
        results = generate_output(img,model)
        end_time = time.time()
    
        print('predicted time', end_time-start_time)
        print(fileName[i].split('clean')[0][0:-1]+'.mat')
        savemat(output_path+fileName[i].split('clean')[0][0:-1]+'.mat', {'cube': results})
    
        print(i)
        
    print("output files saved in "+output_path)
