from keras.models import load_model
from numpy import newaxis
import numpy as np
import cv2
import os
import argparse
import time
import pandas as pd
from modelzoo import modifiedEfficientNet

if __name__ == '__main__':
         
    parser = argparse.ArgumentParser(description='track_2_Couger AI')
    parser.add_argument("--testSARImagePath", type=str,dest="test_SAR_path" ,help="Path of test Images",default='./test/',action="store")
    parser.add_argument("--testEOImagePath", type=str,dest="test_EO_path" ,help="Path of test Images",default='./test/',action="store")

    args = parser.parse_args()
    
    model = modifiedEfficientNet(input_shape=(64,64,3),learningRate=0.00001)
    model = load_model('./model/track_2-val_total-loss--0.1274---validation_acc--0.9779.hdf5')
    
    SARImagePath = args.test_SAR_path

    EOImagePath = args.test_EO_path
    
    imageName = os.listdir(SARImagePath)
    IMG_WIDTH = IMG_HEIGHT =64
    features = []
    predictedLabel = []
    image_name0= []
    
    image_batch0 = np.zeros((1,IMG_WIDTH,IMG_HEIGHT, 3))
    
    for i in range(len(imageName)):
        
        sample0 = SARImagePath+imageName[i]
    
        img=cv2.imread(sample0)
    
        colorImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(colorImg, (IMG_WIDTH,IMG_HEIGHT))
        
        image_resizedGray=cv2.cvtColor(image_resized,cv2.COLOR_BGR2GRAY) 
    
        subName = imageName[i].replace("SAR","EO")
                
        imgEO=cv2.imread(EOImagePath+subName)
        image_resized0 = cv2.resize(imgEO, (IMG_WIDTH,IMG_HEIGHT))
                
        image_batch0[0,:,:,0] = image_resizedGray
        image_batch0[0,:,:,1] = image_resized0[:,:,0]
        image_batch0[0,:,:,2] = image_resized0[:,:,1]
                
        start = time.time()
        output = model.predict(image_batch0)
        
        print("prediction time", time.time()-start)
    
        predictedLabel.append(np.argmax(output[0]))
        image_name0.append(imageName[i].split("_")[1][0:-4])

    

valid_result = pd.read_csv("results.csv")
valid_result["image_id"] = image_name0
valid_result["class_id"] = predictedLabel
valid_result.to_csv("D:/sabari/CVPR2022/SAR 2022 Classification/results6000_track2_regnet.csv")