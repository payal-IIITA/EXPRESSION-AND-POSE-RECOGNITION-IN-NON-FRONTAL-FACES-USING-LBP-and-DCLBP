import numpy as np
import cv2
import glob
from sklearn.externals import joblib
import os

# This function calculates the DCLBP of img and returns it in a 1-D list
def diagonal_cris_cross_lbp(img):
    flat = []
    
    #apply the standard procedure to calculate DCLBP
    for x in range(1, len(img)-1):
        for y in range(1, len(img[0])-1):
            val = 0
            diff = img[x-1][y-1] - img[x+1][y+1]
            if diff >= 0:
                val = val + 2
            
            diff = img[x-1][y] - img[x+1][y]
            if diff >= 0:
                val = val + 8

            diff = img[x-1][y+1] - img[x+1][y-1]
            if diff >= 0:
                val = val + 32

            diff = img[x][y+1] - img[x][y-1]
            if diff >= 0:
                val = val + 128
            
            idx = int((img[x][y] + val)/2)
            
            flat.append(idx)
    
    return flat

def expand(string):
    if string=="N":
        return "Ninety"
    if string=="F":
        return "Forty Five"
    if string=="Z":
        return "Zero"
    if string=="AF":
        return "Fear"
    if string=="AN":
        return "Angry"
    if string=="DI":
        return "Disgust"
    if string=="HA":
        return "Happy"
    if string=="NE":
        return "Neutral"
    if string=="SA":
        return "Sad"
    if string=="SU":
        return "Surprise"

#obtain the relative path of the directory containing test images
script_dir = os.path.dirname(os.path.dirname(__file__))
rel_path = script_dir + "/Datasets/Test_Images"

#iterate over all the images
for image in glob.glob(rel_path + "/*.jpg") :
        file_name = image
        #read the image in grayscale
        img = cv2.imread(file_name, 0)
        #resize the image to 169x229 size
        img = cv2.resize(img, (169, 229))
        
        x = file_name.split("/")
        g = x[len(x)-1]
        #varivle b contains the name of the current image file
        a, b = g.split("\\")
        
        # calculate the DCLBP of image
        hist_angle = diagonal_cris_cross_lbp(img)
        hist_angle = np.array(hist_angle)

        #load the classifier        
        clf = joblib.load('dclbp_classifier.pkl') 
        
        #predict the answer        
        prediction = clf.predict([hist_angle])[0]

        #default value setting
        if prediction == '':
            prediction = 'AF'
            
        print("\n\n****************************************")
        #print the image name
        print("Image Name: ", b)
        #print the expression
        print("Predicted Expression = ", expand(prediction))
        print("****************************************")