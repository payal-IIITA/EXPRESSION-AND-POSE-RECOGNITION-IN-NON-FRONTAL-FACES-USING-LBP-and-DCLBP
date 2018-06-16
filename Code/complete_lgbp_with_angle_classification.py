import numpy as np
import cv2
import glob
from sklearn.externals import joblib
import os

#------------- Gabor Filter Modules START----------------------------
def build_filters(ksize, theta):
    filters = []
    #ksize = 31
    
    kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    kern /= 1.5 * kern.sum()
    
    filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum
#------------- Gabor Filter Modules END----------------------------

#------------- uniform LBP Modules START---------------------------------
def is_safe(x, y, m, n) :
        if x >= 0 and y >= 0 and x < m and y < n :
            return 1
        else :
            return 0
            

def uniform_circular_lbp(img):
    #index[] contains all the possible uniform LBPs possible
    index = [255, 127, 191, 223, 239, 247, 251, 253, 254, 63, 126, 159, 207, 231, 243, 249, 252, 31, 62, 124, 143, 199, 227, 241, 248, 15, 30, 60, 120, 135, 195, 225, 240, 7, 14, 28, 56, 112, 131, 193, 224, 3, 6, 12, 24, 48, 96, 129, 192, 1, 2, 4, 8, 16, 32, 64, 128, 0]
    
    #initialize the histogram
    hist = [0] * 60
    
    #apply the standard procedure to determine LBP pixel by pixel and compute histogram 
    for x in range(0, len(img)):
        for y in range(0, len(img[0])):
            pixels = []
                       
            if is_safe(x + 1, y - 1, len(img), len(img[0])) == 1 :
                if img[x][y] > img[x + 1][y - 1] :
                    pixels.append(0)
                else :
                    pixels.append(1)
            else :
                pixels.append(0)
            
            
            if is_safe(x, y - 1, len(img), len(img[0])) == 1 :
                if img[x][y] > img[x][y - 1] :
                    pixels.append(0)
                else :
                    pixels.append(1)
            else :
                pixels.append(0)
            
            if is_safe(x - 1, y - 1, len(img), len(img[0])) == 1 :
                if img[x][y] > img[x - 1][y - 1] :
                    pixels.append(0)
                else :
                    pixels.append(1)
            else :
                pixels.append(0)
            
            
            if is_safe(x - 1, y, len(img), len(img[0])) == 1 :
                if img[x][y] > img[x - 1][y] :
                    pixels.append(0)
                else :
                    pixels.append(1)
            else :
                pixels.append(0)
            
            
            if is_safe(x - 1, y + 1, len(img), len(img[0])) == 1 :
                if img[x][y] > img[x - 1][y + 1] :
                    pixels.append(0)
                else :
                    pixels.append(1)
            else :
                pixels.append(0)
            
            
            if is_safe(x, y + 1, len(img), len(img[0])) == 1 :
                if img[x][y] > img[x][y + 1] :
                    pixels.append(0)
                else :
                    pixels.append(1)
            else :
                pixels.append(0)
            
            if is_safe(x + 1, y + 1, len(img), len(img[0])) == 1 :
                if img[x][y] > img[x + 1][y + 1] :
                    pixels.append(0)
                else :
                    pixels.append(1)
            else :
                pixels.append(0)
            
            if is_safe(x + 1, y, len(img), len(img[0])) == 1 :
                if img[x][y] > img[x + 1][y] :
                    pixels.append(0)
                else :
                    pixels.append(1)
            else :
                pixels.append(0)
                

            res = 0
            for a in range(0, len(pixels)):
                res += pixels[a] * (1 << (7 - a))
            
            flag = 0
            for i in range(0, len(index)) :
                if index[i] == res :
                    hist[i] += 1
                    flag = 1
                    break
            if flag == 0 : 
                hist[59] += 1
    return hist

#---------------- uniform LBP Modules END ---------------------------------------

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

#---------- Main Function--------------------------------------------------
'''
Convention for Angle 
FL = N (Ninety) 
S = Straight (Zero)
HL = F (Forty Five)
FR = Ignore
HR = Ignore

Convention for Expression
AF = Fear
AN = Angry
DI = Disgust
HA = Happy
NE = Neutral
SA = Sad
SU = Surprise
'''

#relative path of the directory containing test images
script_dir = os.path.dirname(os.path.dirname(__file__))
rel_path = script_dir + "/Datasets/Test_Images"

#iterate over all the images present in that directory
for image in glob.glob(rel_path + "/*.jpg") :
        file_name = image

        #read the image in gray scale        
        img = cv2.imread(file_name, 0)
        
        #resize the image (downscaling only)
        img = cv2.resize(img, (169, 229))
        
        #calculate row and column seperator used to divide the image into 81 blocks
        rows = len(img)
        cols = len(img[0])
        row_separator = rows / 9
        col_spearator = cols / 9   
        
        x = file_name.split("/")
        g = x[len(x)-1]
        #variable b contains the name of the current image file        
        a, b = g.split("\\")
        
        #calculate the LBP histogram of the complete image
        hist_angle = uniform_circular_lbp(img)
        hist_angle = [int(i) for i in hist_angle]
        
        #convert the list into a numpy array
        hist_angle = np.array(hist_angle)

        #load the classifier from the disk        
        clf_pose = joblib.load('pose_classifier.pkl')
        
        #use the loaded classifier to predict the pose of the image
        predicted_angle = clf_pose.predict([hist_angle])[0]

        #default vale in case of boundary condition
        if predicted_angle == '':
            predicted_angle = 'N'
            
        print("\n\n****************************************")
        print("Image Name: ", b)
        print("Predicted angle = ", expand(predicted_angle))
        
        hist_angle_exp = []
        
        #vary the angle from 0 to pi with pi/8 intervals
        for theta in np.arange(0, np.pi, np.pi / 8) :
            
            #apply gabor filters to the image
            filters = build_filters(31, theta)
            res1 = process(img, filters)
            #cv2.imshow ("",res1)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #create LBP histogram for each block of the image obtained after applying gabor filter
            for a in range(1, 10):
                for b in range(1, 10):
                    block = res1[row_separator * (a - 1):row_separator * a, col_spearator * (b - 1):col_spearator * b]
                    r = uniform_circular_lbp(block)
                    hist_angle_exp.extend(r)
        
        hist_angle_exp = np.array(hist_angle_exp)

        #load the classifiers for zero, ninety and forty five degrees        
        clf_exp_z = joblib.load('expression_classifier_z.pkl')
        clf_exp_f = joblib.load('expression_classifier_f.pkl')
        clf_exp_n = joblib.load('expression_classifier_n.pkl')
    
        #use the appropriate classifier to predict the expression based on the pose predicted earlier
        if predicted_angle=='Z':
            print("Predicted expression = ", expand(clf_exp_z.predict([hist_angle_exp])[0]))
        if predicted_angle=='F':
            print("Predicted expression = ", expand(clf_exp_f.predict([hist_angle_exp])[0]))
        if predicted_angle=='N':
            print("Predicted expression = ", expand(clf_exp_n.predict([hist_angle_exp])[0]))
        print("****************************************")