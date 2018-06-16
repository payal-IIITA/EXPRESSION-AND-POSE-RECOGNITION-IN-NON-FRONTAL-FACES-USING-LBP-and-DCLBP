import numpy as np
import cv2
import glob
import os

#------------- Gabor Filter Modules START----------------------------
def build_filters(ksize, theta):
    filters = []
    #ksize = 31
    
    kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F) #returns gabor filters coefficients
    
    kern /= 1.5 * kern.sum()
    #print(kern)
    filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


#------------- Gabor Filter Modules END----------------------------

#------------- LBPu2 Modules START---------------------------------

def is_safe(x, y, m, n) :
        if x >= 0 and y >= 0 and x < m and y < n :
            return 1
        else :
            return 0
            

def uniform_circular_lbp(img):
    index = [255, 127, 191, 223, 239, 247, 251, 253, 254, 63, 126, 159, 207, 231, 243, 249, 252, 31, 62, 124, 143, 199, 227, 241, 248, 15, 30, 60, 120, 135, 195, 225, 240, 7, 14, 28, 56, 112, 131, 193, 224, 3, 6, 12, 24, 48, 96, 129, 192, 1, 2, 4, 8, 16, 32, 64, 128, 0]
    
    hist = {}
    for h in range(0, 60):  # initialize the histogram
        hist[h] = 0
        
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
            #print(hist)
    return hist

#----------------LBPu2 Modules END ---------------------------------------

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

fo1 = open("train_angle.txt", "w+")
fo2 = open("train_expression.txt", "w+")

#absolute path of the current script
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#relative path of the dataset directory
rel_path = script_dir + "/Datasets/KDEF_All_In_One"
'''
The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell,
'''
for image in glob.glob(rel_path + "/*.jpg") :
        file_name = image
        img = cv2.imread(file_name, 0)
        
        rows = len(img)
        cols = len(img[0])
        row_separator = rows / 9
        col_spearator = cols / 9   
        
        x = file_name.split("/")
        g = x[len(x)-1]
        a, b = g.split("\\")
    
        ang = ''
        angle = ''
        expr = b[4:6]
        
        if len(b) == 11 :
            ang = b[6:7]
        else :
            ang = b[6:8]
            
        if ang == 'S' :
            angle = 'Z'
        elif ang == 'FL' :
            angle = 'N'
        elif ang == 'HL' :
            angle = 'F'
        
        
        fo1.write(format(b))
        fo1.write(format(" : "))
        
        hist_angle = []
        #returns a dictionary of histogram values acc to indices in uclbp func
        hist_angle.append(uniform_circular_lbp(img)) 
        #print("histangle:",hist_angle)
        for i in range(0, 60) :
            fo1.write('{:d} '.format(hist_angle[0][i]))
        
        fo1.write(": ")
        fo1.write(angle)
        fo1.write("\n")
        
        
        
        fo2.write(format(b))
        fo2.write(format(" : "))
        
        for theta in np.arange(0, np.pi, np.pi / 8) : 
                filters = build_filters(31, theta)
                res1 = process(img, filters)
                hist_angle_exp = []
                for a in range(1, 10):
                    for b in range(1, 10):
                        block = res1[row_separator * (a - 1):row_separator * a, col_spearator * (b - 1):col_spearator * b]
                        hist_angle_exp.append(uniform_circular_lbp(block))
                #print("histangleexp:",hist_angle)
    
                for i in range(0, 81) :
                    for j in range(0, 60) :
                        fo2.write('{:d} '.format(hist_angle_exp[i][j]))
                        
        fo2.write(": ")
        fo2.write(angle)
        fo2.write(" : ")
        fo2.write(expr)
        fo2.write("\n")
                
fo1.close()
fo2.close()
