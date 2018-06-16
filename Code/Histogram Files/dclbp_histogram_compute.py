import cv2
import glob
import os

def diagonal_cris_cross_lbp(img):
    flat = []
    
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
    
fo1 = open("dclbp_histogram.txt", "w+")
cnt=1

#absolute path of the current script
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#relative path of the dataset directory
rel_path = script_dir + "/Datasets/Test_Images"

for image in glob.glob(rel_path + "/*.jpg") :
        file_name = image
        img = cv2.imread(file_name, 0)
        
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
        hist_angle.append(diagonal_cris_cross_lbp(img))
        
        for i in range(0, len(hist_angle[0])):
            fo1.write('{:d} '.format(hist_angle[0][i]))
        
        fo1.write(": ")
        fo1.write(expr)
        fo1.write("\n")
        
        print(cnt)
        cnt = cnt+1
                
fo1.close()
