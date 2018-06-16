import numpy as np
from sklearn.externals import joblib
import os
#hc,sac,suc,anc,dic,afc

#function to creae the confusion matrix
def update_confusion_matrix(arr, p, a):
    global hc,sac,suc,anc,dic,afc
    if p=="HA":
        if a=="HA":
            arr[0][0] = arr[0][0] + 1
            
        if a=="SA":
            arr[0][1] = arr[0][1] + 1
            hc=hc+1
        if a=="SU":
            arr[0][2] = arr[0][2] + 1
            hc=hc+1
        if a=="AN":
            arr[0][3] = arr[0][3] + 1
            hc=hc+1
        if a=="DI":
            arr[0][4] = arr[0][4] + 1
            hc=hc+1
        if a=="AF":
            arr[0][5] = arr[0][5] + 1
            hc=hc+1

    if p=="SA":
        if a=="HA":
            arr[1][0] = arr[1][0] + 1
            sac=sac+1
        if a=="SA":
            arr[1][1] = arr[1][1] + 1
            
        if a=="SU":
            arr[1][2] = arr[1][2] + 1
            sac=sac+1
        if a=="AN":
            arr[1][3] = arr[1][3] + 1
            sac=sac+1
        if a=="DI":
            arr[1][4] = arr[1][4] + 1
            sac=sac+1
        if a=="AF":
            arr[1][5] = arr[1][5] + 1
            sac=sac+1
    
    if p=="SU":
        if a=="HA":
            arr[2][0] = arr[2][0] + 1
            suc=suc+1
        if a=="SA":
            arr[2][1] = arr[2][1] + 1
            suc=suc+1
        if a=="SU":
            arr[2][2] = arr[2][2] + 1
            
        if a=="AN":
            arr[2][3] = arr[2][3] + 1
            suc=suc+1
        if a=="DI":
            arr[2][4] = arr[2][4] + 1
            suc=suc+1
        if a=="AF":
            arr[2][5] = arr[2][5] + 1
            suc=suc+1
    
    if p=="AN":
        if a=="HA":
            arr[3][0] = arr[3][0] + 1
            anc=anc+1
        if a=="SA":
            arr[3][1] = arr[3][1] + 1
            anc=anc+1
        if a=="SU":
            arr[3][2] = arr[3][2] + 1
            anc=anc+1
        if a=="AN":
            arr[3][3] = arr[3][3] + 1
            
        if a=="DI":
            arr[3][4] = arr[3][4] + 1
            anc=anc+1
        if a=="AF":
            arr[3][5] = arr[3][5] + 1
            anc=anc+1
    
    if p=="DI":
        if a=="HA":
            arr[4][0] = arr[4][0] + 1
            dic=dic+1
        if a=="SA":
            arr[4][1] = arr[4][1] + 1
            dic=dic+1
        if a=="SU":
            arr[4][2] = arr[4][2] + 1
            dic=dic+1
        if a=="AN":
            arr[4][3] = arr[4][3] + 1
            dic=dic+1
        if a=="DI":
            arr[4][4] = arr[4][4] + 1
            
        if a=="AF":
            arr[4][5] = arr[4][5] + 1
            dic=dic+1
            
    if p=="AF":
        if a=="HA":
            arr[5][0] = arr[5][0] + 1
            afc=afc+1
        if a=="SA":
            arr[5][1] = arr[5][1] + 1
            afc=afc+1
        if a=="SU":
            arr[5][2] = arr[5][2] + 1
            afc=afc+1
        if a=="AN":
            arr[5][3] = arr[5][3] + 1
            afc=afc+1
        if a=="DI":
            arr[5][4] = arr[5][4] + 1
            afc=afc+1
        if a=="AF":
            arr[5][5] = arr[5][5] + 1
            

#load the classifier from the disk
hc=0
sac=0
suc=0
anc=0
dic=0
afc=0
clf = joblib.load('expression_classifier_without_angle.pkl')

#test[] will contain the histograms of images to be tested
test = []
#actual[] will contain the actual expressions of the corresponding histograms
actual = []

#relative path of test file
script_dir = os.path.dirname(__file__)
rel_path = script_dir + "/Histogram_AngleExpresion.txt"

test_file = rel_path
#read the file line by line
with open(test_file, "r") as ins:
    for line in ins:
        #extract file name, histogram, angle and expression from each line
        file_name, histogram, angle, expression = line.split(':')
        
        histogram = histogram.strip()
        h = histogram.split()
        #convert string to integers
        h = [int(i) for i in h]
        #add the histogram to test[]
        test.append(h)
        
        expression = expression.strip()
        #add the expression to actual[]
        actual.append(expression)

#convert the lists to numpy arrays
test = np.array(test)
actual = np.array(actual)

#variables used to calculate number of correct and incorrect predictions
correct = 0
incorrect = 0

#create a 2D list for confusion matrix
arr = [[0]*6 for _ in range(6)]
srn = 1

#iterate over all the histograms
for i in range(0, len(test)):
    
    #predict using the loaded classifier
    prediction = clf.predict([test[i]])[0]
    
    #increase the corresponding count depending on the result being correct or incorrect
    if prediction==actual[i]:
        correct = correct + 1
    else:
        incorrect = incorrect + 1

    #update the confusion matrix
    update_confusion_matrix(arr, prediction, actual[i])
    
    #print the result
    print(srn, ". Prediction = ", prediction, "actual = ", actual[i])
            
    srn = srn+1

#print the number of correct and incorrect predictions and the confusion matrix
print("\nCorrect = ", correct, "Incorrect = ", incorrect, "Total = ", correct+incorrect)
print(hc,sac,suc,anc,dic,afc,sep=" ")
for i in range(6):
        print(arr[i])