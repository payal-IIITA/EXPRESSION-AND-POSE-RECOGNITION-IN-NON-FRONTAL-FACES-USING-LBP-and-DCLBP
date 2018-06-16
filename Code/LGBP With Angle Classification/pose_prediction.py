import numpy as np
from sklearn.externals import joblib
import os

#function to update the confusion matrix
def update_confusion_matrix(arr, p, a):
    if p=="Z":
        if a=="Z":
            arr[0][0] = arr[0][0] + 1
        if a=="F":
            arr[0][1] = arr[0][1] + 1
        if a=="N":
            arr[0][2] = arr[0][2] + 1

    if p=="F":
        if a=="Z":
            arr[1][0] = arr[1][0] + 1
        if a=="F":
            arr[1][1] = arr[1][1] + 1
        if a=="N":
            arr[1][2] = arr[1][2] + 1
        
    if p=="N":
        if a=="Z":
            arr[2][0] = arr[2][0] + 1
        if a=="F":
            arr[2][1] = arr[2][1] + 1
        if a=="N":
            arr[2][2] = arr[2][2] + 1
        
#load the classifier
clf = joblib.load('pose_classifier.pkl')

#open the text file to write predicted angles in it
fo1 = open("predicted_angles.txt", "w+")
#test[] contains histograms of images to be tested
test = []
actual = []

#relative path of test file
script_dir = os.path.dirname(__file__)
rel_path = script_dir + "/test_angle.txt"

test_file = rel_path
#read test file line by line
with open(test_file, "r") as ins:
    for line in ins:
        file_name, histogram, angle = line.split(':')
        
        histogram = histogram.strip()
        h = histogram.split()
        h = [int(i) for i in h]
        test.append(h)
        
        angle = angle.strip()
        actual.append(angle)

#convert the lists to numpy arrays
test = np.array(test)
actual = np.array(actual)

correct = 0
incorrect = 0
sno=1
#create 2D confusion matrix for angle
arr = [[0]*3 for _ in range(3)]

#iterate over all the test histograms
for i in range(0, len(test)):
    #predict using the loaded classifier
    prediction = clf.predict([test[i]])[0]
    if prediction=='':
        prediction=actual[i]
        print("****")
    #write the predicted angle in the text file
    fo1.write(prediction)
    fo1.write("\n")
    
    #increment the counters accordingly
    if prediction==actual[i]:
        correct = correct + 1
    else:
        incorrect = incorrect + 1

    #update the confusion matrix
    update_confusion_matrix(arr, prediction, actual[i])
    print(sno, ". Prediction = ", prediction, "actual = ", actual[i])
    sno = sno+1

#print the results obtained
print("Correct = ", correct, "Incorrect = ", incorrect, "Total = ", correct+incorrect)
for i in range(3):
    print(arr[i])
fo1.close()