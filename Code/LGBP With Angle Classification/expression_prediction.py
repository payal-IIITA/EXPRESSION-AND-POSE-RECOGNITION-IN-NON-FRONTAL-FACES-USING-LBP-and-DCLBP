import numpy as np
from sklearn.externals import joblib
import os

#function to update the confusion matrix
def update_confusion_matrix(arr, p, a):
    if p=="HA":
        if a=="HA":
            arr[0][0] = arr[0][0] + 1
        if a=="SA":
            arr[0][1] = arr[0][1] + 1
        if a=="SU":
            arr[0][2] = arr[0][2] + 1
        if a=="AN":
            arr[0][3] = arr[0][3] + 1
        if a=="DI":
            arr[0][4] = arr[0][4] + 1
        if a=="AF":
            arr[0][5] = arr[0][5] + 1

    if p=="SA":
        if a=="HA":
            arr[1][0] = arr[1][0] + 1
        if a=="SA":
            arr[1][1] = arr[1][1] + 1
        if a=="SU":
            arr[1][2] = arr[1][2] + 1
        if a=="AN":
            arr[1][3] = arr[1][3] + 1
        if a=="DI":
            arr[1][4] = arr[1][4] + 1
        if a=="AF":
            arr[1][5] = arr[1][5] + 1
    
    if p=="SU":
        if a=="HA":
            arr[2][0] = arr[2][0] + 1
        if a=="SA":
            arr[2][1] = arr[2][1] + 1
        if a=="SU":
            arr[2][2] = arr[2][2] + 1
        if a=="AN":
            arr[2][3] = arr[2][3] + 1
        if a=="DI":
            arr[2][4] = arr[2][4] + 1
        if a=="AF":
            arr[2][5] = arr[2][5] + 1
    
    if p=="AN":
        if a=="HA":
            arr[3][0] = arr[3][0] + 1
        if a=="SA":
            arr[3][1] = arr[3][1] + 1
        if a=="SU":
            arr[3][2] = arr[3][2] + 1
        if a=="AN":
            arr[3][3] = arr[3][3] + 1
        if a=="DI":
            arr[3][4] = arr[3][4] + 1
        if a=="AF":
            arr[3][5] = arr[3][5] + 1
    
    if p=="DI":
        if a=="HA":
            arr[4][0] = arr[4][0] + 1
        if a=="SA":
            arr[4][1] = arr[4][1] + 1
        if a=="SU":
            arr[4][2] = arr[4][2] + 1
        if a=="AN":
            arr[4][3] = arr[4][3] + 1
        if a=="DI":
            arr[4][4] = arr[4][4] + 1
        if a=="AF":
            arr[4][5] = arr[4][5] + 1
            
    if p=="AF":
        if a=="HA":
            arr[5][0] = arr[5][0] + 1
        if a=="SA":
            arr[5][1] = arr[5][1] + 1
        if a=="SU":
            arr[5][2] = arr[5][2] + 1
        if a=="AN":
            arr[5][3] = arr[5][3] + 1
        if a=="DI":
            arr[5][4] = arr[5][4] + 1
        if a=="AF":
            arr[5][5] = arr[5][5] + 1

#load the three classifiers from the disk            
clf_z = joblib.load('expression_classifier_z.pkl')
clf_f = joblib.load('expression_classifier_f.pkl')
clf_n = joblib.load('expression_classifier_n.pkl')

#test[] contains histograms of the images to be tested
test = []
#actual[] contains the actual expression of the image
actual = []
#angles[] contains the angles predicted by the angle_prediction.py file
angles = []
#actual_angles[] contains the actual angles of the images
actual_angles = []

#relative path of test file
script_dir = os.path.dirname(__file__)
rel_path = script_dir + "/test_expression.txt"
test_file = rel_path

#iterate over the file line by line
with open(test_file, "r") as ins:
    for line in ins:
        file_name, histogram, angle, expression = line.split(':')
        
        histogram = histogram.strip()
        h = histogram.split()
        h = [int(i) for i in h]
        test.append(h)
        
        angle = angle.strip()
        actual_angles.append(angle)
        
        expression = expression.strip()
        actual.append(expression)

test = np.array(test)
actual = np.array(actual)
actual_angles = np.array(actual_angles)

#relative path of the file containing predicted angles from the pose_prediction.py file
script_dir = os.path.dirname(__file__)
rel_path = script_dir + "/predicted_angles.txt"
predicted_angles_file = rel_path

#read the file line by line and store the predicted angles in angles[]
with open(predicted_angles_file, "r") as ins:
    for line in ins:
        line = line.strip()
        angles.append(line)

#convert te angles list to numpy array
angles = np.array(angles)

#number of correct and incorrect predictions
correct = 0
incorrect = 0

#number of correct predictions at zero, forty five and ninety degrees
correct_z = 0
correct_f = 0
correct_n = 0

#create 2D list for confusion matrix
arr = [[0]*6 for _ in range(6)]
srn = 1

#iterate over the test[]
for i in range(0, len(test)):
    #use the appropriate classifier to predict the expression according to the predicted angles
    if angles[i]=='Z':
        prediction = clf_z.predict([test[i]])[0]
        #update various counters according to the obtained prediction values
        if prediction==actual[i]:
            correct = correct + 1
            if actual_angles[i]=='Z':
                correct_z = correct_z + 1
            if actual_angles[i]=='F':
                correct_f = correct_f + 1
            if actual_angles[i]=='N':
                correct_n = correct_n + 1
        else:
            incorrect = incorrect + 1
        #update the confusion matrix
        update_confusion_matrix(arr, prediction, actual[i])
        #print the result
        print(srn, ". Prediction = ", prediction, "actual = ", actual[i])
        
    if angles[i]=='F':
        prediction = clf_f.predict([test[i]])[0]
        if prediction==actual[i]:
            correct = correct + 1
            if actual_angles[i]=='Z':
                correct_z = correct_z + 1
            if actual_angles[i]=='F':
                correct_f = correct_f + 1
            if actual_angles[i]=='N':
                correct_n = correct_n + 1
        else:
            incorrect = incorrect + 1
        update_confusion_matrix(arr, prediction, actual[i])
        print(srn, ". Prediction = ", prediction, "actual = ", actual[i])

    if angles[i]=='N':
        prediction = clf_n.predict([test[i]])[0]
        if prediction==actual[i]:
            correct = correct + 1
            if actual_angles[i]=='Z':
                correct_z = correct_z + 1
            if actual_angles[i]=='F':
                correct_f = correct_f + 1
            if actual_angles[i]=='N':
                correct_n = correct_n + 1
        else:
            incorrect = incorrect + 1
        update_confusion_matrix(arr, prediction, actual[i])
        print(srn, ". Prediction = ", prediction, "actual = ", actual[i])
    
    srn = srn+1

#print the total correct, incorrect predictions
print("\nCorrect = ", correct, "Incorrect = ", incorrect, "Total = ", correct+incorrect)
#print number of correct predictions anglewise
print("Correct at Z, F, N: ", correct_z, "  ", correct_f, "  ", correct_n)
#print the confusion matrix
for i in range(6):
        print(arr[i])