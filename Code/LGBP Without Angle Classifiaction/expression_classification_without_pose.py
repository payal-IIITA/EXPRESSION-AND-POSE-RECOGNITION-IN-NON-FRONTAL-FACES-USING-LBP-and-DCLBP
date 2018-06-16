from sklearn.svm import SVC
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
import os

#relative path of training file
script_dir = os.path.dirname(__file__)
rel_path = script_dir + "/train_expression.txt"

train_file = rel_path

#X[] will contain the histograms of all the images
X = []
#Y[] will contain the corresponding expressions 
Y = []

#read the file line by line
with open(train_file, "r") as ins:
    for line in ins:
    	#extract file_name, histogram, angle and epression from the line
        file_name, histogram, angle, expression = line.split(':')
        
        histogram = histogram.strip()
        h = histogram.split()
        #convert string into integers
        h = [int(i) for i in h]
        #add the h[] to X[]
        X.append(h)

        expression = expression.strip() 
        #add the expression to Y[]       
        Y.append(expression)
        
#convert both the lists to numpy arrays
X = np.array(X)
Y = np.array(Y)

#build the classifier
clf = OneVsRestClassifier(SVC(kernel='linear'))
clf.fit(X, Y)

#store the constructed classifier on the disk
joblib.dump(clf, 'expression_classifier_without_angle.pkl')