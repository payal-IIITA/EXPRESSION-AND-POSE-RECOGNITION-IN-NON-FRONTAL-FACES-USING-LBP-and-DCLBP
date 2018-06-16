from sklearn.svm import SVC
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
import os

#relative path of training file
script_dir = os.path.dirname(__file__)
rel_path = script_dir + "/train_expression.txt"
train_file = rel_path

#X[] contains the histograms of the images
X = []
#Y[] contains the corresponding expressions
Y = []

#iterate over the file line by line
with open(train_file, "r") as ins:
    for line in ins:
    	#extract file name, histogram and expression
        file_name, histogram, expression = line.split(':')
        
        histogram = histogram.strip()
        expression = expression.strip()
        h = histogram.split()
        h = [int(i) for i in h]    
        #add the histogram to X[]
        X.append(h)
        #add the expression to Y[]
        Y.append(expression)
        
#convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

#build the classifier
clf = OneVsRestClassifier(SVC(kernel='linear'))
clf.fit(X, Y)
print("Classifier built!")

#save the classifier on the disk
joblib.dump(clf, 'dclbp_classifier.pkl')