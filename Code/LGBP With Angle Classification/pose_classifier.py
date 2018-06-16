from sklearn.svm import SVC
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
import os 

#relative path of training file
script_dir = os.path.dirname(__file__)
rel_path = script_dir + "/train_angle.txt"

train_file = rel_path

#X[] will contain all the histograms of images
X = []
#Y[] will contain all the corresponding angles
Y = []

#iterate over the file line by line
with open(train_file, "r") as ins:
    for line in ins:
    	#extract the file name, histogram and angle from each line
        file_name, histogram, angle = line.split(':')
        
        histogram = histogram.strip()
        h = histogram.split()
        h = [int(i) for i in h]
        #add the histogram to the X[]
        X.append(h)
        
        angle = angle.strip() 
        #add the angle to Y[]
        Y.append(angle)

#convert the lists to numpy arrays
X = np.array(X)
Y = np.array(Y)
'''
The objective of a Linear SVC (Support Vector Classifier) is to fit to the data you provide, returning a "best fit" 
hyperplane that divides, or categorizes, your data.
'''
#build the classifier
clf = OneVsRestClassifier(SVC(kernel='linear'))
clf.fit(X, Y)

#Store the classifier on disk in current directory
joblib.dump(clf, 'pose_classifier.pkl')