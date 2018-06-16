from sklearn.svm import SVC
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
import os

#relative path of training file
script_dir = os.path.dirname(__file__)
rel_path = script_dir + "/train_expression.txt"
train_file = rel_path

#X_z[] contains histograms of images clicked from angle zero(Z)
X_z = []
#Y_z[] contains the corresponding expression of the image
Y_z = []

#X_f[] contains histograms of images clicked from angle forty five(F)
X_f = []
#Y_f[] contains the corresponding expression of the image
Y_f = []

#X_n[] contains histograms of images clicked from angle Ninety(N)
X_n = []
#Y_n[] contains the corresponding expression of the image
Y_n = []

#iterate over the training file line by line
with open(train_file, "r") as ins:
    for line in ins:
        #extract the file name, histogram, angle, and expression from each line
        file_name, histogram, angle, expression = line.split(':')
        
        histogram = histogram.strip()
        angle = angle.strip()
        expression = expression.strip()
        h = histogram.split()
        h = [int(i) for i in h]
    
        #add the histogram and expression to the corresponding X[] according to the angle of the image
        if angle=='Z':
            X_z.append(h)
            Y_z.append(expression)
        if angle=='F':
            X_f.append(h)
            Y_f.append(expression)
        if angle=='N':
            X_n.append(h)
            Y_n.append(expression)

#convert the lists to numpy arrays
X_z = np.array(X_z)
Y_z = np.array(Y_z)

X_f = np.array(X_f)
Y_f = np.array(Y_f)

X_n = np.array(X_n)
Y_n = np.array(Y_n)

#create 3 classifiers each for Zero, Forty Five and Ninety degrees
clf_z = OneVsRestClassifier(SVC(kernel='linear'))
clf_z.fit(X_z, Y_z)

clf_f = OneVsRestClassifier(SVC(kernel='linear'))
clf_f.fit(X_f, Y_f)

clf_n = OneVsRestClassifier(SVC(kernel='linear'))
clf_n.fit(X_n, Y_n)

#store the classifiers on the disk
joblib.dump(clf_z, 'expression_classifier_z.pkl')
joblib.dump(clf_f, 'expression_classifier_f.pkl')
joblib.dump(clf_n, 'expression_classifier_n.pkl')