							Indian Institute of Information Technology-Allahabad
						            Pose and Expression Recognition on Multiple Yaw Angles 


* Group Members:
----------------
1.Richa Vinian IIT2015015
2.Payal Prasad IIT2015052
3.Puja Kumari LIT2015017
4.Samriddhi Niranjan IIT2015021

* Requirements:
---------------
CPU: Intel Core i7
RAM: 8GB
Operating System: Windows 10 Pro.
Anaconda IDE with Python 3
EmEditor.
Libraries: Numpy, Scikit-Learn, OpenCV3

* Installation:
---------------
1. Run the Anaconda Setup. Follow the instructions on the screen. This will Install Anaconda IDE on the system.
2. Open command prompt. Type "conda update conda" command to update the packages.
3. Install the libraries mentioned in the previous section using command prompt by using appropriate command for your system. Look on the web for appropriate command.
4. Now the environment is ready.


* Folder and File Description:
------------------------------

1. Code: Contains the source code.
2. Datasets: Contains the datasets used in the project.
3. Presentation: Contains the PPT.
4. Report: Contains the Report in PDF and DOC format.
5. Reference Papers: Contains the papers mentioned in the references section.
6. ReadME File
7. One Page Summary.

* Running the Project:
----------------------
For Training:-
Histogram files could be generated for any dataset using the 2 scripts present in "\Code\Histogram Files\" folder.
1.Generate the histogram files for the folder KDEF_All_in_one  present in the \Datasets folder by changing the relative path of the folder in the histogram_compute file.
2.Histogram Files are of two types:
   - Histogram_Angle -> each line contains the tuple (filename, histogram, angle)
   - Historgram_Angle_Expression -> each line contains the tuple (filename, histogram{different dimension than in the previous file}, angle, expression).
3.Paste the contents of both the generated files in  "\Code\LGBP With Angle Classification\train_angle.txt" and"\Code\LGBP With Angle Classification\train_expression.txt" respectively .Save and Close.
4. Before running the projct the first and foremost step is to generate the classifier files ("xxx.pkl") using the scripts given the folders named "DCLBP", "LGBP With Angle Classification" and "LGBP Without Angle Classifiaction" in the Code folder. This is done as follows.
5. Launch Spyder from the Start menu.
6. Now, run the following files in Spyder in the exact same order.
	i. \Code\LGBP With Angle Classification\pose_classifier.py
	ii. \Code\LGBP With Angle Classification\expression_classifier.py
7. The previous step generated 4 (.pkl) files in the directory "\Code\LGBP With Angle Classification\". These files are the classifiers we built using the training data which was present in the two files from which we copied the content.
8. Copy these files and paste in the directory "\Code\"
9. Now, open the file : "\Code\complete_lgbp_with_angle_classification.py" in Spyder.
10. Take the image to be tested and resize it to 169 x 229 using the "resizer.exe" tool present in the directory "\Datasets\resizer\"
11. Paste the resized image to be tested in the directory : \Datasets\Test_Images\.
12. Run the file opened in Step 8. Output wiil be displayed on the console after a few seconds.

To perform batch testing, paste the histogram of files in the "test_angle.txt" ,"test_expression.txt" files. Now first run "pose_prediction.py" and then "expression_prediction.py". The relevant stats will be displayed on the terminal.

The steps mententioned to generate the classifier files ("xxx.pkl") [Steps 1-5] can be used to generete the classifier of any of the three methods used in the project by choosing the appropriate script from any one of these folders : "DCLBP", "LGBP With Angle Classification" and "LGBP Without Angle Classifiaction" and pasting the appropriate training data in the "train.txt" files present in each of these folders seperately.
Accordingly the "\Code\complete_... .py" script could be chosen to test the image on any of the three approaches.
