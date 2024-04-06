# DL_number_recognition
# Introduction
This is a pure math made digit recognition model (in model folder) .In the app folder is a drawing pad made with tkinter that allows users to  sent their drawings to the pridiction model.The check.py(in model folder ) can let us see the testing image data and the pridiction all together at the same window by enter the desire data index.Inside New_model folder is the latest version which is the faster and accurate one . 
# Installation
### Option 1
Assume that you have python(Python 3.11.3 is the developing env) installed .There are four required package to install inorder to run all the files.You can use the following command on the terminal to install those packages 
```
pip install tensorflow
pip install matplotlib
pip install numpy
pip install numba
pip install random
```
Then cd to the location where your downloaded files are ,run the main.py file 
```
python main.py
```
After a few minutes (or less) the training of the model should be finished with accuracy around 90%.By the time you will see two additional files updated which is weight1.csv and weight2.csv .The two file is crucial to app.py and check.py ,any modifacation may cause malfuntion to app.py and check.py .
### Option 2
If you don't want to train a model your self or don't want to install tensorflow,matplotlib
on your machine ,you can download download use the pretrained weights .check and app.py  can also work well.

