
Instructions
============
1. The zip folder contains two scripts and two train datasets

2. textData_data1.py - this assumes the training data file name is textTrainData.txt
		     - this assumes the training data file name is textTestData.txt
		     - both the train data and the test data should be in the same directory as the python script. 

3. carData_data2.py - this assumes the training data file name is carTrainData.csv
		    - this assumes the training data file name is carTestData.csv
		    - both the train data and the test data should be in the same directory as the python script.

4. run the following command
	- python.exe textData_data1.py - windows
	- python3 textData_data1.py - mac os
	- python.exe carData_data2.py - windows		
	- python3 carData_data2.py - macos
		- will print the train accuracy and train confusion matrix first 
		- will print the test accuracy and test confusion matrix second

5. Packages Used
	- pandas
	- numpy
	- sklearn
	- collections
	- re
