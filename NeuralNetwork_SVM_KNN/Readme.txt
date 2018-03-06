Members
=======
1. Nayana Thomas - nxt170630
2. Nihal Abdulla PT - nxp171730

Contents
========
1. Assignment2_Part1.ipynb - Jupyter notebook file for the classification of Census income to <=50K or >50K
2. Assignment2_Part2.ipynb - Jupyter notebook file for the number of bikeRental per hour prediction
3. Result.docx - File containing output format of both the parts.

Assumptions
===========
1. Assumed that the python file/jupyter notebbok file is in the same directory as that of input dataset.
2. It is given that the file names will be adultTrain.data, adultTest.data and adult.names for Part 1. Assumed that it is unchanged.
3. It is given that the file names will be bikeRentalHourlyTrain.csv and bikeRentalHourlyTest.csv for Part 2. Assumed that it is unchanged.

Part 1
======
1. During data cleaning replaced all '?' values with NaN type.
2. Based on the datatype of the attribute, the NaN values are replaced with standard values
		-if datatype is object, replaced with empty string
		-if datatype is int64, replaced with rounded off mean value
		-if datatype is float64, replaced with meanvalue
3. Implemented models
		- Neural Network
		- SVM (Both Linear and Radial Models)
		- KNN (With k value 6) 
4. Result for each model consists of Confusion Matrix and Accuracy Score

Part 2
======
1. After reading the data, unwanted columns are removed
2. Based on the datatype of the attribute, the NaN values are replaced with standard values
		-if datatype is object, replaced with empty string
		-if datatype is int64, replaced with rounded off mean value
		-if datatype is float64, replaced with meanvalue
3. Implemented models
		- Neural Network
		- Linear Regression (Both Lasso and Ridge Models)
		- KNN (With k value 8) 
4. Result for each model consists of Mean Squared Error and Cross Validation using 5-fold validation



