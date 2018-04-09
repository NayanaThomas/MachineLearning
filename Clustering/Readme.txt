Contents
========
1. Assignment3_Part1.ipynb - Jupyter notebook file for the Anuran Calls dataset. 
2. Assignment3_Part2.ipynb - Jupyter notebook file for the Facebook Post metrics dataset.
3. Assignment3_Part1.py - Python file for the Anuran Calls dataset. 
4. Assignment3_Part2.py - Python file for the Facebook Post metrics dataset.
5. Result.docx - File containing output format of both the parts.

Assumptions
===========
1. Assumed that the python file/jupyter notebbok file is in the same directory as that of input dataset.
2. It is given that the file names will be Frogs_MFCCs.csv for Part 1 and dataset_Facebook.csv for Part 2. Assumed that it is unchanged.

Part 1
======
1. Only the columns 2-22 are used for the clustering
2. Based on the datatype of the attribute, the NaN values are replaced with standard values
		-if datatype is object, replaced with empty string
		-if datatype is int64, replaced with rounded off mean value
		-if datatype is float64, replaced with meanvalue
3. Methods used for performing clustering
		- K-Means clustering
		- H-Clustering
		- Gaussian Mixture Models
4. Result for each method consists of ratio of total within sum of squares/total sum of squares for k=1, …, 10

Part 2
======
1. Based on the datatype of the attribute, the NaN values are replaced with standard values
		-if datatype is object, replaced with empty string
		-if datatype is int64, replaced with rounded off mean value
		-if datatype is float64, replaced with meanvalue
2. Methods used for performing clustering
		- K-Means clustering
		- H-Clustering
		- Gaussian Mixture Models
3. Result for each method consists of ratio of total within sum of squares/total sum of squares for k=1, …, 10
