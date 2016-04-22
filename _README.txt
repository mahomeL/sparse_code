Sparse Coordinate Coding

Input:

1. name of the file containing time series input data: file_S 
Input data should be a 2D matrix, with dimension MxN:
M (number of rows) is the total number of observations (e.g. time points) in the input data;
N (number of columns) is the total number of variables (e.g. voxels) in the input data;

Output:
1. file containing the learned dictionary: file_D
Output dictionary is a 2D matirx, with dimension MxK:
M (number of rows) is the total number of observations (e.g. time points) in the input data;
K (number of columns) is the pre-defined total number of dictionaries;

2. file containing the loading coefficient: file_z
Ouput loading is a 2D matrix, with dimension KxN:
K (number of columns) is the pre-defined total number of dictionaries;
N (number of columns) is the total number of variables (e.g. voxels) in the input data;



Operating Environment:
Linux (ubuntu/red hat)
g++ 4.6 or higher versions


How to run the program:
1. Open "run.cpp"
2. Compile the program:  g++ run.cpp -o run -fopenmp
4. Run the program:   ./run

