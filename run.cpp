/*
	Sparse Coordinate Coding  version 1.0.2
*/
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <ctime>
#include <iomanip>
#include <string>
#include <omp.h>
#include "DictionaryGeneration.h"
#include "SampleNormalization.h"
#include "LR.h"
#include "SCC.h"

int main(int argc, char* argv[])
{
if (argc!=16)
	{
		std::cout<<"1 file_input 2 file_outputD 3 file_outputF 4 count_D 5 length_D 6 count_layers 7 count_epoch 8 double_lambda 9 ifNonNegative 10 file_summary 11 double_stopping 12 double_lambda2 13 file_DP 14 length_DP	15 ifInitialize\n";
		exit(1);
	}
	char* file_input =argv[1];// "1.WM.sig.txt";
	char* file_outputD =argv[2]; //"Dictionary.txt";
	char* file_outputF = argv[3];//"mtx2D_F.txt";
	int count_D =atoi(argv[4]); //400;
	int length_D =atoi(argv[5]);// 405;
	int count_layers =atoi(argv[6]);// 3;
	int count_epoch = atoi(argv[7]);//10;
	double double_lambda=atof(argv[8]);//0.13;
	bool ifNonNegative = atoi(argv[9]);//true;
	char* file_summary = argv[10]; // including error and number of iterations;
	double double_stopping = atof(argv[11]); // stopping criteria;
	double double_lambda2 = atof(argv[12]); // balancing constant for enforced learning of mtx2D_M (increasing correlation to mtx2D_DP);
	char* file_DP =argv[13]; //"predefinedD.txt";
	int length_DP =atoi(argv[14]);// 405;
	bool ifInitialize =atoi(argv[15]);// 1;
	double** mtx2D_DP = dpl::readDictionary(file_DP, count_D, length_D);
	char file_generatedD[100] = "randomlyGeneratedD.txt";
	

	double **mtx2D_D;
	double **mtx2D_F;
	double **mtx2D_input;
	int count_F = dpl::getSampleNumber( file_input );
	int count_iteration = count_F*count_epoch;

	std::cout<<"Number of samples is "<<count_F<<std::endl;
	std::cout<<"Length of samples is "<<length_D<<std::endl;
	std::cout<<"Number of dictionaries is "<<count_D<<std::endl;
	std::cout<<"Number of iterations is "<<count_iteration<<std::endl;
	std::cout<<"lambda is "<<double_lambda<<std::endl;
	
	std::cout<<"Begin to read input file..."<<std::endl;
	mtx2D_input = dpl::ReadSample( file_input, count_F, length_D );
	dpl::SampleNormalization( mtx2D_input, count_F, length_D );

	std::cout<<"Begin to initialize dictionary..."<<std::endl;
	mtx2D_D = dpl::GenerateRandomPatchDictionary( count_D, length_D, count_F, mtx2D_input );	
	dpl::DictionaryNormalization( count_D, length_D, mtx2D_D );
	if( ifInitialize )
	{
		std::cout<<"Use pre-defined dictionary for (partial) D initialization..."<<std::endl;
		dpl::predefineDictionary( mtx2D_D, mtx2D_DP, count_D, length_D, length_DP);
	}
	dpl::DictionaryNormalization( count_D, length_D, mtx2D_D );
	//dpl::saveDictionary( count_D, length_D, mtx2D_D, file_generatedD);

	mtx2D_F = dpl::FeatureInitialization( count_D, count_F);
	std::cout<<"Begin to train "<<std::endl;
	dpl::trainDecoder( mtx2D_D, mtx2D_F, mtx2D_input, double_lambda, count_layers, count_D, count_F,  length_D, count_iteration, ifNonNegative, file_summary, double_stopping, double_lambda2, mtx2D_DP, length_DP);
	std::cout<<"Finish training "<<std::endl;

	dpl::saveDictionary( count_D, length_D, mtx2D_D, file_outputD );	
	dpl::saveFeature( mtx2D_F, file_outputF, count_D, count_F );
	
	dpl::clearSample( count_F, mtx2D_input );
	dpl::clearFeature( count_F, mtx2D_F );
	dpl::clearDictionary( length_D, mtx2D_D );
	std::cout<<"Hello World!"<<std::endl;
	
	return 0;
}
