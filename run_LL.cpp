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
#include "LLAdd.h"


void FeatureCopy(double** mtx2D_F_Avg,double** mtx2D_F,int count_D,int count_F){

	for( unsigned int i=0; i<count_F; i++ ){
		for( unsigned int j=0; j<count_D; j++ )
			mtx2D_F_Avg[i][j]=mtx2D_F[i][j];
	}
}

int main(int argc, char* argv[])
{
if (argc!=18)
	{
		std::cout<<"1 file_input 2 file_outputD 3 file_outputF 4 count_D 5 length_D 6 count_layers 7 count_epoch 8 double_lambda 9 ifNonNegative 10 file_summary 11 double_stopping 12 double_lambda2 13 file_DP 14 length_DP	15 ifInitialize\n";
		exit(1);
	}
	char* file_input_1 =argv[1];// "1.WM.sig.txt";
	char* file_outputD_1 =argv[2]; //"Dictionary.txt";
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
	char* file_DP =argv[13]; //"predefinedD.txt" file_outputD_1 
	int length_DP =atoi(argv[14]);// 405;                         5==14,2==13
	bool ifInitialize =atoi(argv[15]);// 1;

	char* file_input_2 =argv[16];// "1.WM.sig.txt";
	char* file_outputD_2 =argv[17]; //"Dictionary.txt";

	char* file_input[2] ={argv[1],argv[16]};
	/*char* file_outputD[2] ={file_outputD_1,file_outputD_2};*/ //"Dictionary.txt";
	char* file_outputD[2] ={argv[2],argv[17]};
	int count_F = dpl::getSampleNumber( file_input[0] );

	
	double** mtx2D_F_Avg=dpl::FeatureInitialization( count_D, count_F);
	for(int inter=0;inter<count_epoch;++inter){
		double **mtx2D_input;
		double **mtx2D_D;
		double** mtx2D_F_Avg_temp=dpl::FeatureInitialization( count_D, count_F);
		for(int sample=0;sample<2;++sample){
	   std::cout<<"***************interation   "<<inter<<"  ***************"<<std::endl;
			double** mtx2D_F;
			double** mtx2D_DP;
			//std::cout<<"Begin to read input file..."<<std::endl;
			mtx2D_input = dpl::ReadSample( file_input[sample], count_F, length_D );
			dpl::SampleNormalization( mtx2D_input, count_F, length_D );
	
			if(inter==0){
				
				//std::cout<<"Begin to initialize dictionary..."<<std::endl;
				mtx2D_D = dpl::GenerateRandomPatchDictionary( count_D, length_D, count_F, mtx2D_input );	
				dpl::DictionaryNormalization( count_D, length_D, mtx2D_D );
				mtx2D_DP= dpl::GenerateRandomPatchDictionary( count_D, length_D, count_F, mtx2D_input );	
				dpl::DictionaryNormalization( count_D, length_D, mtx2D_D );
			}
			else{
				mtx2D_DP = dpl::readDictionary(file_outputD[sample], count_D, length_D);
				mtx2D_D=dpl::readDictionary(file_outputD[sample], count_D, length_D);
			}

		
			
			/*std::cout<<"Number of samples is "<<count_F<<std::endl;
			std::cout<<"Length of samples is "<<length_D<<std::endl;
			std::cout<<"Number of dictionaries is "<<count_D<<std::endl;
			std::cout<<"lambda is "<<double_lambda<<std::endl;*/
	
			if(inter==0 && ifInitialize  )
			{
				std::cout<<"Use pre-defined dictionary for (partial) D initialization..."<<std::endl;
				dpl::predefineDictionary( mtx2D_D, mtx2D_DP, count_D, length_D, length_DP);
			}
			dpl::DictionaryNormalization( count_D, length_D, mtx2D_D );
			dpl::DictionaryNormalization( count_D, length_D, mtx2D_DP );
			//dpl::saveDictionary( count_D, length_D, mtx2D_D, file_generatedD);

			if(inter==0)
				mtx2D_F = dpl::FeatureInitialization( count_D, count_F);
			else{
				mtx2D_F = dpl::FeatureInitialization( count_D, count_F);
				FeatureCopy(mtx2D_F,mtx2D_F_Avg,count_D,count_F);
			}

			//std::cout<<"Begin to train "<<std::endl;
			dpl::trainDecoder( mtx2D_D, mtx2D_F, mtx2D_input, double_lambda, count_layers, count_D, count_F,  length_D, count_F, ifNonNegative, file_summary, double_stopping, double_lambda2, mtx2D_DP, length_DP);
			//std::cout<<"Finish training "<<std::endl;

			dpl::FeatureAvg(mtx2D_F_Avg_temp,mtx2D_F,count_D,count_F);
			
			dpl::saveDictionary( count_D, length_D, mtx2D_D, file_outputD[sample] );	
			//dpl::saveFeature( mtx2D_F, file_outputF, count_D, count_F );
	
			dpl::clearSample( count_F, mtx2D_input );
			dpl::clearFeature( count_F, mtx2D_F );
			dpl::clearDictionary( length_D, mtx2D_D );
			dpl::clearDictionary( length_DP, mtx2D_DP );
			std::cout<<"!!!!NEXT!!!!"<<std::endl;
		}
		FeatureCopy(mtx2D_F_Avg,mtx2D_F_Avg_temp,count_D,count_F);
		dpl::clearFeature( count_F, mtx2D_F_Avg_temp );
	}
	dpl::saveFeature( mtx2D_F_Avg, file_outputF, count_D, count_F );
	dpl::clearFeature( count_F, mtx2D_F_Avg );
	
	std::cout<<"\nALL DONE\n";
	system("pause");
	return 0;
}
