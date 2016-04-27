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

double FangCha(double data[],int length){
	double sum=0;
	for(int i=0;i<length;++i)
		sum+=data[i];

	sum/=length;
	double fangcha=0;
	for(int i=0;i<length;++i)
		fangcha+=(data[i]-sum)*(data[i]-sum);

	return fangcha/length;
}

double FangChaMatrix(double **matrix,int d,int f){
	double sum=0;
	for(int i=0;i<f;++i)
		for(int j=0;j<d;++j)
			sum+=matrix[i][j];

	sum/=(d*f);

	double fangcha=0;
	for(int i=0;i<f;++i)
		for(int j=0;j<d;++j)
		fangcha+=(matrix[i][j]-sum)*(matrix[i][j]-sum);

	return fangcha/(d*f);
}

int main(int argc, char* argv[])
{
//if (argc!=18)////////
//	{
//		std::cout<<"1 file_input 2 file_outputD 3 file_outputF 4 count_D 5 length_D 6 count_layers 7 count_epoch 8 double_lambda 9 ifNonNegative 10 file_summary 11 double_stopping 12 double_lambda2 13 file_DP 14 length_DP	15 ifInitialize\n";
//		exit(1);
//	}
	char* file_input_1 =argv[1];// "1.WM.sig.txt";
	char* file_outputD_1 =argv[2]; //"Dictionary.txt";
	char* file_outputF = argv[3];//"mtx2D_F.txt";
	int count_D =atoi(argv[4]); //100
	int length_D =atoi(argv[5]);// 70
	int count_layers =atoi(argv[6]);// 3;
	int count_epoch = atoi(argv[7]);//20;
	double double_lambda=atof(argv[8]);//0.08;
	bool ifNonNegative = atoi(argv[9]);//true;
	char* file_summary = argv[10]; // including error and number of iterations;
	double double_stopping = atof(argv[11]); // stopping criteria;
	double double_lambda2 = atof(argv[12]); // balancing constant for enforced learning of mtx2D_M (increasing correlation to mtx2D_DP);
	char* file_DP =argv[13]; //用来提前初始化的特征Z
	int length_DP =atoi(argv[14]);//   70                    5==14,2==13
	bool ifInitialize =atoi(argv[15]);// 1;


	const int people=atoi(argv[16]); //总共训练的个体数
	
	char** file_input =new char*[people]; //1,17,19,21,23,25
	file_input[0]=argv[1];
	for(int i=1;i<people;++i)
		file_input[i]=argv[16+2*i-1];

	char** file_outputD =new char*[people];//2,18,20,22
	file_outputD[0]=argv[2];
	for(int i=1;i<people;++i)
		file_outputD[i]=argv[16+2*i];


	int count_F = dpl::getSampleNumber( file_input[0] );

	std::cout<<"原始文件长度为："<<count_F<<std::endl;

	double** mtx2D_F_Avg=dpl::FeatureInitialization( count_D, count_F);

////////////////////////////////////////////////////////////////////////////////////
	//初始化各种,residuals,A,A_Copy,nonZeroIndex
	//double *residuals = (double*)malloc(length_D*sizeof(double));
	double** residuals=new double*[people];
	for(int i=0;i<people;++i)
		residuals[i]=new double[length_D];
	
	double **A=new double*[people];
	for(int i=0;i<people;++i)
		A[i]=dpl::Initialize_A( count_D );

	double **A_Copy=new double*[people];
	for(int i=0;i<people;++i)
		A_Copy[i]=dpl::Initialize_A_Copy( count_D );

	/*double *A = Initialize_A( count_D );
	double *A_Copy = Initialize_A_Copy( count_D );*/
	std::vector<int>**nonZeroIndex=new std::vector<int>*[people];
	for(int i=0;i<people;++i)
		nonZeroIndex[i] = dpl::NonZeroIndexInitialization( count_F );


	double* double_errorNew=new double[people];
	for(int i=0;i<people;++i)
		double_errorNew[i]=0;

	double* double_errorOld=new double[people];
	for(int i=0;i<people;++i)
		double_errorOld[i]=1;
	 
/////////////////////////////////////////////////////////////////////////////////////	
//Feature(count_d,count_f):new 32792*100的f*d，存的时候是100*32792的
//Dictionary(count_D, length_D):new 70*100的lengthD*count_D,存是70*100
/////////////////////////////////////////////////////////////////////////////////////
	for(int inter=0;inter<count_epoch;++inter){

		double **mtx2D_input;
		double **mtx2D_D;
		double** mtx2D_F_Avg_temp=dpl::FeatureInitialization( count_D, count_F);

		for(int sample=0;sample<people;++sample){

			std::cout<<"***************interation   "<<inter+1<<"-"<<sample+1<<"  ***************"<<std::endl;
			double** mtx2D_F;
			double** mtx2D_DP;
			//std::cout<<"Begin to read input file..."<<std::endl;
			mtx2D_input = dpl::ReadSample( file_input[sample], count_F, length_D );
			dpl::SampleNormalization( mtx2D_input, count_F, length_D );
	

			//下面这段作用是，产生随机的D，然后用来初始化
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
			if(inter==0 && ifInitialize  )
			{
				std::cout<<"Use pre-defined dictionary for (partial) D initialization..."<<std::endl;
				dpl::predefineDictionary( mtx2D_D, mtx2D_DP, count_D, length_D, length_DP);
			}
			dpl::DictionaryNormalization( count_D, length_D, mtx2D_D );
			dpl::DictionaryNormalization( count_D, length_D, mtx2D_DP );
			////////////////////////////


			if(inter==0){
				std::cout<<"Begin to initialize Z..."<<std::endl;
				mtx2D_F = dpl::FeatureInitialization( count_D, count_F);
			   mtx2D_F = dpl::readFeature(file_DP,count_D,count_F);//读入的Feature为32792*100txt
			}
			else{
				mtx2D_F = dpl::FeatureInitialization( count_D, count_F);
				FeatureCopy(mtx2D_F,mtx2D_F_Avg,count_D,count_F);
			}

			//std::cout<<"Begin to train "<<std::endl;
			double_errorNew[sample]=dpl::trainDecoder(A[sample],A_Copy[sample],residuals[sample],nonZeroIndex[sample],mtx2D_D, mtx2D_F, mtx2D_input, double_lambda, count_layers, count_D, count_F,  length_D, count_F, ifNonNegative, file_summary, double_stopping, double_lambda2, mtx2D_DP, length_DP);
			FILE* errornew;
			errornew=fopen(file_summary,"a");//不覆盖写入
			fprintf(errornew,"%f\t",double_errorNew[sample]);
			fclose(errornew);

			// double_errorNew[people] =dpl::trainDecoder(A[sample],A_Copy[sample],residuals[sample],nonZeroIndex[sample],mtx2D_D, mtx2D_F, mtx2D_input, double_lambda, count_layers, count_D, count_F,  length_D, count_F, ifNonNegative, file_summary, double_stopping, double_lambda2, mtx2D_DP, length_DP);
			//std::cout<<"Finish training "<<std::endl;

			dpl::FeatureAvg(mtx2D_F_Avg_temp,mtx2D_F,count_D,count_F,people);
			//矩阵方差
			double Featurefangcha=FangChaMatrix(mtx2D_F_Avg_temp,count_D, count_F);
			std::cout<<"****Z方差为：\t"<<Featurefangcha<<"*******"<<std::endl;
			FILE* file_fangcha;
			file_fangcha=fopen(file_summary,"a");//不覆盖写入
			fprintf(file_fangcha,"%f\t",Featurefangcha);
			fclose(file_fangcha);

			dpl::saveDictionary( count_D, length_D, mtx2D_D, file_outputD[sample] );	
			//dpl::saveFeature( mtx2D_F, file_outputF, count_D, count_F );
	
			dpl::clearSample( count_F, mtx2D_input );
			dpl::clearFeature( count_F, mtx2D_F );
			dpl::clearDictionary( length_D, mtx2D_D );
			dpl::clearDictionary( length_DP, mtx2D_DP );
		}
		/*double fangcha=FangCha(double_errorNew,people);
		std::cout<<"****方差为：\t"<<fangcha<<std::endl;*/
		
		//mtx2D_F_Avg=dpl::FeatureInitialization( count_D, count_F);

		FeatureCopy(mtx2D_F_Avg,mtx2D_F_Avg_temp,count_D,count_F);
		dpl::clearFeature( count_F, mtx2D_F_Avg_temp );
		
	}


	for(int i=0;i<people;++i){
		free(A[i]);
		free(A_Copy[i]);
		delete [] residuals[i];
		delete [] nonZeroIndex[i];
	}
	
	delete [] A;
	delete [] A_Copy;
	delete [] residuals;
	delete [] nonZeroIndex;
	delete [] double_errorNew;
	delete [] double_errorOld;

	dpl::saveFeature( mtx2D_F_Avg, file_outputF, count_D, count_F );
	dpl::clearFeature( count_F, mtx2D_F_Avg );
	
	std::cout<<"\nALL DONE\n";
	system("pause");
	return 0;
}

