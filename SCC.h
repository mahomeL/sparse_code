#ifndef SPARSE_COORDINATE_CODING_H
#define SPARSE_COORDINATE_CODING_H

namespace dpl{

static unsigned int myseed;

double getAbs( double value ){
	if( value < 0 )
		return -1*value;
	else 
		return value;
}

double **FeatureInitialization( int count_D, int count_F ){

	double **feature = (double**)malloc(count_F*sizeof(double*));
        for( unsigned int i=0; i<count_F; i++ ){
		feature[i] = (double*)malloc(count_D*sizeof(double));
		for( unsigned int j=0; j<count_D; j++ )
			feature[i][j] = 0;
	}
	return feature;
}

std::vector<int> *NonZeroIndexInitialization( int count_F ){
	std::vector<int> *nonZeroIndex = new std::vector<int> [count_F];
	return nonZeroIndex;
}

double ShrinkageFunction( double value, double theta ){

	if( value < -theta )
		return value+theta;
	else if( value > theta )
		return value-theta;
	else
		return 0;
}

double *Initialize_A_Copy( int count_D ){

	double *A_Copy = (double*)malloc(count_D*sizeof(double));
	for( unsigned int i=0; i<count_D; i++ )
		A_Copy[i]=0;
	return A_Copy;
}

double *Initialize_A( int count_D ){

	double *A = (double*)malloc(count_D*sizeof(double));
	for( unsigned int i=0; i<count_D; i++ )
		A[i]=0;
	return A;
}

void Initialize_A( double *A, double *A_Copy, int count_D ){
	for( unsigned int i=0; i<count_D; i++ ){
		A[i]=A_Copy[i];
		A_Copy[i]=0;
    	}
}

void Update_A( double *A, double *A_Copy, double *feature, std::vector<int> &nonZeroIndex ){
	for( unsigned int i=0; i<nonZeroIndex.size(); i++ ){
		A[nonZeroIndex[i]] += feature[nonZeroIndex[i]]*feature[nonZeroIndex[i]];
		A_Copy[nonZeroIndex[i]] += feature[nonZeroIndex[i]]*feature[nonZeroIndex[i]];
	}
}

double getNonNegativeFeature( double featureElement, double optimalT ){
	if( featureElement+optimalT>=0 )
		return optimalT;
	else
		return -1*featureElement;
}

int *getRandomIndex( int size ){
 
	std::vector<int> index (size);
	int *data=(int*)malloc(size*sizeof(int));
	for( unsigned int i=0; i<size; i++ )
	        index[i] = i;
 
	for( unsigned int i=0; i<size; i++ ){
    		int randomIndex = rand()%index.size();
        	data[i] = index[randomIndex];
        	index.erase(index.begin()+randomIndex);
    	}	
    	return data;
}


//UpdateFeature( mtx2D_D, mtx2D_input[index], residuals, mtx2D_F[index], nonZeroIndex[index], double_lambda, count_layers, count_D, length_D, ifNonNegative );
void UpdateFeature( double **Wd, double *sample, double *residuals, double *feature, std::vector<int> &nonZeroIndex, double lambda, int layers, int count_D, int sampleElementNumber, bool NonNegative ){
    
    	for( unsigned int i = 0; i<sampleElementNumber; i++ ){
  		residuals[i] = -sample[i];
    		for( unsigned int j = 0; j<nonZeroIndex.size(); j++ )
    		        residuals[i] += Wd[i][nonZeroIndex[j]]*feature[nonZeroIndex[j]]; //
    	}

	nonZeroIndex.resize(0);
	int *randomIndex = getRandomIndex(count_D );

	for ( unsigned int i = 0; i < count_D; i++ ){

        	double optimalT;
        	double derivative = 0;

        	for (unsigned int j = 0;j < sampleElementNumber; j++)
                	derivative += (residuals[j]*Wd[j][randomIndex[i]]);
                
		optimalT = ShrinkageFunction( feature[randomIndex[i]]-derivative, lambda )-feature[randomIndex[i]];

		if( NonNegative ) 
			optimalT = getNonNegativeFeature( feature[randomIndex[i]], optimalT ); 

		feature[randomIndex[i]] += optimalT;

        	if ( optimalT!=0 ){
            		for (unsigned int j = 0;j < sampleElementNumber; j++)
                		residuals[j] += optimalT*Wd[j][randomIndex[i]];
        	}

		if( feature[randomIndex[i]]!=0 )
			nonZeroIndex.push_back(randomIndex[i]);

	}	

	for ( unsigned int k = 1; k < layers; k++ ){
		for ( unsigned int i = 0; i < nonZeroIndex.size(); i++ ){
        		double optimalT;
        		double derivative = 0;
        		for (unsigned int j = 0;j < sampleElementNumber; j++)
                		derivative += (residuals[j]*Wd[j][nonZeroIndex[i]]);
                
			optimalT = ShrinkageFunction( feature[nonZeroIndex[i]]-derivative, lambda )-feature[nonZeroIndex[i]];

			feature[nonZeroIndex[i]] += optimalT;

        		if ( optimalT!=0 ){
            			for (unsigned int j = 0;j < sampleElementNumber; j++)
                			residuals[j] += optimalT*Wd[j][nonZeroIndex[i]];
        		}
		}
	}

	nonZeroIndex.resize(0);
	for ( unsigned int i = 0; i < count_D; i++ ){
		if( feature[i]!=0 )
			nonZeroIndex.push_back(i);
	}
	free(randomIndex);
}

//UpdateWd( mtx2D_D, residuals, mtx2D_F[index], A, nonZeroIndex[index], length_D, (it+1), lambda2, mtx2D_DP, length_DP);
void UpdateWd( double **Wd, double *residuals, double *feature, double *A, std::vector<int> &nonZeroIndex, int length_D, unsigned int idx_iteration, double lambda2, double **mtx2D_DP, int length_DP)
{
	for ( unsigned int i = 0; i < length_DP; i++ )
    {
		for ( unsigned int j = 0; j < nonZeroIndex.size(); j++ )
		{
			//Wd[i][nonZeroIndex[j]] = Wd[i][nonZeroIndex[j]] - feature[nonZeroIndex[j]]*residuals[i]*dpl::learningRate(A, nonZeroIndex[j]);
			Wd[i][nonZeroIndex[j]] = Wd[i][nonZeroIndex[j]] - dpl::learningRate(A, nonZeroIndex[j])*(feature[nonZeroIndex[j]]*residuals[i]-lambda2*mtx2D_DP[i+(length_D-length_DP)][nonZeroIndex[j]]);
        }
    }
	for ( unsigned int i = length_DP; i < length_D; i++ )
    {
		for ( unsigned int j = 0; j < nonZeroIndex.size(); j++ )
		{
			//Wd[i][nonZeroIndex[j]] = Wd[i][nonZeroIndex[j]] - feature[nonZeroIndex[j]]*residuals[i]*dpl::learningRate(A, nonZeroIndex[j]);
			Wd[i][nonZeroIndex[j]] = Wd[i][nonZeroIndex[j]] - dpl::learningRate(A, nonZeroIndex[j])*feature[nonZeroIndex[j]]*residuals[i];
         }
    }    
}

void NormalizeWd( double **Wd, std::vector<int> &nonZeroIndex, int sampleElementNumber ){
	for( unsigned int i=0; i<nonZeroIndex.size(); i++ ){
		double sum = 0;
		for( unsigned int j=0; j<sampleElementNumber; j++ )
			sum += Wd[j][nonZeroIndex[i]]*Wd[j][nonZeroIndex[i]];
		sum = sqrt(sum);
		
		if( sum!=0 ){
			for( unsigned int j=0; j<sampleElementNumber; j++ )
				Wd[j][nonZeroIndex[i]] = Wd[j][nonZeroIndex[i]]/sum;
		}
	}
}

void saveFeature( double **feature, char *FeatureFileName, int count_D, int count_F ){
	
	printf("Save Features in %s\n", FeatureFileName);

	FILE *fp;
        fp = fopen( FeatureFileName, "w");
        if( fp == NULL ){
		printf("could not find feature file %s\n", FeatureFileName);
            	exit(0);
	}

	for( unsigned int i=0; i<count_D; i++ ){
		for( unsigned int j=0; j<count_F; j++)
	        	fprintf(fp, "%.15lf ", feature[j][i]);
		fprintf(fp, "\n");		
	}
	fclose(fp);
}

void saveNonZeroIndex( std::vector<int> *nonZeroIndex, char *IndexFileName, int count_D, int count_F ){
	
	printf("Save nonZero index in %s\n", IndexFileName);

	FILE *fp;
        fp = fopen( IndexFileName, "w");
        if( fp == NULL ){
		printf("could not find index file %s\n", IndexFileName);
            	exit(0);
	}

	for( unsigned int i=0; i<count_F; i++ ){
		for( unsigned int j=0; j<nonZeroIndex[i].size(); j++)
	        	fprintf(fp, "%d ", nonZeroIndex[i][j]);
		fprintf(fp, "\n");		
	}
	fclose(fp);
}

void clearFeature( int count_F, double **feature ){
	
	for( unsigned int i=0; i<count_F; i++ )
		free(feature[i]);
	free(feature);
}

double computeLassoResult( double **Wd, double *sample, double *feature, double lambda, int sampleElementNumber, int count_D ){
    
	double LassoResult = 0;
	double residuals;
	for( unsigned int i=0; i<sampleElementNumber; i++ ){
		residuals = -sample[i];
		for( unsigned int j=0; j<count_D; j++ )
			residuals += Wd[i][j]*feature[j];

		LassoResult += residuals*residuals;
	}
    
	double sum_feature = 0;
	for( unsigned int j=0; j<count_D; j++ )
		sum_feature += getAbs(feature[j]);
    
    	return 0.5*LassoResult+lambda*sum_feature;
}


double calculateError(  double **Wd,  double **sample, double **feature, double lambda, int count_F, int sampleElementNumber, int count_D ) {

	double TotalDecError = 0;
	for( unsigned int t=0; t<count_F; t++ ){
		TotalDecError += computeLassoResult( Wd, sample[t], feature[t], lambda, sampleElementNumber, count_D);
	}
	TotalDecError /= count_F;
	std::cout<<"Total Decode Error is "<<TotalDecError<<std::endl;
	return TotalDecError;
}


/////////////////////////////

/////////////////////////////
//////////////////////
///////////////
///////////
void trainDecoder( double **mtx2D_D, double **mtx2D_F, double **mtx2D_input, double double_lambda, int count_layers, int count_D, int count_F, int length_D, int count_iteration, bool ifNonNegative, char* file_summary, double double_stopping, double lambda2, double **mtx2D_DP, int length_DP)
{
	double *residuals = (double*)malloc(length_D*sizeof(double));
	double *A = Initialize_A( count_D );
	double *A_Copy = Initialize_A_Copy( count_D );
	std::vector<int> *nonZeroIndex = NonZeroIndexInitialization( count_F );

	FILE *fp;
	fp = fopen( file_summary, "w");
	srand((unsigned)time(0));
	myseed = (unsigned int) RAND_MAX * rand();	

	std::cout<<"Train decoder"<<std::endl;				
	double ComputionalTime = 0;
	double BeginTime = omp_get_wtime();  
	double double_errorNew = 0;
	double double_errorOld = 1;
	for( unsigned int it=0; it<count_iteration; it++ )
	{
		int index = it%count_F;
		if( index==0 )
		{
		Initialize_A( A, A_Copy, count_D );
		double_errorNew = calculateError( mtx2D_D, mtx2D_input, mtx2D_F, double_lambda, count_F, length_D, count_D );
		fprintf(fp, "%.15lf ", double_errorNew);
		if ((getAbs(double_errorNew-double_errorOld)/double_errorOld)<double_stopping)
		{
			break;
		}
		double_errorOld = double_errorNew;
		}
		UpdateFeature( mtx2D_D, mtx2D_input[index], residuals, mtx2D_F[index], nonZeroIndex[index], double_lambda, count_layers, count_D, length_D, ifNonNegative );
		Update_A( A, A_Copy, mtx2D_F[index], nonZeroIndex[index] );
		UpdateWd( mtx2D_D, residuals, mtx2D_F[index], A, nonZeroIndex[index], length_D, (it+1), lambda2, mtx2D_DP, length_DP);
		NormalizeWd( mtx2D_D, nonZeroIndex[index], length_D ); 
		/*	if( it%10000==0 )
		{
		std::cout<<it+1<<" iterations finished"<<std::endl;
		}*/
		
		
	}
	double EndTime = omp_get_wtime();  	
	ComputionalTime += (EndTime-BeginTime);
	fprintf(fp, "\n %f", ComputionalTime);

   	std::cout<<"Finish decoding process:"<<std::endl;
	std::cout<<"Train Decode Time is "<<ComputionalTime<<" seconds."<<std::endl;
	free(A_Copy);
	free(residuals);
	delete [] nonZeroIndex;
	fclose(fp);
}
//////////////////////////////////////
///////////////////////////////

///////////////////
///////////

}

#endif /* Sparse Coordinate Coding */

