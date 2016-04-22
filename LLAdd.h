#ifndef LL_NEWADD_H
#define LL_NEWADD_H

namespace dpl{


	//double **InitializeDictionary( int featureNumber, int sampleElementNumber ){

	//	double **Wd = (double**)malloc(sampleElementNumber*sizeof(double*));	
	//	for( unsigned int i=0; i<sampleElementNumber; i++ )
	//		Wd[i] = (double*)malloc(featureNumber*sizeof(double));	

	//	return Wd;	
	//}

void FeatureAvg(double** mtx2D_F_Avg,double** mtx2D_F,int count_D,int count_F){
		
		for( unsigned int i=0; i<count_F; i++ ){
			for( unsigned int j=0; j<count_D; j++ )
				mtx2D_F_Avg[i][j]+=(mtx2D_F[i][j])/3;
		}
	}
}



#endif /*  learning rate */



