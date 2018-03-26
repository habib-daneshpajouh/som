
/**
 *
 * @author Habib Daneshpajouh
 * @email habib.dpajouh@gmail.com
 *
 **/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>

typedef struct {
	int x;
	int y;
} Point2D;

const char* INPUT_FILE_PATH = "iris.data";

float* vectors;
float* weights;

char* labels;
Point2D* clusterIdxs;

const int NUM_VEC = 150;
const int VEC_DIM = 4;
const int LATTICE_X_DIM = 16;
const int LATTICE_Y_DIM = 4;
const int NUM_NEURONS = LATTICE_X_DIM * LATTICE_Y_DIM;
const int TOTAL_NUM_DATA_PARAMS = NUM_VEC * VEC_DIM; //total no. of parameters in all the data vectors
const int TOTAL_NUM_WEIGHTS = NUM_NEURONS * VEC_DIM; //total no. of weights in all the neurons
const int NUM_EPOCHS = LATTICE_X_DIM * NUM_VEC * 20; //epoch = training iteration
const float START_LEARNING_RATE = 0.01;

float LATTICE_RADIUS;
float learningRate = START_LEARNING_RATE;


void readInputData();
void readInputDataWithLabel();
void initLatticeRandom();
Point2D findBMU(int vecStartIdx, int vecEndIdx);
void updateWeights(int vecStartIdx, int vecEndIdx, Point2D bmuPos, float neighWidth);
void assignCluster();
void printWeights();
void printVectorsWithLabels();
void printVectorsWithLabelsAndClusters();
float calcAvgQuantizationError();


int main(int argc, char** argv) {

	if(LATTICE_X_DIM > LATTICE_Y_DIM)
		LATTICE_RADIUS = LATTICE_X_DIM / 2.0;
	else
		LATTICE_RADIUS = LATTICE_Y_DIM / 2.0;

	const size_t DATA_SIZE = TOTAL_NUM_DATA_PARAMS * sizeof(float);
	const size_t WEIGHTS_SIZE = TOTAL_NUM_WEIGHTS * sizeof(float);
	const size_t LABELS_SIZE = NUM_VEC * sizeof(char);
	const size_t CLUSTER_IDXS_SIZE = NUM_VEC * sizeof(Point2D);
	const float TIME_CONSTANT = NUM_EPOCHS / log(LATTICE_RADIUS);

	//init random seed
	srand(time(NULL));

	//* read input data *//
	vectors = (float*) malloc(DATA_SIZE);
	labels = (char*) malloc(LABELS_SIZE);
	readInputDataWithLabel(); 

	//* initialize the lattice *//
	weights = (float*) malloc(WEIGHTS_SIZE);
	initLatticeRandom();
	//printWeights();

	//* start training *//
	for(int epoch=1; epoch<=NUM_EPOCHS; epoch++) {

		//take a data vector by random
		int vecStartIdx = (rand() % NUM_VEC) * VEC_DIM;
		int vecEndIdx = vecStartIdx + VEC_DIM;

		//* find Best Matching Unit (BMU) *//
		Point2D bmuPos = findBMU(vecStartIdx, vecEndIdx);

		//* update weights *//
		float neighRadius = LATTICE_RADIUS * exp(-(float)epoch/TIME_CONSTANT);
		float neighWidth = pow(neighRadius, 2);
		updateWeights(vecStartIdx, vecEndIdx, bmuPos, neighWidth);

		//reduce the learning rate
		learningRate = START_LEARNING_RATE * exp(-(float)epoch / NUM_EPOCHS);

		//Print stats
		if(epoch%1000 == 0) {
			float avgQuantError = calcAvgQuantizationError();
			printf("Epoch #%d avgQuantError = %.2f\n", epoch, avgQuantError*100);
		}
	}

	//assign a cluster to each vector
	clusterIdxs = (Point2D*) malloc(CLUSTER_IDXS_SIZE);
	assignCluster();
	//printVectorsWithLabelsAndClusters();

	//printWeights();
	printf("\nDone.\n");
}


void readInputData() {

	FILE* inFile = fopen(INPUT_FILE_PATH, "r");

	for(int i=0; i<TOTAL_NUM_DATA_PARAMS; i++)
		fscanf(inFile, "%f", &vectors[i]);
}


void readInputDataWithLabel() {

	FILE* inFile = fopen(INPUT_FILE_PATH, "r");

	for(int i=0; i<NUM_VEC; i++) {
		for(int j=0; j<VEC_DIM; j++)
			fscanf(inFile, "%f", &vectors[i*VEC_DIM + j]); 
		fscanf(inFile, "%s", &labels[i]); 
	}
}


void initLatticeRandom() { //initialize the weights of neurons in the lattice

	for(int i=0; i<NUM_NEURONS; i++) {

		//take a data vector by random
		int vecStartIdx = (rand() % NUM_VEC) * VEC_DIM;
		int vecEndIdx = vecStartIdx + VEC_DIM;

		int weightStartIdx = i * VEC_DIM;
		int weightEndIdx = weightStartIdx + VEC_DIM;

		for(int v=vecStartIdx, w=weightStartIdx; v<vecEndIdx && w<weightEndIdx; v++, w++)
			weights[w] = vectors[v];
	}
}


Point2D findBMU(int vecStartIdx, int vecEndIdx) {

	Point2D p; //BMU position

	float minDist = FLT_MAX;

	for(int i=0; i<LATTICE_Y_DIM; i++) {
		for(int j=0; j<LATTICE_X_DIM; j++) {

			float dist = 0;
			for(int v=vecStartIdx, w=0; v<vecEndIdx && w<VEC_DIM; v++, w++) {

				int weightIdx = (i*LATTICE_X_DIM*VEC_DIM) + (j*VEC_DIM) + w;
				dist += pow(vectors[v]-weights[weightIdx], 2);
			}

			dist = sqrt(dist);
			if(dist < minDist) {
				minDist = dist;
				p.y = i;
				p.x = j;
			}
		}
	}

	return p;
}


void updateWeights(int vecStartIdx, int vecEndIdx, Point2D bmuPos, float neighWidth) {

	//go through each neuron and update its weights if it's located within the BMU's neighborhood
	for(int i=0; i<LATTICE_Y_DIM; i++) {
		for(int j=0; j<LATTICE_X_DIM; j++) {

			float distToBMUsq = pow(i-bmuPos.y, 2) + pow(j-bmuPos.x, 2);
			if(distToBMUsq < neighWidth) {

				//calculate the distance influence rate
				double distInfluenceRate = exp(-distToBMUsq / (2*neighWidth));
				for(int v=vecStartIdx, w=0; v<vecEndIdx && w<VEC_DIM; v++, w++) {

					int weightIdx = (i*LATTICE_X_DIM*VEC_DIM) + (j*VEC_DIM) + w;
					weights[weightIdx] += learningRate * distInfluenceRate * (vectors[v] - weights[weightIdx]);
				}
			}
		}
	}
}


void assignCluster() {

	for(int v=0; v<NUM_VEC; v++) {

		int vecStartIdx = v * VEC_DIM;
		int vecEndIdx = vecStartIdx + VEC_DIM;

		//* find Best Matching Unit (BMU) *//
		Point2D bmuPos = findBMU(vecStartIdx, vecEndIdx);
		clusterIdxs[v] = bmuPos;
	}
}


void printWeights() {

	printf("\n");
	for(int i=0; i<LATTICE_Y_DIM; i++) {
		for(int j=0; j<LATTICE_X_DIM; j++) {

			printf("Neuron[%d,%d]: ", i, j);
			for(int w=0; w<VEC_DIM; w++) {

				int weightIdx = (i*LATTICE_X_DIM*VEC_DIM) + (j*VEC_DIM) + w;
				printf("%f ", weights[weightIdx]);
			}
			printf("\n\n");
		}
	}

}


void printVectorsWithLabels() {

	printf("\n");
	for(int i=0; i<NUM_VEC; i++) {
		printf("Vec#%d", i+1);

		for(int j=0; j<VEC_DIM; j++)
			printf(" %.1f", vectors[i*VEC_DIM + j]);

		printf("  Label:%c\n", labels[i]);
	}
}


void printVectorsWithLabelsAndClusters() {

	printf("\n");
	for(int i=0; i<NUM_VEC; i++) {
		printf("Vec#%d", i+1);

		for(int j=0; j<VEC_DIM; j++)
			printf(" %.1f", vectors[i*VEC_DIM + j]);

		printf("  Label:%c  ", labels[i]);
		printf("  Cluster: %d,%d \n", clusterIdxs[i].x, clusterIdxs[i].y);
	}
}


float calcAvgQuantizationError() {

	float avgQuantError = 0;

	//go through each input vector
	for(int n=0; n<NUM_VEC; n++) {

		int vecStartIdx = n * VEC_DIM;
		int vecEndIdx  = vecStartIdx + VEC_DIM;

		float minDist = FLT_MAX;

		for(int i=0; i<LATTICE_Y_DIM; i++) {
			for(int j=0; j<LATTICE_X_DIM; j++) {

				float dist = 0;
				for(int v=vecStartIdx, w=0; v<vecEndIdx && w<VEC_DIM; v++, w++) {

					int weightIdx = (i*LATTICE_X_DIM*VEC_DIM) + (j*VEC_DIM) + w;
					dist += pow(vectors[v]-weights[weightIdx], 2);
				}

				dist = sqrt(dist);
				if(dist < minDist)
					minDist = dist;
			}
		}

		avgQuantError += minDist;
	}

	avgQuantError /= NUM_VEC;

	return avgQuantError;
}
