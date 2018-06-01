#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>

using namespace std;

ofstream myfile;

__global__ void mykernel(void){}

//generates random integers
//seed helps generate the seed
void random_ints(int **arr, int size, int seed)
{
	srand(time(NULL) * seed);

	for (int r = 0; r < size; ++r){
   		
   		for (int c = 0; c < size; ++c){
   
    		arr[r][c] = (rand() % 10);
    	
    	}
	}
}

//returns 2D array of size, size *size
int** initialize2DArray(int size){
	int **arr = new int*[size];
	for(int i = 0; i < size; ++i) {
    	arr[i] = new int[size];
	}
	
	return arr;
}

//deletes arr of size* size
void delete2DArray (int** arr, int size){
	for(int i = 0; i < size; ++i) {
    	delete [] arr[i];
	}
	delete [] arr;
}

//Exectutes matric multiplication on a 2D array using CPU
//1 thread only
//returns time it took to do the computation
double matrixMultCPU (int** a, int** b, int** c, int size){

	for (int rc = 0; rc < size; rc++){ //row of array c
	
		for (int cc = 0; cc < size; cc++){ //column of array c, size of row
			
			//each index of matrix mult we are going to iterate col times
				//get arrA[i][rc]  
				//get arrB[cc][i]
				//multiply and add to total
			//set c[rc][cc] = total
			
			int total = 0; 
			for (int i = 0; i < size; i++){
				int curA = a[rc][i];
				int curB = b[i][cc];
				total += curA * curB;
			}
			c[rc][cc] = total; 
				
		} 
	
	}
	
	return 0.0;

}

//computes matrix multiplication on 1D arrays a and b and puts the results into c
__global__ void MatrixMultGPU (int* a, int* b, int* c, int size){

	//gets the ID of the current thread in the block
	int x = threadIdx.x;
	int y = threadIdx.y; 
	
	int product = 0;
	
	for (int i = 0; i < size; i++){
		
		//get coordinates in array a
		int coorA = y * size + i;
		
		//get coordinates in array b 
		int coorB = x * size + i;
		
		//set product to what u got 
		product += a[coorA] * b[coorB];
	
	}
	
	c[y * size + x] = product; 
 
 	print1DArray(c, size);
 
}

void print1DArray(int * arr, int size){

	for (int r = 0; r < size; r++){
	
		for (int c = 0; c < size; c++){
		
			cout << setw(4) << arr[r * size + c];
		
		}
		
		cout << endl;
		
	}

}

void printArray(int** arr, int size){

	for (int r = 0; r < size; r++){
	
		for (int c = 0; c < size; c++){
			
			cout << setw(4) << arr[r][c] << " ";
				
		}
		
		cout << endl;
	
	}

}

//copies the int** in to int* out 
void copyArrToArr(int** in, int* out, int size){

	for (int r = 0; r < size; r++){
	
		for (int c = 0; c < size; c++){
		
			out[r * size + c] = in[r][c];
		
		}
	
	}

}

//conducts matrix mult with inA and inB matrixes
void executeCudaCalculations(int** inA, int** inB, int size){

	//create arrays 
	
	cout << "creating cuda arrays" << endl;
	
	int* a;
	int* b;
	int* c; //array to hold c, which is result of
	
	cout << "copying CPU arrays to local CPU 1D arrays" << endl;
	
	
	int memSize = size * size * sizeof(int); //technically 1D array which is size * size
	
	a = (int *)malloc(memSize); 
	b = (int *)malloc(memSize); 
	c = (int *)malloc(memSize); 
	
	//copies inA and inB to a and b
	copyArrToArr(inA, a, size); 
	copyArrToArr(inB, b, size);
	
	cout << "creating device variables" << endl;
	
	//device variables	
	int* da;
	int* db;
	int* dc;
	
	cout << "allocating memory on GPU" << endl;
	
	//allocate memory for the arrays 
	cudaMalloc( (void**) &da, memSize);
	cudaMalloc( (void**) &db, memSize);
	cudaMalloc( (void**) &dc, memSize);

	cout << "transfering memory to gpu" << endl;

	//transfer to GPU 
	cudaMemcpy(da, a,size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b,size, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(4,4);
	dim3 dimBlock(size, size);
	
	cout << "executing multiplication " <<endl;
	
	MatrixMultGPU<<<1, dimBlock>>>(da, db, dc, size);
	
	cudaDeviceSynchronize();
	
	cout << "Transfer from gpu to cpu" << endl;
	
	cudaMemcpy(c, dc , size, cudaMemcpyHostToDevice);

	cout << "GPU A" << endl;
	print1DArray(a, size);
	cout << "GPU B" << endl;
	print1DArray(b, size);
	cout << "GPU results c" << endl;
	print1DArray(c, size);
	
	free(a); free(b); free(c);
	cudaFree(da);cudaFree(db);cudaFree(dc);

}

int main(int argc, char** argv) {
	
	//creating 2d matrixes
	
	if (argc <= 1){
		cout << "Enter size of array as an argument" << endl;
		exit(0);
	}
	
	string s = argv[1];
	
	int size = stoi(s);
	
	int** a = initialize2DArray(size);
	int** b = initialize2DArray(size); 
	int** c = initialize2DArray(size); //has to be of size row * row
	
	random_ints(a, size, 73);
	random_ints(b, size, 11);
	
	matrixMultCPU(a,b,c,size);
	cout << "CPU calculations" << endl;
	cout << "A" << endl;
	printArray(a, size);
	cout << "B" << endl;
	printArray(b, size);
	cout << "Result C" << endl;
	printArray(c, size);
	
	//now time to multiply a and b with cuda
	executeCudaCalculations(a, b, size); 
	
	delete2DArray (a, size);
	delete2DArray (b, size);
	delete2DArray (c, size);
	
	
	
}
