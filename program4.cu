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

void printArray(int** arr, int size){

	for (int r = 0; r < size; r++){
	
		for (int c = 0; c < size; c++){
			
			cout << setw(4) << arr[r][c] << " ";
				
		}
		cout << endl;
	}

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
	int** b = initialize2DArray(size); //switch them around so you can multiply together properly
	int** c = initialize2DArray(size); //has to be of size row * row
	
	random_ints(a, size, 73);
	random_ints(b, size, 11);
	
	matrixMultCPU(a,b,c,size);
	
	cout << "A" << endl;
	printArray(a, size);
	cout << "B" << endl;
	printArray(b, size);
	cout << "Result C" << endl;
	printArray(c, size);
	
	delete2DArray (a, size);
	delete2DArray (b, size);
	delete2DArray (c, size);

}
