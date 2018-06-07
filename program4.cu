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

/*
HOW TO COMPILE: nvcc program4.cu -std=c++11
HOW TO RUN: ./a.out
*/

__global__ void mykernel(void){}

//generates random integers
//seed helps generate the seed
void random_ints(int **arr, int size, int seed)
{
	srand(time(NULL) * seed);

	for (int r = 0; r < size; ++r)
	{
   		for (int c = 0; c < size; ++c)
   		{
    		arr[r][c] = (rand() % 10);
    	}
	}
}


//returns 2D array of size, size *size
int** initialize2DArray(int size)
{
	int **arr = new int*[size];
	for(int i = 0; i < size; ++i)
	{
    	arr[i] = new int[size];
	}
	return arr;
}

//deletes arr of size* size
void delete2DArray (int** arr, int size)
{
	for(int i = 0; i < size; ++i)
	{
    	delete [] arr[i];
	}
	delete [] arr;
}

//used to print a given 1D array
void print1DArray(int * arr, int size)
{
	for (int r = 0; r < size; r++)
	{
		for (int c = 0; c < size; c++)
		{
			cout << setw(4) << arr[r * size + c];
		}
		cout << endl;
	}
}

//used to print a given 2D array
void printArray(int** arr, int size)
{
	for (int r = 0; r < size; r++)
	{
		for (int c = 0; c < size; c++)
		{
			cout << setw(4) << arr[r][c];// << " ";
		}
		cout << endl;
	}
}

//Exectutes matrix multiplication on a 2D array using CPU
//1 thread only
//returns time it took to do the computation
double matrixMultCPU (int** a, int** b, int** c, int size)
{
    clock_t cpuStart = clock();
    for (int rc = 0; rc < size; rc++) //row of array c
    {
        for (int cc = 0; cc < size; cc++)//column of array c, size of row
        {
            int total = 0; 
            for (int i = 0; i < size; i++)
            {
                int curA = a[rc][i];
                int curB = b[i][cc];
                total += curA * curB;
            }
            c[rc][cc] = total;      
        } 
    }
    
    clock_t cpuEnd = clock();
    double cpuSecs = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
	int operations = 2 * size * size * size;
    double flops = operations / cpuSecs;
	
	//output cpu stats
    cout << "---------------------------------------------------------------------------------------------" << endl;
    cout << "CPU SIZE: " << size << " by " << size << endl << endl;
    cout << "AREA: " << size * size << endl << endl;
    cout << "CPU time for matrix multiplication (sec): " << fixed << cpuSecs << endl;
    cout << "Num Operations (2 * size ^ 3)" << operations << endl;
	cout << "Flop time for CPU calculation (2 * size^3 / sec): " << fixed <<  flops << endl << endl;
	
	/*
    cout << "CPU A" << endl;
    printArray(a, size);
    cout << endl;
    cout << "CPU B" << endl;
    printArray(b, size);
    cout << endl;
    cout << "CPU Result C" << endl;
    printArray(c, size);
    cout << endl;
	*/

    return 0.0;
}

//computes matrix multiplication on 1D arrays a and b and puts the results into c
__global__ void MatrixMultGPU (int* a, int* b, int* c, int size, int tile_size)
{
	//gets the ID of the current thread in the block
	int x = blockIdx.x * tile_size + threadIdx.x; 
	int y = blockIdx.y * tile_size + threadIdx.y; 
	
	int product = 0;
	
	for (int i = 0; i < size; i++)
	{
		//get coordinates in array a
		int coorA = y * size + i;
		//get coordinates in array b 
		int coorB = i * size + x;
		//set product to what u got 
		product += a[coorA] * b[coorB];
	} 
	c[y * size + x] = product;
}

//copies the int** in to int* out 
void copyArrToArr(int** in, int* out, int size)
{
	for (int r = 0; r < size; r++)
	{
		for (int c = 0; c < size; c++)
		{
			out[r * size + c] = in[r][c];
		}
	}
}

//conducts matrix mult with inA and inB matrixes
void executeCudaCalculations(int** inA, int** inB, int size, int tile_size)
{
	//create arrays 
	int* a;
	int* b;
	int* c; //array to hold c, which is result of
	
	//copying CPU arrays to local CPU 1D arrays
	int memSize = size * size * sizeof(int); //technically 1D array which is size * size
	
	//allocating memory on GPU
	a = (int *)malloc(memSize); 
	b = (int *)malloc(memSize); 
	c = (int *)malloc(memSize); 
	
	//copies inA and inB to a and b
	copyArrToArr(inA, a, size); 
	copyArrToArr(inB, b, size);

	//device variables	
	int* da;
	int* db;
	int* dc;
	
	clock_t start = clock();
	//allocate memory for the arrays 
	cudaMalloc( (void**) &da, memSize);
	cudaMalloc( (void**) &db, memSize);
	cudaMalloc( (void**) &dc, memSize);

	//transfer to GPU 
	cudaMemcpy(da, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, memSize, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(size / tile_size, size / tile_size);
	dim3 dimBlock(tile_size, tile_size);

	clock_t gpuStart = clock();
	MatrixMultGPU<<<dimGrid, dimBlock>>>(da, db, dc, size, tile_size);
	cudaDeviceSynchronize();
	clock_t gpuEnd = clock();
	
	//transfer from gpu to cpu	
	cudaMemcpy(c, dc , memSize, cudaMemcpyDeviceToHost);
	clock_t end = clock();
	
	//printing out GPU naive statistics 
    double gpuSecs = (double)(gpuEnd - gpuStart) / CLOCKS_PER_SEC;
    double secs = (double)(end - start) / CLOCKS_PER_SEC;
    int operations = 2 * size * size * size;
    double flops = operations / gpuSecs;
    cout << "---------------------------------------------------------------------------------------------" << endl;
    cout << "DUMB GPU SIZE: " << size << " by " << size << endl << endl;
    cout << "AREA: " << size * size << endl;
    cout << "BLOCKS: " << (size / tile_size) * (size / tile_size) << endl;
    cout << "THREADS/BLOCK: " << tile_size * tile_size << endl;
    cout << "TILE_SIZE: " << tile_size << endl << endl;
	cout << "Total time including CPU to DEVICE and vice versa (sec)" << secs << endl << endl;
	cout << "Only GPU calculation time (sec): " << gpuSecs << endl;
	cout << "Num Operations (2 * size ^ 3)" << operations << endl;
	cout << "Number of flops (2 * size ^ 3/ gpuSecs): " << flops << endl << endl;
	
	/*
    cout << "GPU A" << endl;
    print1DArray(a, size);
    cout << endl;
    cout << "GPU B" << endl;
    print1DArray(b, size);
    cout << endl;
    cout << "GPU Results C" << endl;
    print1DArray(c, size);
    cout << endl;
	*/
	
    free(a); free(b); free(c);
    cudaFree(da);cudaFree(db);cudaFree(dc);

}

//computes matrix multiplication on 1D arrays a and b and puts the results into c
__global__ void blockMM (int* a, int* b, int* c, int size, int tile_size)
{
	int bx = blockIdx.x;
	int by = blockIdx.y; 
	int tx = threadIdx.x;
	int ty = threadIdx.y;
 
	//shared memory for submatrix A
	//need to dynamically allocate shared memory 
	extern __shared__ int temp[]; //used as temporary allocated memory for the two arrays
	int* As = &temp[0]; //point to first half of temp
	int* Bs = &temp[tile_size * tile_size]; //point to second half of temp

	//identify row and column of c element to work on
	int row = by * tile_size + ty;
	int col = bx * tile_size + tx;

	int sum = 0;
	int totalFlop = 0;
	//loop to compute tiles required
	for (int i = 0; i < size / tile_size; i++) //size / BLOCK_SIZE
	{
		//load matrix from global to shared memory
		int aIndex = row * size + i * tile_size + threadIdx.x;
		int bIndex = (i * tile_size + threadIdx.y) * size + col;
		//each thread loads 1 element of each element
		As[ty * tile_size + tx] = a[aIndex];
		Bs[ty * tile_size + tx] = b[bIndex];
		
		//synchronize
		__syncthreads();
		
		//multiply submatrix
		//each thread computes 1 element of csub
		for (int k = 0; k < tile_size; k++)
		{
			sum += As[ty * tile_size + k] * Bs[k * tile_size + tx]; 
			totalFlop ++;
		}  
		//sync
		__syncthreads();
	}
	
	//write result to global memory
	c[row * size + col] = sum;
}

//conducts matrix mult with blocks
void executeBlockMM(int** inA, int** inB, int size, int tile_size)
{
	//create arrays 
	int* a;
	int* b;
	int* c; //array to hold c, which is result of
	
	//copying CPU arrays to local CPU 1D arrays
	int memSize = size * size * sizeof(int); //technically 1D array which is size * size
	
	//allocating memory on GPU
	a = (int *)malloc(memSize); 
	b = (int *)malloc(memSize); 
	c = (int *)malloc(memSize); 
	
	//copies inA and inB to a and b
	copyArrToArr(inA, a, size); 
	copyArrToArr(inB, b, size);
	
	//device variables	
	int* da;
	int* db;
	int* dc;
	
	clock_t start = clock();
	//allocate memory for the arrays 
	cudaMalloc( (void**) &da, memSize);
	cudaMalloc( (void**) &db, memSize);
	cudaMalloc( (void**) &dc, memSize);

	//transfer to GPU 
	cudaMemcpy(da, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, memSize, cudaMemcpyHostToDevice);

	dim3 dimBlock(tile_size, tile_size);
	dim3 dimGrid(size / tile_size, size / tile_size);

	clock_t gpuStart = clock();
	blockMM<<<dimGrid, dimBlock, tile_size * tile_size * sizeof(int) * 2>>>(da, db, dc, size, tile_size);
	cudaDeviceSynchronize();
	clock_t gpuEnd = clock();
	
	//transfer from gpu to cpu
	cudaMemcpy(c, dc , memSize, cudaMemcpyDeviceToHost);
	clock_t end = clock();
	
	//printing out GPU smart statistics 
	double gpuSecs = (double)(gpuEnd - gpuStart) / CLOCKS_PER_SEC;
	double secs = (double)(end - start) / CLOCKS_PER_SEC;
    int operations = 2 * size * size * size;
    double flops = operations / gpuSecs;
	cout << "---------------------------------------------------------------------------------------------" << endl;
	cout << "SMART GPU SIZE: " <<  size << " by " << size << endl << endl;
	cout << "TILE SIZE: " << tile_size << endl << endl;
	cout << "AREA: " << size * size << endl;
	cout << "BLOCKS: " << (size / tile_size) * (size / tile_size) << endl;
	cout << "THREADS/BLOCK: " << tile_size * tile_size << endl;
	cout << "Total time including CPU to DEVICE and vice versa (sec)" << secs << endl << endl;
	cout << "Only GPU calculation time (sec): " << gpuSecs << endl;
	cout << "Num Operations (2 * size ^ 3)" << operations << endl;
	cout << "Number of flops (2 * size ^ 3/ gpuSecs): " << flops << endl << endl;
	
	/*
	cout << "GPU A" << endl;
	print1DArray(a, size);
	cout << endl;
	cout << "GPU B" << endl;
	print1DArray(b, size);
	cout << endl;
	cout << "GPU Results C" << endl;
	print1DArray(c, size);
	cout << endl;
	*/
	
	free(a); free(b); free(c);
	cudaFree(da);cudaFree(db);cudaFree(dc);
}

int main()
{	
	
	int tile_size = 0;
	int size = 0;
	
	//Arrays used for calculations
	int** a = initialize2DArray(size);
	int** b = initialize2DArray(size); 
	int** c = initialize2DArray(size); //has to be of size row * row
	
	cout << "---------------------------------------------------------------------------------------------" << endl;
	cout << "---------------------------------------------------------------------------------------------" << endl;
	cout << "Below are results for CPU, Naive GPU and Smart GPU where" << endl;
	cout << "Size is 8 to 48 with increments of 8 and tiles size are all 4x4" << endl;	
	for (int i = 8; i <= 48; i += 8)
	{
		
		tile_size = 2;
		size = i;

		a = initialize2DArray(size);
		b = initialize2DArray(size); 		
		c = initialize2DArray(size); //has to be of size row * row
	
		random_ints(a, size, 73);
		random_ints(b, size, 11);
	
		matrixMultCPU(a, b, c, size);
		executeCudaCalculations(a, b, size, tile_size);
		executeBlockMM(a, b, size, tile_size);
	
		delete2DArray(a, size);
		delete2DArray(b, size);
		delete2DArray(c, size);
	} 
	
	cout << "---------------------------------------------------------------------------------------------" << endl;
	cout << "---------------------------------------------------------------------------------------------" << endl;
	cout << "Below are results for Naive GPU and Smart GPU where" << endl; //num operations is #threads * 
	cout << "Size is 8 to 64 with increments of 8 and tiles size are all (size / 4) + ( size / 4)" << endl;	

	for (int i = 8; i <= 64; i += 8)
	{
		tile_size = 4; //16 threads total;
		size = i;
		
		a = initialize2DArray(size);
		b = initialize2DArray(size);
	
		random_ints(a, size, 73);
		random_ints(b, size, 11);
	
		executeCudaCalculations(a, b, size, tile_size);
		executeBlockMM(a, b, size, tile_size);
	
		delete2DArray(a, size);
		delete2DArray(b, size);
	}
	cout << "---------------------------------------------------------------------------------------------" << endl;
	cout << "---------------------------------------------------------------------------------------------" << endl;
	cout << "Below are results for CPU, Naive GPU and Smart GPU where" << endl;
	cout << "Size is 32 to 128 with increments of 32 and tiles size are all the max 32 * 32 " << endl;	

	for (int i = 32; i <= 128; i += 32)
	{
		
		tile_size = 32;
		size = i;

		a = initialize2DArray(size);
		b = initialize2DArray(size);
		c = initialize2DArray(size); //has to be of size row * row
	
		random_ints(a, size, 73);
		random_ints(b, size, 11);
	
		matrixMultCPU(a, b, c, size);
		executeCudaCalculations(a, b, size, tile_size);
		executeBlockMM(a, b, size, tile_size);
	
		delete2DArray(a, size);
		delete2DArray(b, size);
		delete2DArray(c, size);
	}
}