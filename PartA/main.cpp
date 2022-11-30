#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <fstream>
#include <assert.h>

using namespace std;

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(gran, start, end) std::chrono::duration_cast<gran>(end - start).count()

#include "single_thread.h"
#include "multi_thread.h"

// Used to cross-check answer. DO NOT MODIFY!
void reference(int N, int *matA, int *matB, int *output)
{
  // enforce N to be power of 2 and greater than 2
  assert( N>=4 and N == ( N &~ (N-1)));
  for(int rowA = 0; rowA < N; rowA +=2) {
    for(int colB = 0; colB < N; colB += 2){
      int sum = 0;
      for(int iter = 0; iter < N; iter++) 
      {
        sum += matA[rowA * N + iter] * matB[iter * N + colB];
        sum += matA[(rowA+1) * N + iter] * matB[iter * N + colB];
        sum += matA[rowA * N + iter] * matB[iter * N + (colB+1)];
        sum += matA[(rowA+1) * N + iter] * matB[iter * N + (colB+1)];
      }

      // compute output indices
      int rowC = rowA>>1;
      int colC = colB>>1;
      int indexC = rowC * (N>>1) + colC;
      output[indexC] = sum;
    }
  }
}

int main(int argc, char *argv[])
{
  // Input size of square matrices
  int N;
  string file_name; 
  if (argc < 2) 
    file_name = "data/input_8192.in"; 
  else 
    file_name = argv[1]; 
  ifstream input_file; 
  input_file.open(file_name); 
  input_file >> N;
  cout << "Input matrix of size " << N << "\n";

  // Input matrix A
  int *matA = new int[N * N];
  for(int i = 0; i < N; ++i)
    for(int j = 0; j < N; ++j)
      input_file >> matA[i * N + j];

  // Input matrix B
  int *matB = new int[N * N];
  for(int i = 0; i < N; ++i)
    for(int j = 0; j < N; ++j)
      input_file >> matB[i * N + j];

  // Untimed, warmup caches and TLB
  int *output_reference = new int[(N>>1)*(N>>1)];
  reference(N, matA, matB, output_reference);

  // Execute reference program
  auto begin = TIME_NOW;
  reference(N, matA, matB, output_reference);
  auto end = TIME_NOW;
  cout << "Reference execution time: " << 
    (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n";    

  /* if(N<=8){ */
  cerr << "matA: " << endl;
  for(int i=0; i<(N); i++){
    for(int j=0; j<N; j++){
      cerr << matA[i*N+j] << "\t";
    }
    cerr << endl;
  }
  cerr << "matB: " << endl;
  for(int i=0; i<(N); i++){
    for(int j=0; j<N; j++){
      cerr << matB[i*N+j] << "\t";
    }
    cerr << endl;
  }
  cerr << "output_reference: " << endl;
  for(int i=0; i<(N>>1); i++){
    for(int j=0; j<(N>>1); j++){
      cerr << output_reference[i*(N>>1)+j] << "\t";
    }
    cerr << endl;
  }
  /* } */

  // Execute single thread
  int *output_single = new int[(N>>1)*(N>>1)];
  begin = TIME_NOW;
  singleThread(N, matA, matB, output_single);
  end = TIME_NOW;
  cout << "Single thread execution time: " << 
    (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n";

  for(int i = 0; i < ((N>>1)*(N>>1)); ++i)
    if(output_single[i] != output_reference[i]) {
      cout << "Mismatch at " << i << "\n";
      exit(0);
    }

  // Execute multi-thread
  int *output_multi = new int[(N>>1)*(N>>1)];
  begin = TIME_NOW;
  multiThread(N, matA, matB, output_multi);
  end = TIME_NOW;
  cout << "Multi-threaded execution time: " << 
    (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n";

  for(int i = 0; i < ((N>>1)*(N>>1)); ++i)
    if(output_multi[i] != output_reference[i]) {
      cout << "Mismatch at " << i << "\n";
      exit(0);
    }

  input_file.close(); 
  return 0; 
}
