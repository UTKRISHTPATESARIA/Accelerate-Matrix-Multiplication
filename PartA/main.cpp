#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <fstream>
#include <assert.h>

#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

#include "header/single_thread.h"
#include "header/multi_thread.h"

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(gran, start, end) std::chrono::duration_cast<gran>(end - start).count()
// Adding logic for perf calculation. ---------------------------------------------------------------------------------------

char ename[6][50]={"CPU CYCLES","L1D LOAD MISSES","L1I LOAD MISSES","LL LOAD MISSES","LL STORE MISSES","PAGE FAULTS"};

struct perf_event_attr pe[6];
long long count;
int fd[6];

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
     int ret;
     ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,group_fd, flags);
     return ret;
}

void init_perf(){

    for(int ptr=0;ptr<6;ptr++)
    {
      memset(&(pe[ptr]), 0, sizeof(pe[0]));
      pe[ptr].size = sizeof(pe[0]);       
    } 

    pe[0].type = PERF_TYPE_HARDWARE;
    pe[1].type = PERF_TYPE_HW_CACHE;
    pe[2].type = PERF_TYPE_HW_CACHE;
    pe[3].type = PERF_TYPE_HW_CACHE;
    pe[4].type = PERF_TYPE_HW_CACHE;
    pe[5].type = PERF_TYPE_SOFTWARE;  //SOFTWARE TYPE EVENT
    //-------------------------------CPU CYCLES---------
    pe[0].config =PERF_COUNT_HW_CPU_CYCLES; 

    //---------------------------------L1 CACHE-----------------
    pe[1].config =(PERF_COUNT_HW_CACHE_L1D) |(PERF_COUNT_HW_CACHE_OP_READ << 8) |(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    //pe[6].config =(PERF_COUNT_HW_CACHE_L1D) |(PERF_COUNT_HW_CACHE_OP_WRITE<< 8) |(PERF_COUNT_HW_CACHE_RESULT_MISS<< 16);
    pe[2].config =(PERF_COUNT_HW_CACHE_L1I) |(PERF_COUNT_HW_CACHE_OP_READ << 8) |(PERF_COUNT_HW_CACHE_RESULT_MISS<< 16); 
    //pe[8].config =(PERF_COUNT_HW_CACHE_L1I) |(PERF_COUNT_HW_CACHE_OP_WRITE << 8) |(PERF_COUNT_HW_CACHE_RESULT_MISS<< 16);
    //--------------------------------LL CACHE-----------------------
    pe[3].config =(PERF_COUNT_HW_CACHE_LL) |(PERF_COUNT_HW_CACHE_OP_READ << 8) |(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    pe[4].config =(PERF_COUNT_HW_CACHE_LL) |(PERF_COUNT_HW_CACHE_OP_WRITE << 8) |(PERF_COUNT_HW_CACHE_RESULT_MISS<< 16);
    //------------------------------- PAGE FAULT
    pe[5].config =PERF_COUNT_SW_PAGE_FAULTS;

    for(int ptr=0;ptr<6;ptr++)
      {
        pe[ptr].disabled = 1;
        pe[ptr].exclude_kernel = 1;
        pe[ptr].exclude_hv = 1;
        fd[ptr] = perf_event_open(pe+ptr, 0, -1, -1, 0);
        if (fd[ptr] == -1) 
              {
                printf("%d\n",ptr);
                printf("Error opening leader %llx\n", pe[ptr].config);
                      exit(EXIT_FAILURE);
              }
      } 
}

void perf_start(){
  //-----------------------------------------EVENT RESETTING----------------
  ioctl(fd[0], PERF_EVENT_IOC_RESET, 0);
  ioctl(fd[1], PERF_EVENT_IOC_RESET, 0);	
  ioctl(fd[2], PERF_EVENT_IOC_RESET, 0);	
  ioctl(fd[3], PERF_EVENT_IOC_RESET, 0);	
  ioctl(fd[4], PERF_EVENT_IOC_RESET, 0);	
  ioctl(fd[5], PERF_EVENT_IOC_RESET, 0);		
  //-----------------------------------------ENABLING EVENT COUNTING----------------	
  ioctl(fd[0], PERF_EVENT_IOC_ENABLE, 0);
  ioctl(fd[1], PERF_EVENT_IOC_ENABLE, 0);
  ioctl(fd[2], PERF_EVENT_IOC_ENABLE, 0);
  ioctl(fd[3], PERF_EVENT_IOC_ENABLE, 0);
  ioctl(fd[4], PERF_EVENT_IOC_ENABLE, 0);
  ioctl(fd[5], PERF_EVENT_IOC_ENABLE, 0);
}

void perf_stop(){

  ioctl(fd[0], PERF_EVENT_IOC_DISABLE, 0);
  ioctl(fd[1], PERF_EVENT_IOC_DISABLE, 0);
  ioctl(fd[2], PERF_EVENT_IOC_DISABLE, 0);
  ioctl(fd[3], PERF_EVENT_IOC_DISABLE, 0);
  ioctl(fd[4], PERF_EVENT_IOC_DISABLE, 0);
  ioctl(fd[5], PERF_EVENT_IOC_DISABLE, 0);
}

void write_(std::ofstream& out_perf){

   for(int ptr=0;ptr<6;ptr++)
        {
        	read(fd[ptr], &count, sizeof(count));
          out_perf <<count << "\t" << ename[ptr] << endl;
        	close(fd[ptr]);
        }
}

// ------------------------------------------------------------------- Perf Ends 
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
 
  int N = atoi(argv[1]);
  int i = atoi(argv[2]);

        srand(i);
        ofstream output_file; 
        string file_name = "data/input_" + to_string(N) + ".in";  
        output_file.open(file_name); 
        output_file << N << "\n"; 
        // Generate matrix A
        for(int i = 0; i < N; ++i) {
            for(int j = 0; j < N; ++j)
                output_file << rand() % 256 << " ";
            output_file << "\n";
        }
        
        // Generate matrix B
        for(int i = 0; i < N; ++i) {
            for(int j = 0; j < N; ++j)
                output_file << rand() % 256 << " ";
            output_file << "\n";
        }
        output_file.close();

        ifstream input_file; 
        input_file.open(file_name); 

        int *matA = new int[N * N];
        for(int i = 0; i < N; ++i)
          for(int j = 0; j < N; ++j)
            input_file >> matA[i * N + j];

        // Input matrix B
        int *matB = new int[N * N];
        for(int i = 0; i < N; ++i)
          for(int j = 0; j < N; ++j)
            input_file >> matB[i * N + j];

        ofstream output_perf;
        string fn = "result-final/reference-"+to_string(N)+".txt";
        output_perf.open(fn, std::ios_base::app);
        init_perf();
        // Untimed, warmup caches and TLB
        int *output_reference = new int[(N>>1)*(N>>1)];
        perf_start();
        reference(N, matA, matB, output_reference);
        perf_stop();
        output_perf << "\n\n";
        write_(output_perf);
        output_perf.close();

        // Execute single thread type 0
        
        fn = "result-final/single-"+to_string(N)+"-0.txt";
        output_perf.open(fn, std::ios_base::app);
        int *output_single = new int[(N>>1)*(N>>1)];
        init_perf();
        perf_start();
        auto begin = TIME_NOW;
        singleThread(N, matA, matB, output_single, 0);
        auto end = TIME_NOW;
        perf_stop();
        write_(output_perf);
        output_perf << ((double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0) << "\tExecution time\n\n";
        output_perf.close();
        // Execute single thread type 3
        
        fn = "result-final/single-"+to_string(N)+"-3.txt";
        output_perf.open(fn, std::ios_base::app);
        init_perf();
        perf_start();
        begin = TIME_NOW;
        singleThread(N, matA, matB, output_single, 3);
        end = TIME_NOW;
        perf_stop();
        write_(output_perf);
        output_perf << ((double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0) << "\tExecution time\n\n";

        // Execute multi-thread 
        ofstream results; 
        fn = "result-final/multi-" + to_string(N) + ".txt";
        results.open(fn, std::ios_base::app);  
        int *output_multi = new int[(N>>1)*(N>>1)];

        results << "32 Thread-Size\n";
        init_perf();
        perf_start();
        begin = TIME_NOW;
        multiThread(N, matA, matB, output_multi, 32);
        end = TIME_NOW;
        perf_stop();
        write_(results);
        results << ((double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0) << "\tExecution time\n\n";
        
        results << "64 Thread-Size\n";
        init_perf();
        perf_start();
        begin = TIME_NOW;
        multiThread(N, matA, matB, output_multi, 64);
        end = TIME_NOW;
        perf_stop();
        write_(results);
        results << ((double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0) << "\tExecution time\n\n";
        
        results << "128 Thread-Size\n";
        init_perf();
        perf_start();
        begin = TIME_NOW;
        multiThread(N, matA, matB, output_multi, 128);
        end = TIME_NOW;
        perf_stop();
        write_(results);
        results << ((double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0) << "\tExecution time\n\n";

        results << "256 Thread-Size\n";
        init_perf();
        perf_start();
        begin = TIME_NOW;
        multiThread(N, matA, matB, output_multi, 256);
        end = TIME_NOW;
        perf_stop(); 
        write_(results);
        results << ((double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0) << "\tExecution time\n\n";
          input_file.close(); 

  
  
/*
  cerr << "multi_opt: " << endl;
  for(int i=0; i<(N>>1); i++){
    for(int j=0; j<(N>>1); j++){
      cerr << output_multi[i*(N>>1)+j] << "\t";
    }
    cerr << endl;
  }
  

  for(int i = 0; i < ((N>>1)*(N>>1)); ++i)
    if(output_multi[i] != output_reference[i]) {
      cout << "Mismatch at " << i << "\n";
      exit(0);
    }
  */
  return 0; 
}
