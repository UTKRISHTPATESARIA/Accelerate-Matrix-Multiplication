#include <pthread.h>
#include <math.h>

#define MAX_T 256
int *matAG, *matBG, m_size;

typedef struct arg{
  int start_row;
  int end_row;
  int *out;
}args;


void get_transpose(int *new_matrix, int *old_matrix, int size)
{
  int diagnol_index, index2, length, shift_factor = (int)(log10(size) / log10(2)), index = 0;
  diagnol_index = (index << shift_factor) + index;
  while (index < size - 1)
  {
    length = size - index;
    index2 = index * (size + 1);
    for (int i = 1; i < length; ++i)
    {
      new_matrix[index2 + (i << shift_factor)] = old_matrix[i + index2];
      new_matrix[i + index2] = old_matrix[index2 + (i << shift_factor)];
    }
    new_matrix[diagnol_index] = old_matrix[diagnol_index];
    ++index;
    diagnol_index = (index << shift_factor) + index;
  }
  new_matrix[diagnol_index] = old_matrix[diagnol_index];
}

void *multiply(void *arg)
{
    args* temp_args = (args *)arg;
    int row_start = temp_args -> start_row;
    int row_end = temp_args -> end_row;
    int sum = 0;
    int pos = 0;
    temp_args -> out = new int[(row_end - row_start) * (m_size >> 1)];

    int row_seek = 0;

    for(int i = row_start; i < row_end; i++){
      for(int j = 0; j < (m_size >> 1); j++){
        for(int k = 0; k < m_size; k++)
          sum += matAG[i * m_size + k] * matBG[j * m_size + k];
        temp_args -> out[pos++] = sum;
        sum = 0;
      }
    }   
    pthread_exit(NULL);
}

void multiThread(int N, int *matA, int *mat_B, int *output)
{
    assert( N>=4 and N == ( N &~ (N-1)));

    m_size = N;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    pthread_t thread_pool[MAX_T];
    args args_pool[MAX_T];

    matAG = matA;
    int *matB = new int[N * N];
    get_transpose(matB, mat_B, N);
    matBG = matB;

    // reduce matrix size;
    for(int i = 0; i < N; i += 2){
      for(int j = 0; j < N; j++){
        matAG[(i >> 1)*N + j] = matAG[i*N + j] + matAG[(i + 1)*N + j];
        matBG[(i >> 1)*N + j] = matBG[i*N + j] + matBG[(i + 1)*N + j];
      }
    }
    int N1 = N >> 1;    
    int n_split = N1 < MAX_T ? N1 : MAX_T;
    int n_work = N1 < MAX_T ? 1 : N1 / MAX_T;

    for (int i = 0; i < n_split; i++) {
          args args_1;
          // calculate the starting and ending index of the row
          args_1.start_row = i * n_work;
          args_1.end_row = args_1.start_row + n_work;
          args_pool[i] = args_1;
          // create a thread, store in the thread pool, and assign its work
          // by argments pool
          pthread_create(&thread_pool[i], &attr, multiply, (void*) &args_pool[i]);
      }

      int pos = 0;

      for(int i = 0; i < n_split; i++){

          pthread_join(thread_pool[i], NULL);
          int seek = 0;
          args t = args_pool[i];
          int size = (t.end_row - t.start_row) * N1;
          while(seek < size){
            output[pos++] = t.out[seek++];
          }
      }
      pthread_attr_destroy(&attr);
}
