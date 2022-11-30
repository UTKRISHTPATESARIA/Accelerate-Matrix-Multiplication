// Optimize this function
#include <immintrin.h>
#include <math.h>

void get_tranpose(int *new_matrix, int *old_matrix, int size)
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

void cache_transpose(int rb, int re, int cb, int ce, int size, int *matrix, int *result)
{
  int r = re - rb, c = ce - cb;
  if (r <= 16 && c <= 16)
  {
    for (int i = rb; i < re; i++)
    {
      for (int j = cb; j < ce; j++)
      {
        result[j * size + i] = matrix[i * size + j];
      }
    }
  }
  else if (r >= c)
  {
    cache_transpose(rb, rb + (r / 2), cb, ce, size, matrix, result);
    cache_transpose(rb + (r / 2), re, cb, ce, size, matrix, result);
  }
  else
  {
    cache_transpose(rb, re, cb, cb + (c / 2), size, matrix, result);
    cache_transpose(rb, re, cb + (c / 2), ce, size, matrix, result);
  }
}

void addVectorizedNormal(int *output, int *data, int size, int half_size, int shift_factor, int iterations)
{
  int afirst_row_factor, asecond_row_factor, shift_factor_plus_one, output_factor;
  shift_factor_plus_one = shift_factor + 1;
  for (int i = 0; i < half_size; ++i)
  {
    afirst_row_factor = (i << shift_factor_plus_one);
    asecond_row_factor = (afirst_row_factor + size);
    output_factor = (i << shift_factor);
    for (int k = 0; k < iterations; ++k)
    {
      __m256i A_first = _mm256_loadu_si256((__m256i *)&data[afirst_row_factor + (k << 3)]);
      __m256i A_second = _mm256_loadu_si256((__m256i *)&data[asecond_row_factor + (k << 3)]);
      __m256i A_result = _mm256_add_epi32(A_first, A_second);
      _mm256_storeu_si256((__m256i *)&output[output_factor + (k << 3)], A_result);
    }
  }
}

void singleThreadVectorizedOptimized(int size, int *matA, int *matB, int *output)
{
  int shift_factor, half_size, inner_iterations, sum, shift_factor_minus_one, output_shift_factor, a_row_factor, b_row_factor;
  half_size = (size >> 1);
  shift_factor = (int)(log10(size) / log10(2));
  shift_factor_minus_one = shift_factor - 1;
  inner_iterations = (size >> 3);
  int *matT = new int[size * size];
  if (size > 128)
  {
    get_tranpose(matT, matB, size);
  }
  else
  {
    cache_transpose(0, size, 0, size, size, matB, matT);
  }
  int *data = new int[1 << (shift_factor + shift_factor_minus_one)];
  int *data1 = new int[1 << (shift_factor + shift_factor_minus_one)];
  addVectorizedNormal(data, matA, size, half_size, shift_factor, inner_iterations);
  addVectorizedNormal(data1, matT, size, half_size, shift_factor, inner_iterations);
  for (int i = 0; i < half_size; ++i)
  {
    output_shift_factor = (i << shift_factor_minus_one);
    a_row_factor = (i << shift_factor);
    for (int j = 0; j < half_size; ++j)
    {
      b_row_factor = (j << shift_factor);
      sum = 0;
      for (int k = 0; k < inner_iterations; ++k)
      {
        __m256i A_first = _mm256_loadu_si256((__m256i *)&data[a_row_factor + (k << 3)]);
        __m256i B_first = _mm256_loadu_si256((__m256i *)&data1[b_row_factor + (k << 3)]);
        __m256i result = _mm256_mullo_epi32(A_first, B_first);

        sum += (_mm256_extract_epi32(result, 0) + _mm256_extract_epi32(result, 1) + _mm256_extract_epi32(result, 2) + _mm256_extract_epi32(result, 3) + _mm256_extract_epi32(result, 4) + _mm256_extract_epi32(result, 5) + _mm256_extract_epi32(result, 6) + _mm256_extract_epi32(result, 7));
      }
      output[output_shift_factor + j] = sum;
    }
  }
}

void singleThreadVectorized(int N, int *matA, int *matB, int *output)
{
  int *matT = new int[N * N];
  if (N > 128)
  {
    get_tranpose(matT, matB, N);
  }
  else
  {
    cache_transpose(0, N, 0, N, N, matB, matT);
  }
  int afirst_row_factor, bfirst_row_factor, asecond_row_factor, bsecond_row_factor, output_shift_factor, shift_factor, shift_factor_minus_one, shift_factor_plus_one, half_size = (N >> 1), inner_iterations = (N >> 3), sum = 0;
  shift_factor = (int)(log10(N) / log10(2));
  shift_factor_minus_one = shift_factor - 1;
  shift_factor_plus_one = shift_factor + 1;
  for (int i = 0; i < half_size; ++i)
  {
    output_shift_factor = (i << shift_factor_minus_one);
    afirst_row_factor = (i << shift_factor_plus_one);
    asecond_row_factor = (afirst_row_factor + N);
    for (int j = 0; j < half_size; ++j)
    {
      sum = 0;
      bfirst_row_factor = (j << shift_factor_plus_one);
      bsecond_row_factor = (bfirst_row_factor + N);
      for (int k = 0; k < inner_iterations; ++k)
      {
        __m256i A_first = _mm256_loadu_si256((__m256i *)&matA[afirst_row_factor + (k << 3)]);
        __m256i A_second = _mm256_loadu_si256((__m256i *)&matA[asecond_row_factor + (k << 3)]);
        __m256i B_first = _mm256_loadu_si256((__m256i *)&matT[bfirst_row_factor + (k << 3)]);
        __m256i B_second = _mm256_loadu_si256((__m256i *)&matT[bsecond_row_factor + (k << 3)]);
        __m256i result = _mm256_mullo_epi32(_mm256_add_epi32(A_first, A_second), _mm256_add_epi32(B_first, B_second));
        sum += (_mm256_extract_epi32(result, 0) + _mm256_extract_epi32(result, 1) + _mm256_extract_epi32(result, 2) + _mm256_extract_epi32(result, 3) + _mm256_extract_epi32(result, 4) + _mm256_extract_epi32(result, 5) + _mm256_extract_epi32(result, 6) + _mm256_extract_epi32(result, 7));
      }
      output[output_shift_factor + j] = sum;
    }
  }
}

void singleThreadNormal_ikj(int N, int *A, int *B, int *output)
{
  int shift_factor, shift_factor_minus_one, shift_factor_plus_one, half_size = (N >> 1);
  shift_factor = (int)(log10(N) / log10(2));
  shift_factor_minus_one = shift_factor - 1;
  shift_factor_plus_one = shift_factor + 1;
  for (int i = 0; i < half_size; ++i)
  {
    int a_first_shift_factor = (i << shift_factor_plus_one);
    int a_second_shift_factor = (((i << 1) + 1) << shift_factor);
    int output_shift_factor = (i << shift_factor_minus_one);
    for (int k = 0; k < N; ++k)
    {
      int a_first_index = a_first_shift_factor + k;
      int a_second_index = a_second_shift_factor + k;
      int b_first_shift_factor = (k << shift_factor);
      for (int j = 0; j < half_size; ++j)
      {
        output[output_shift_factor + j] += (A[a_first_index] + A[a_second_index]) * (B[b_first_shift_factor + (j << 1)] + B[b_first_shift_factor + (j << 1) + 1]);
      }
    }
  }
}

void singleThreadNormal_kij(int N, int *A, int *B, int *output)
{
  assert(N >= 4 and N == (N & ~(N - 1)));
  int shift_factor, shift_factor_minus_one, shift_factor_plus_one, half_size = N / 2;
  shift_factor = (int)(log10(N) / log10(2));
  shift_factor_minus_one = shift_factor - 1;
  shift_factor_plus_one = shift_factor + 1;
  for (int k = 0; k < N; ++k)
  {
    int b_first_index = (k << shift_factor);
    for (int i = 0; i < half_size; ++i)
    {
      int a_first_shift_factor = (i << shift_factor_plus_one);
      int a_second_shift_factor = (((i << 1) + 1) << shift_factor);
      int output_shift_factor = (i << shift_factor_minus_one);
      for (int j = 0; j < half_size; ++j)
      {
        output[output_shift_factor + j] += (A[a_first_shift_factor + k] + A[a_second_shift_factor + k]) * (B[b_first_index + (j << 1)] + B[b_first_index + ((j << 1) + 1)]);
      }
    }
  }
}

void singleThread(int N, int *matA, int *matB, int *output)
{
  assert(N >= 4 and N == (N & ~(N - 1)));
  singleThreadVectorizedOptimized(N, matA, matB, output);
}
