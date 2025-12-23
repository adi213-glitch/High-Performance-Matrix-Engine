
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional> // For passing functions
#include <immintrin.h>
#include <omp.h> //
/* generate 1024x1024 matrix with all cells set to double value 1.0 and return the matrix*/
std::vector<double>  generate_matrix(int N){
	std::vector<double> matrix (N*N,1);
	return matrix;
}

/* generate zero matrix */
std::vector<double> generate_zero_matrix(int N){
	std::vector<double> matrix (N*N,0);
	return matrix;
}

void naive_multiplication (int N, int BS,const std::vector<double> &A, const std::vector<double> &B, 
						std::vector<double>&C)
{
    //(ignore BS for now)
	// O(n^3) implementation 
	// implementing C = A.B
	for(int i = 0 ; i< N ; i++){
		for(int j = 0 ; j<N ; j++){
			for(int k = 0 ; k< N ; k++){
				C[N*i + j] += (A[N*i + k] * B[N*k +j]);
			}
		}
	}
}
void loop_reordered_multiplication (int N, int BS,const std::vector<double> &A, const std::vector<double> &B, 
						std::vector<double>&C)
{
    //(ignore BS for now)
	// O(n^3) implementation 
	// implementing C = A.B
    // BETTTER BECAUSE OF STRIDE-1 ACCESS
	for(int i = 0 ; i< N ; i++){
		for(int k = 0 ; k<N ; k++){
			
			double * C_ith_row = &C[N*i];
			const double * B_kth_row = &B[N*k];
			double A_element = A[N*i + k];
			for(int j = 0 ; j < N ; j++){
				C_ith_row[j] += (A_element * B_kth_row[j]);
			}
		}
	}
}
void loop_unrolled_multiplication (int N, int BS,const std::vector<double> &A, const std::vector<double> &B, 
						std::vector<double>&C)
{
    //(ignore BS for now)
	// O(n^3) implementation 
	// implementing C = A.B
	for(int i = 0 ; i< N ; i++){
		for(int k = 0 ; k<N ; k++){
			
			double * C_ith_row = &C[N*i];
			const double * B_kth_row = &B[N*k];
			double A_element = A[N*i + k];
			// 4x unroll
			int j;
			for(j = 0 ; j < N-3 ; j+=4){
				C_ith_row[j] += (A_element * B_kth_row[j]);
				C_ith_row[j+1] += (A_element * B_kth_row[j+1]);
				C_ith_row[j+2] += (A_element * B_kth_row[j+2]);
				C_ith_row[j+3] += (A_element * B_kth_row[j+3]);
			}
			// remaining elements (if any)
			for( ; j < N ; j++){
				C_ith_row[j] += (A_element * B_kth_row[j]);
			}
		}
	}
}
void cache_blocking_multiplication (int N, int BS, const std::vector<double> &A,  const std::vector<double> &B, 
						std::vector<double>&C)
{
	
	// number of blocks needed (Ceiling division)
    int num_blocks = (N + BS - 1) / BS;


	// OUTER LOOPS: Iterate through Block Coordinates (0, 1, 2...)
    for(int i = 0; i < num_blocks; i++) {
        for(int j = 0; j < num_blocks; j++) {
            for(int k = 0; k < num_blocks; k++) {
            


				// arithemtic on sub matrix 32x32 sized of A,B,C
				// uses maximum of 25% of L1 cache for each A,B,C (keeping remaining 25% for other tasks by Os, cpu)

				//CALCULATE LIMITS FOR THIS SPECIFIC BLOCK
                // Usually 32, but less if we are at the edge.
                int i_limit = std::min(BS, N - (i * BS)); 
                int k_limit = std::min(BS, N - (k * BS));
                int j_limit = std::min(BS, N - (j * BS));


				// INNER LOOPS: Run up to the calculated limit (not always 32)
				for( int p = 0 ; p < i_limit; p++){
					// (i*32 + p) th row of C
					double * C_row { &C[(i*BS + p)*N] };

					for(int r= 0 ; r <k_limit; r++){
						// (k*32 + r)th row of B;
						// (p*32 + r) element of C;
						const double * B_row { &B[(k*BS + r) * N] };
						const double A_element { A[(i*BS + p)*N + ((k*BS) + r ) ] };

						for(int q=0 ; q<j_limit; q++){
							//access required element of submatrix of B and C;
							C_row[j*BS + q] += (A_element * B_row[j*BS + q]);
						}
					}
				}
			}
		}
	}
}
// VECTOR CODE (AVX2 refinement) SIMD!!!!!!!

void SIMD_refined_multiplication (int N,int BS, const std::vector<double> &A,  const std::vector<double> &B,

                        std::vector<double>&C)

{
    
    // number of blocks needed (Ceiling division)
    int num_blocks = (N + BS - 1) / BS;

    // OUTER LOOPS: Iterate through Block Coordinates (0, 1, 2...)
    for(int i = 0; i < num_blocks; i++) {
        for(int j = 0; j < num_blocks; j++) {
            for(int k = 0; k < num_blocks; k++) {
            
                // arithemtic on sub matrix 32x32 sized of A,B,C
                // uses maximum of 25% of L1 cache for each A,B,C (keeping remaining 25% for other tasks by Os, cpu)
                //CALCULATE LIMITS FOR THIS SPECIFIC BLOCK
                // Usually 32, but less if we are at the edge.
                int i_limit = std::min(BS, N - (i * BS));
                int k_limit = std::min(BS, N - (k * BS));
                int j_limit = std::min(BS, N - (j * BS));
                // INNER LOOPS: Run up to the calculated limit (not always 32)
                for( int p = 0 ; p < i_limit; p++){
                    // (i*32 + p) th row of C
                    double * C_row { &C[(i*BS + p)*N] };
                    for(int r= 0 ; r <k_limit; r++){
                        // (k*32 + r)th row of B;
                        // (p*32 + r) element of C;
                        const double * B_row { &B[(k*BS + r) * N] };
                        const double A_element { A[(i*BS + p)*N + ((k*BS) + r ) ] };
                        // since A_ele is going to be repeated 4 times at once, broadcast it in a vector register
                        const __m256d vector_a { _mm256_set1_pd(A_element)};
                        for(int q=0 ; q<j_limit; q+=4){
                            //start pointer for C and B
                            double  * C_ptr { &C_row[j*BS + q]  };
                            const double  * B_ptr {&B_row[j*BS + q]};


                            // SAFETY CHECK: Ensure we don't go out of bounds if j_limit isn't divisible by 4.
                            // (Since BS=32, we are safe inside the block, but edge cases might need care.
                            // For this specific demo, we assume N is nice or we handle leftovers later).
                            if (q + 4 > j_limit) {
                                // Fallback to scalar for the last 1-3 elements

                                for (; q < j_limit; q++) {
                                    C_row[j*BS + q] += (A_element * B_row[j*BS + q]);
                                }
                                break;
							}


                            // create vector sized variable that holds 4doubles
                            __m256d vector_c { _mm256_loadu_pd(C_ptr)};
                            const __m256d vector_b { _mm256_loadu_pd(B_ptr)};
                            // computes  (vector_a * vector_b + vector_c) directly!!!!!
                            vector_c = _mm256_fmadd_pd(vector_a, vector_b, vector_c);
                            // write 4 doubles in register back to memory
                            _mm256_storeu_pd(C_ptr, vector_c);
                        }
                    }
                }
            }
        }
    }
}
// VECTOR CODE with loop unrolled
void SIMD_unrolled_multiplication (int N, int BS, const std::vector<double> &A, const std::vector<double> &B, 
						std::vector<double>&C)
{
	

	// number of blocks needed (Ceiling division)
    int num_blocks = (N + BS - 1) / BS;


	// OUTER LOOPS: Iterate through Block Coordinates (0, 1, 2...)
    for(int i = 0; i < num_blocks; i++) {
        for(int j = 0; j < num_blocks; j++) {
            for(int k = 0; k < num_blocks; k++) {

				// arithemtic on sub matrix 32x32 sized of A,B,C
				// uses maximum of 25% of L1 cache for each A,B,C (keeping remaining 25% for other tasks by Os, cpu)

				//CALCULATE LIMITS FOR THIS SPECIFIC BLOCK
                // Usually 32, but less if we are at the edge.
                int i_limit = std::min(BS, N - (i * BS)); 
                int k_limit = std::min(BS, N - (k * BS));
                int j_limit = std::min(BS, N - (j * BS));


				// INNER LOOPS: Run up to the calculated limit (not always 32)
				for( int p = 0 ; p < i_limit; p++){
					// (i*32 + p) th row of C
					double * C_row { &C[(i*BS + p)*N] };

					for(int r= 0 ; r <k_limit; r++){
						// (k*32 + r)th row of B;
						// (p*32 + r) element of C;
						const double * B_row { &B[(k*BS + r) * N] };
						const double A_element { A[(i*BS + p)*N + ((k*BS) + r ) ] };
						// since A_ele is going to be repeated 4 times at once, broadcast it in a vector register
						const __m256d vector_a { _mm256_set1_pd(A_element)};

						int q;
						// UNROLLED AVX LOOP (Process 16 doubles / 4 Vectors at a time)
                        // This breaks the dependency chain and saturates the FMA units
						for( q=0 ; q< j_limit - 15; q+=16){
							//start pointer for C and B
							double * C_ptr { &C_row[j*BS + q] };
                            const double * B_ptr { &B_row[j*BS + q] };

                            // Load 4 vectors from C
                            __m256d vector_c1 { _mm256_loadu_pd(C_ptr) };
                            __m256d vector_c2 { _mm256_loadu_pd(C_ptr + 4) };
                            __m256d vector_c3 { _mm256_loadu_pd(C_ptr + 8) };
                            __m256d vector_c4 { _mm256_loadu_pd(C_ptr + 12) };
                            
                            // Load 4 vectors from B
                            const __m256d vector_b1 { _mm256_loadu_pd(B_ptr) };
                            const __m256d vector_b2 { _mm256_loadu_pd(B_ptr + 4) };
                            const __m256d vector_b3 { _mm256_loadu_pd(B_ptr + 8) };
                            const __m256d vector_b4 { _mm256_loadu_pd(B_ptr + 12) };
                           
                            // Compute all 4 independent streams
                            vector_c1 = _mm256_fmadd_pd(vector_a, vector_b1, vector_c1);
                            vector_c2 = _mm256_fmadd_pd(vector_a, vector_b2, vector_c2);
                            vector_c3 = _mm256_fmadd_pd(vector_a, vector_b3, vector_c3);
                            vector_c4 = _mm256_fmadd_pd(vector_a, vector_b4, vector_c4);
                          
                            // Store all 4 back
                            _mm256_storeu_pd(C_ptr, vector_c1);
                            _mm256_storeu_pd(C_ptr + 4, vector_c2);
                            _mm256_storeu_pd(C_ptr + 8, vector_c3);
                            _mm256_storeu_pd(C_ptr + 12, vector_c4);
                           
                        }

                        // CLEANUP LOOP (Vector): Handle remaining chunks of 4
                        for(; q < j_limit - 3; q += 4) {
                            double * C_ptr { &C_row[j*BS + q] };
                            const double * B_ptr { &B_row[j*BS + q] };

                            __m256d vector_c { _mm256_loadu_pd(C_ptr) };
                            const __m256d vector_b { _mm256_loadu_pd(B_ptr) };
                            
                            vector_c = _mm256_fmadd_pd(vector_a, vector_b, vector_c);
                            
                            _mm256_storeu_pd(C_ptr, vector_c);
                        }

                        // CLEANUP LOOP (Scalar): Handle final 1-3 elements
                        for(; q < j_limit; q++) {
                            C_row[j*BS + q] += A_element * B_row[j*BS + q];
                        }
                    }
				}
			}
		}
	}
}
void cache_blocking_and_multithreaded_multiplication (int N,int BS, const std::vector<double> &A, const std::vector<double> &B, 
                        std::vector<double>&C)
{
	
	// number of blocks needed (Ceiling division)
    int num_blocks = (N + BS - 1) / BS;


	// "parallel for": Run loop in parallel
    // "collapse(2)": Parallelize both i and k loops for better load balancing
    #pragma omp parallel for collapse(2)

	// OUTER LOOPS: Iterate through Block Coordinates (0, 1, 2...)
    for(int i = 0; i < num_blocks; i++) {
        for(int j = 0; j < num_blocks; j++) {
            for(int k = 0; k < num_blocks; k++) {
            


				// arithemtic on sub matrix 32x32 sized of A,B,C
				// uses maximum of 25% of L1 cache for each A,B,C (keeping remaining 25% for other tasks by Os, cpu)

				//CALCULATE LIMITS FOR THIS SPECIFIC BLOCK
                // Usually 32, but less if we are at the edge.
                int i_limit = std::min(BS, N - (i * BS)); 
                int k_limit = std::min(BS, N - (k * BS));
                int j_limit = std::min(BS, N - (j * BS));


				// INNER LOOPS: Run up to the calculated limit (not always 32)
				for( int p = 0 ; p < i_limit; p++){
					// (i*32 + p) th row of C
					double * C_row { &C[(i*BS + p)*N] };

					for(int r= 0 ; r <k_limit; r++){
						// (k*32 + r)th row of B;
						// (p*32 + r) element of C;
						const double * B_row { &B[(k*BS + r) * N] };
						const double A_element { A[(i*BS + p)*N + ((k*BS) + r ) ] };

						for(int q=0 ; q<j_limit; q++){
							//access required element of submatrix of B and C;
							C_row[j*BS + q] += (A_element * B_row[j*BS + q]);
						}
					}
				}
			}
		}
	}
}
void ultimate_k_unrolled_multiplication(int N,int BS, const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    
    int num_blocks = (N + BS - 1) / BS;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < num_blocks; i++) {
        for (int j = 0; j < num_blocks; j++) {
            for (int k = 0; k < num_blocks; k++) {

                int i_limit = std::min(BS, N - (i * BS));
                int j_limit = std::min(BS, N - (j * BS));
                int k_limit = std::min(BS, N - (k * BS));

                const double* A_block_start = &A[0] + (i * BS * N + k * BS);
                const double* B_block_start = &B[0] + (k * BS * N + j * BS);
                double* C_block_start       = &C[0] + (i * BS * N + j * BS);

                for (int p = 0; p < i_limit; p++) {
                    double* C_row = C_block_start + p * N;
                    const double* A_row = A_block_start + p * N;

                    // --- K-UNROLLING OPTIMIZATION ---
                    // We process 4 'k' steps at once to minimize C load/stores.
                    int r = 0;
                    for (; r < k_limit - 3; r += 4) {
                        
                        // Pre-load 4 values of A into vectors (Broadcast)
                        __m256d vec_A0 = _mm256_set1_pd(A_row[r]);
                        __m256d vec_A1 = _mm256_set1_pd(A_row[r+1]);
                        __m256d vec_A2 = _mm256_set1_pd(A_row[r+2]);
                        __m256d vec_A3 = _mm256_set1_pd(A_row[r+3]);

                        // Pointers to the 4 rows of B we need
                        const double* B_row0 = B_block_start + r * N;
                        const double* B_row1 = B_block_start + (r+1) * N;
                        const double* B_row2 = B_block_start + (r+2) * N;
                        const double* B_row3 = B_block_start + (r+3) * N;

                        // Inner J-Loop (Vectorized)
                        int q = 0;
                        for (; q  < j_limit - 3; q += 4) {
                            
                            // LOAD C from memory ONLY ONCE
                            __m256d vec_C = _mm256_loadu_pd(&C_row[q]);

                            // load 4 B vectors and Perform 4 FMAs (Accumulate in Register)
                            __m256d vec_B0 = _mm256_loadu_pd(&B_row0[q]);
                            vec_C = _mm256_fmadd_pd(vec_A0, vec_B0, vec_C);

                            __m256d vec_B1 = _mm256_loadu_pd(&B_row1[q]);
                            vec_C = _mm256_fmadd_pd(vec_A1, vec_B1, vec_C);

                            __m256d vec_B2 = _mm256_loadu_pd(&B_row2[q]);
                            vec_C = _mm256_fmadd_pd(vec_A2, vec_B2, vec_C);

                            __m256d vec_B3 = _mm256_loadu_pd(&B_row3[q]);
                            vec_C = _mm256_fmadd_pd(vec_A3, vec_B3, vec_C);

                            // STORE C to memory ONLY ONCE
                            _mm256_storeu_pd(&C_row[q], vec_C);
                        }

                        // Cleanup for J (scalar)
                        for (; q < j_limit; q++) {
                            C_row[q] += A_row[r] * B_row0[q];
                            C_row[q] += A_row[r+1] * B_row1[q];
                            C_row[q] += A_row[r+2] * B_row2[q];
                            C_row[q] += A_row[r+3] * B_row3[q];
                        }
                    }

                    // Cleanup for K (If k_limit is not divisible by 4)
                    for (; r < k_limit; r++) {
                        const double* B_row = B_block_start + r * N;
                        double A_val = A_row[r];
                        __m256d vec_A = _mm256_set1_pd(A_val);
                        
                        int q = 0;
                        for (; q <= j_limit - 4; q += 4) {
                            __m256d vec_C = _mm256_loadu_pd(&C_row[q]);
                            __m256d vec_B = _mm256_loadu_pd(&B_row[q]);
                            vec_C = _mm256_fmadd_pd(vec_A, vec_B, vec_C);
                            _mm256_storeu_pd(&C_row[q], vec_C);
                        }
                        for (; q < j_limit; q++) {
                            C_row[q] += A_val * B_row[q];
                        }
                    }
                }
            }
        }
    }
}

struct Result {
    double time;
    double gflops;
};

// This function runs any multiplication kernel you pass to it
Result run_benchmark(const std::string& name, 
                     void (*func)(int, int , const std::vector<double>&, const std::vector<double>&, std::vector<double>&),
                     int N, 
                     int BS,
                     const std::vector<double>& A, 
                     const std::vector<double>& B, 
                     std::vector<double>& C) 
{
    // 1. Reset C
    std::fill(C.begin(), C.end(), 0.0);
    
    // 2. Warmup (optional, but good for stability)
    func(N,BS, A, B, C); 
    std::fill(C.begin(), C.end(), 0.0);

    // 3. Run Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    func(N, BS,A, B, C);
    auto end = std::chrono::high_resolution_clock::now();

    // 4. Calculate Metrics
    std::chrono::duration<double> diff = end - start;
    double seconds = diff.count();
    double operations = 2.0 * N * N * N; // 2N^3 FLOPs
    double gflops = (operations * 1e-9) / seconds;

    // 5. Verify (Just checking one cell is usually enough for a quick check)
    if (std::abs(C[0] - N) > 1e-9) {
        std::cout << "[FAILED] " << name << " produced incorrect math!" << std::endl;
        return {seconds, 0.0};
    }

    std::cout << std::left << std::setw(35) << name 
              << " | Time: " << std::fixed << std::setprecision(4) << seconds << " s"
              << " | GFLOPS: " << std::setprecision(2) << gflops << std::endl;

    return {seconds, gflops};
}

int main() {
    int N = 1024;
    int BS = 40;
    std::cout << "================================================================" << std::endl;
    std::cout << "   HIGH-PERFORMANCE MATRIX MULTIPLICATION BENCHMARK (N=" << N << ")" << std::endl;
    std::cout << "================================================================" << std::endl;

    // Generate Data
    std::vector<double> A = generate_matrix(N);
    std::vector<double> B = generate_matrix(N);
    std::vector<double> C = generate_zero_matrix(N);

    // 1. Run Baseline
    Result baseline = run_benchmark("1. Naive (O(n^3))", naive_multiplication, N, BS, A, B, C);

    // 2. Run Memory Optimization
    Result reordered = run_benchmark("2. Loop Reordered (Stride-1)", loop_reordered_multiplication, N,BS, A, B, C);

    // 3. Run Multithreading
    Result openmp = run_benchmark("3. OpenMP + Cache Blocking", cache_blocking_and_multithreaded_multiplication, N, BS,A, B, C);

    // 4. Run Ultimate
    Result ultimate = run_benchmark("4. AVX2 + Register Blocking", ultimate_k_unrolled_multiplication, N,BS, A, B, C);

    std::cout << "\n================================================================" << std::endl;
    std::cout << "   FINAL PERFORMANCE REPORT" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::left << std::setw(30) << "Version" 
              << std::setw(15) << "GFLOPS" 
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    auto print_row = [&](std::string name, double gflops) {
        double speedup = gflops / baseline.gflops;
        std::cout << std::left << std::setw(30) << name 
                  << std::setw(15) << gflops 
                  << std::setw(15) << std::to_string((int)speedup) + "x" << std::endl;
    };

    print_row("Naive", baseline.gflops);
    print_row("Loop Reordered", reordered.gflops);
    print_row("OpenMP + Blocking", openmp.gflops);
    print_row("AVX2 Ultimate", ultimate.gflops);
    
    std::cout << "================================================================" << std::endl;

    return 0;
}
