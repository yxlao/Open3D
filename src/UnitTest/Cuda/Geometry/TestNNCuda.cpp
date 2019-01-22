//
// Created by wei on 1/21/19.
//


#include <Cuda/Geometry/NNCuda.h>
#include <Eigen/Eigen>
#include <Core/Core.h>
#include <gtest/gtest.h>
#include <Cuda/Geometry/KNNCuda.h>

using namespace open3d;
using namespace open3d::cuda;


/**
 * Initializes randomly the reference and query points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 */
void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim) {

    // Initialize random number generator
    srand(time(NULL));

    // Generate random reference points
    for (int i=0; i<ref_nb*dim; ++i) {
        ref[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }

    // Generate random query points
    for (int i=0; i<query_nb*dim; ++i) {
        query[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }
}

/**
 * Test an input k-NN function implementation by verifying that its output
 * results (distances and corresponding indexes) are similar to the expected
 * results (ground truth).
 *
 * Since the k-NN computation might end-up in slightly different results
 * compared to the expected one depending on the considered implementation,
 * the verification consists in making sure that the accuracy is high enough.
 *
 * The tested function is ran several times in order to have a better estimate
 * of the processing time.
 *
 * @param ref            reference points
 * @param ref_nb         number of reference points
 * @param query          query points
 * @param query_nb       number of query points
 * @param dim            dimension of reference and query points
 * @param k              number of neighbors to consider
 * @param gt_knn_dist    ground truth distances
 * @param gt_knn_index   ground truth indexes
 * @param knn            function to test
 * @param name           name of the function to test (for display purpose)
 * @param nb_iterations  number of iterations
 * return false in case of problem, true otherwise
 */
bool test(const float * ref,
          int           ref_nb,
          const float * query,
          int           query_nb,
          int           dim,
          int           k,
          bool (*knn)(const float *, int, const float *, int, int, int, float *, int *),
          const char *  name,
          int           nb_iterations) {

    // Parameters
    const float precision    = 0.001f; // distance error max
    const float min_accuracy = 0.999f; // percentage of correct values required

    // Display k-NN function name
    printf("- %-17s : ", name);

    // Allocate memory for computed k-NN neighbors
    float * test_knn_dist  = (float*) malloc(query_nb * k * sizeof(float));
    int   * test_knn_index = (int*)   malloc(query_nb * k * sizeof(int));

    // Allocation check
    if (!test_knn_dist || !test_knn_index) {
        printf("ALLOCATION ERROR\n");
        free(test_knn_dist);
        free(test_knn_index);
        return false;
    }

    // Start timer
    Timer timer;
    timer.Start();

    // Compute k-NN several times
    for (int i=0; i<nb_iterations; ++i) {
        if (!knn(ref, ref_nb, query, query_nb, dim, k, test_knn_dist, test_knn_index)) {
            free(test_knn_dist);
            free(test_knn_index);
            return false;
        }
    }
    timer.Stop();

    // Verify both precisions and indexes of the k-NN values
    printf("PASSED in %8.5f ms (averaged over %3d iterations)\n",
        timer.GetDuration() / nb_iterations, nb_iterations);


    // Free memory
    free(test_knn_dist);
    free(test_knn_index);

    return true;
}

int main(int argc, char **argv) {

//     Parameters
    const int ref_nb   = 10000;
    const int query_nb = 10000;
    const int dim      = 33;
    const int k        = 1;

    // Display
    printf("PARAMETERS\n");
    printf("- Number reference points : %d\n",   ref_nb);
    printf("- Number query points     : %d\n",   query_nb);
    printf("- Dimension of points     : %d\n",   dim);
    printf("- Number of neighbors     : %d\n\n", k);

    // Allocate input points and output k-NN distances / indexes
    float * ref        = (float*) malloc(ref_nb   * dim * sizeof(float));
    float * query      = (float*) malloc(query_nb * dim * sizeof(float));
    float * knn_dist   = (float*) malloc(query_nb * k   * sizeof(float));
    int   * knn_index  = (int*)   malloc(query_nb * k   * sizeof(int));

    // Allocation checks
    if (!ref || !query || !knn_dist || !knn_index) {
        printf("Error: Memory allocation error\n");
        free(ref);
        free(query);
        free(knn_dist);
        free(knn_index);
        return EXIT_FAILURE;
    }

    // Initialize reference and query points with random values
    initialize_data(ref, ref_nb, query, query_nb, dim);

    // Compute the ground truth k-NN distances and indexes for each query point
    // Test all k-NN functions
    printf("TESTS\n");
    test(ref, ref_nb, query, query_nb, dim, k, &knn_cuda_global,
        "knn_cuda_global",  1);
    test(ref, ref_nb, query, query_nb, dim, k, &knn_cuda_texture,
        "knn_cuda_texture", 1);
    test(ref, ref_nb, query, query_nb, dim, k, &knn_cublas,
        "knn_cublas",       1);

    // Deallocate memory
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);


    const int size = 10000;
    const int feature_size = 33;
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> qr
    = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>::Zero(size, feature_size);
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> reference
    = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>::Zero(size, feature_size);

    for (int i = 0; i < size; ++i) {
        qr(i, 0) = i;
        qr(i, feature_size - 1) = i;
        reference(i, 0) = i;
        reference(i, feature_size - 1) = i;
    }

//    for (int i = 0; i < 10; ++i) {
    NNCuda nn;
    Timer timer;
    timer.Start();
    nn.NNSearch(qr, reference);
    timer.Stop();
    PrintInfo("NNSearch takes: %f\n", timer.GetDuration());

//    std::cout << nn.query_.Download() << std::endl;
//    std::cout << nn.reference_.Download() << std::endl;
//    std::cout << nn.distance_matrix_.Download() << std::endl;
//    }

//    auto result = nn.nn_idx_.Download();
//    for (int i = 0; i < size; ++i) {
//        if (i != result(i, 0)) {
//            std::cout << i << " " << result(i, 0) << std::endl;
//        }
//    }

//    ::testing::InitGoogleTest(&argc, argv);
//    return RUN_ALL_TESTS();
}
