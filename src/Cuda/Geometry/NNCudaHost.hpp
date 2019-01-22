//
// Created by wei on 1/21/19.
//

#include "NNCuda.h"

namespace open3d {
namespace cuda {

NNCuda::NNCuda() {
    server_ = std::make_shared<NNCudaDevice>();
}

NNCuda::~NNCuda() {
    server_ = nullptr;
}

void NNCuda::UpdateServer() {
    if (server_ != nullptr) {
        server_->query_ = *query_.server();
        server_->ref_ = *reference_.server();
        server_->nn_idx_ = *nn_idx_.server();
        server_->nn_dist_ = *nn_dist_.server();
        server_->distance_matrix_ = *distance_matrix_.server();
    }
}

void NNCuda::NNSearch(
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> &query,
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> &reference) {

    query_.Upload(query);
    reference_.Upload(reference);

    nn_idx_.Create(query_.max_rows_, 256);
    nn_dist_.Create(query_.max_rows_, 256);
    distance_matrix_.Create(query_.max_rows_, reference_.max_rows_);

    UpdateServer();

    Timer timer;
    timer.Start();
    NNCudaKernelCaller::ComputeAndReduceDistancesKernelCaller(*this);
    timer.Stop();
    PrintInfo("Compute takes %f ms\n", timer.GetDuration());

    timer.Start();
    NNCudaKernelCaller::ReduceBlockwiseDistancesKernelCaller(*this);
    timer.Stop();
    PrintInfo("Reduce takes %f ms\n", timer.GetDuration());
}
} // cuda
} // open3d