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

void NNCuda::NNSearch(Eigen::MatrixXd &query, Eigen::MatrixXd &reference) {

    /** Change storage format for Array2DCuda **/
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> query_rowmajor;
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> reference_rowmajor;
    query_rowmajor = query.cast<float>();
    reference_rowmajor = reference.cast<float>();

    query_.Upload(query_rowmajor);
    reference_.Upload(reference_rowmajor);

    nn_idx_.Create(1, query_.max_cols_);
    nn_dist_.Create(1, query_.max_cols_);
    distance_matrix_.Create(reference_.max_cols_, query_.max_cols_);

    UpdateServer();

    NNCudaKernelCaller::ComputeDistancesKernelCaller(*this);
    NNCudaKernelCaller::FindNNKernelCaller(*this);
}

void NNCuda::NNSearch(Array2DCuda<float> &query,
                      Array2DCuda<float> &reference) {
    query_ = query;
    reference_ = reference;

    nn_idx_.Create(1, query_.max_cols_);
    nn_dist_.Create(1, query_.max_cols_);
    distance_matrix_.Create(reference_.max_cols_, query_.max_cols_);

    UpdateServer();

    Timer timer;
    timer.Start();
    NNCudaKernelCaller::ComputeDistancesKernelCaller(*this);
    timer.Stop();
    PrintInfo("Compute takes %f ms\n", timer.GetDuration());

    timer.Start();
    NNCudaKernelCaller::FindNNKernelCaller(*this);
    timer.Stop();
    PrintInfo("FindNN takes %f ms\n", timer.GetDuration());
}
} // cuda
} // open3d