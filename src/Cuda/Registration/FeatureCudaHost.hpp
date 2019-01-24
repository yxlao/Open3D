//
// Created by wei on 1/23/19.
//

#include "FeatureCuda.h"

namespace open3d {
namespace cuda {

FeatureCuda::FeatureCuda() {
    Create();
}

FeatureCuda::~FeatureCuda() {
    Release();
}

FeatureCuda::FeatureCuda(const FeatureCuda &other) {
    server_ = other.server_;
    neighbors_ = other.neighbors_;
    spfh_features_ = other.spfh_features_;
    fpfh_features_ = other.fpfh_features_;
    pcl_ = other.pcl_;
}

FeatureCuda& FeatureCuda::operator=(const FeatureCuda &other) {
    if (this != &other) {

    }
}

void FeatureCuda::Create() {
    if (server_ != nullptr) {
        server_ = std::make_shared<FeatureCudaDevice>();
    }
}

void FeatureCuda::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        neighbors_.Release();
        spfh_features_.Release();
        fpfh_features_.Release();
    }

    server_ = nullptr;
}

void FeatureCuda::UpdateServer() {
    if (server_ != nullptr) {
        server_->neighbors_ = *neighbors_.server_;
        server_->spfh_features_ = *spfh_features_.server();
        server_->fpfh_features_ = *fpfh_features_.server();
    }
}

}
}