//
// Created by wei on 1/23/19.
//

//
// Created by wei on 1/21/19.
//

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>
#include <Core/Registration/FastGlobalRegistration.h>
#include <Core/Registration/ColoredICP.h>
#include <iostream>
#include <Cuda/Geometry/NNCuda.h>
#include <Cuda/Registration/FastGlobalRegistrationCuda.h>
#include "ReadDataAssociation.h"

using namespace open3d;
std::shared_ptr<Feature> PreprocessPointCloud(PointCloud &pcd) {
    EstimateNormals(pcd, open3d::KDTreeSearchParamHybrid(0.1, 30));
    auto pcd_fpfh = ComputeFPFHFeature(
        pcd, open3d::KDTreeSearchParamHybrid(0.25, 100));
    return pcd_fpfh;
}

std::shared_ptr<open3d::RGBDImage> ReadRGBDImage(
    const char *color_filename, const char *depth_filename,
    const open3d::PinholeCameraIntrinsic &intrinsic,
    bool visualize) {
    open3d::Image color, depth;
    ReadImage(color_filename, color);
    ReadImage(depth_filename, depth);
    open3d::PrintDebug("Reading RGBD image : \n");
    open3d::PrintDebug("     Color : %d x %d x %d (%d bits per channel)\n",
                       color.width_, color.height_,
                       color.num_of_channels_, color.bytes_per_channel_ * 8);
    open3d::PrintDebug("     Depth : %d x %d x %d (%d bits per channel)\n",
                       depth.width_, depth.height_,
                       depth.num_of_channels_, depth.bytes_per_channel_ * 8);
    double depth_scale = 1000.0, depth_trunc = 4.0;
    bool convert_rgb_to_intensity = true;
    std::shared_ptr<open3d::RGBDImage> rgbd_image =
        CreateRGBDImageFromColorAndDepth(color,
                                         depth,
                                         depth_scale,
                                         depth_trunc,
                                         convert_rgb_to_intensity);
    if (visualize) {
        auto pcd = CreatePointCloudFromRGBDImage(*rgbd_image, intrinsic);
        open3d::DrawGeometries({pcd});
    }
    return rgbd_image;
}

void VisualizeRegistration(const open3d::PointCloud &source,
                           const open3d::PointCloud &target,
                           const Eigen::Matrix4d &Transformation) {
    using namespace open3d;
    std::shared_ptr<PointCloud> source_transformed_ptr(new PointCloud);
    std::shared_ptr<PointCloud> target_ptr(new PointCloud);
    *source_transformed_ptr = source;
    *target_ptr = target;
    source_transformed_ptr->Transform(Transformation);
    DrawGeometries({source_transformed_ptr, target_ptr}, "Registration result");
}


std::vector<std::pair<int, int>> AdvancedMatching(
    const std::vector<PointCloud>& point_cloud_vec,
    const std::vector<Feature>& features_vec,
    const FastGlobalRegistrationOption& option)
{
    // STEP 0) Swap source and target if necessary
    int fi = 0, fj = 1;
    PrintDebug("Advanced matching : [%d - %d]\n", fi, fj);
    bool swapped = false;
    if (point_cloud_vec[fj].points_.size() >
        point_cloud_vec[fi].points_.size())
    {
        int temp = fi;
        fi = fj;
        fj = temp;
        swapped = true;
    }

    Timer timer;
    timer.Start();
    // STEP 1) Initial matching
    int nPti = int(point_cloud_vec[fi].points_.size());
    int nPtj = int(point_cloud_vec[fj].points_.size());
    KDTreeFlann feature_tree_i(features_vec[fi]);
    KDTreeFlann feature_tree_j(features_vec[fj]);
    std::vector<int> corresK;
    std::vector<double> dis;
    std::vector<std::pair<int, int>> corres;
    std::vector<std::pair<int, int>> corres_ij;
    std::vector<std::pair<int, int>> corres_ji;
    std::vector<int> i_to_j(nPti, -1);
    for (int j = 0; j < nPtj; j++) {
        feature_tree_i.SearchKNN(Eigen::VectorXd(features_vec[fj].data_.col(j)),
                                 1, corresK, dis);
        int i = corresK[0];
        if (i_to_j[i] == -1) {
            feature_tree_j.SearchKNN(
                Eigen::VectorXd(features_vec[fi].data_.col(i)),
                1, corresK, dis);
            int ij = corresK[0];
            i_to_j[i] = ij;
        }
        corres_ji.push_back(std::pair<int, int>(i, j));
    }
    for (int i = 0; i < nPti; i++) {
        if (i_to_j[i] != -1)
            corres_ij.push_back(std::pair<int, int>(i, i_to_j[i]));
    }
    int ncorres_ij = int(corres_ij.size());
    int ncorres_ji = int(corres_ji.size());
    for (int i = 0; i < ncorres_ij; ++i)
        corres.push_back(std::pair<int, int>(
            corres_ij[i].first, corres_ij[i].second));
    for (int j = 0; j < ncorres_ji; ++j)
        corres.push_back(std::pair<int, int>(
            corres_ji[j].first, corres_ji[j].second));
    timer.Stop();
    PrintDebug("points are remained : %d in %f ms\n", (int)corres.size(),
               timer.GetDuration());


    // STEP 2) CROSS CHECK
    timer.Start();
    PrintDebug("\t[cross check] ");
    std::vector<std::pair<int, int>> corres_cross;
    std::vector<std::vector<int>> Mi(nPti), Mj(nPtj);
    int ci, cj;
    for (int i = 0; i < ncorres_ij; ++i)
    {
        ci = corres_ij[i].first;
        cj = corres_ij[i].second;
        Mi[ci].push_back(cj);
    }
    for (int j = 0; j < ncorres_ji; ++j)
    {
        ci = corres_ji[j].first;
        cj = corres_ji[j].second;
        Mj[cj].push_back(ci);
    }
    for (int i = 0; i < nPti; ++i) {
        for (int ii = 0; ii < Mi[i].size(); ++ii) {
            int j = Mi[i][ii];
            for (int jj = 0; jj < Mj[j].size(); ++jj) {
                if (Mj[j][jj] == i) {
                    corres_cross.push_back(std::pair<int, int>(i, j));
                    PrintInfo("%d %d\n", i, j);
                }
            }
        }
    }
    timer.Stop();
    PrintDebug("points are remained : %d in %f ms\n", (int)corres_cross.size(),
               timer.GetDuration());

    // STEP 3) TUPLE CONSTRAINT
    timer.Start();
    PrintDebug("\t[tuple constraint] ");
    std::srand((unsigned int)std::time(0));
    int rand0, rand1, rand2, i, cnt = 0;
    int idi0, idi1, idi2, idj0, idj1, idj2;
    double scale = option.tuple_scale_;
    int ncorr = static_cast<int>(corres_cross.size());
    int number_of_trial = ncorr * 100;
    std::vector<std::pair<int, int>> corres_tuple;
    for (i = 0; i < number_of_trial; i++) {
        rand0 = rand() % ncorr;
        rand1 = rand() % ncorr;
        rand2 = rand() % ncorr;
        idi0 = corres_cross[rand0].first;
        idj0 = corres_cross[rand0].second;
        idi1 = corres_cross[rand1].first;
        idj1 = corres_cross[rand1].second;
        idi2 = corres_cross[rand2].first;
        idj2 = corres_cross[rand2].second;

        // collect 3 points from i-th fragment
        Eigen::Vector3d pti0 = point_cloud_vec[fi].points_[idi0];
        Eigen::Vector3d pti1 = point_cloud_vec[fi].points_[idi1];
        Eigen::Vector3d pti2 = point_cloud_vec[fi].points_[idi2];
        double li0 = (pti0 - pti1).norm();
        double li1 = (pti1 - pti2).norm();
        double li2 = (pti2 - pti0).norm();

        // collect 3 points from j-th fragment
        Eigen::Vector3d ptj0 = point_cloud_vec[fj].points_[idj0];
        Eigen::Vector3d ptj1 = point_cloud_vec[fj].points_[idj1];
        Eigen::Vector3d ptj2 = point_cloud_vec[fj].points_[idj2];
        double lj0 = (ptj0 - ptj1).norm();
        double lj1 = (ptj1 - ptj2).norm();
        double lj2 = (ptj2 - ptj0).norm();

        // check tuple constraint
        if ((li0 * scale < lj0) && (lj0 < li0 / scale) &&
            (li1 * scale < lj1) && (lj1 < li1 / scale) &&
            (li2 * scale < lj2) && (lj2 < li2 / scale)) {
            corres_tuple.push_back(std::pair<int, int>(idi0, idj0));
            corres_tuple.push_back(std::pair<int, int>(idi1, idj1));
            corres_tuple.push_back(std::pair<int, int>(idi2, idj2));
            cnt++;
        }
        if (cnt >= option.maximum_tuple_count_)
            break;
    }
    timer.Stop();
    PrintDebug("%d tuples (%d trial, %d actual) in %f ms.\n", cnt,
               number_of_trial, i, timer.GetDuration());

    for (int i = 0; i < corres_tuple.size(); ++i) {
        PrintInfo("(%d %d)\n", corres_tuple[i].second, corres_tuple[i].first);
    }

    if (swapped) {
        std::vector<std::pair<int, int>> temp;
        for (int i = 0; i < corres_tuple.size(); i++)
            temp.push_back(std::pair<int, int>
                               (corres_tuple[i].second, corres_tuple[i].first));
        corres_tuple.clear();
        corres_tuple = temp;
    }
    PrintDebug("\t[final] matches %d.\n", (int)corres_tuple.size());
    return corres_tuple;
}

// Normalize scale of points. X' = (X-\mu)/scale
std::tuple<std::vector<Eigen::Vector3d>, double, double> NormalizePointCloud(
    std::vector<PointCloud>& point_cloud_vec,
    const FastGlobalRegistrationOption& option)
{
    int num = 2;
    double scale = 0;
    std::vector<Eigen::Vector3d> pcd_mean_vec;
    double scale_global, scale_start;

    for (int i = 0; i < num; ++i) {
        double max_scale = 0.0;
        Eigen::Vector3d mean;
        mean.setZero();

        int npti = static_cast<int>(point_cloud_vec[i].points_.size());
        for (int ii = 0; ii < npti; ++ii)
            mean = mean + point_cloud_vec[i].points_[ii];
        mean = mean / npti;
        pcd_mean_vec.push_back(mean);

        PrintDebug("normalize points :: mean = [%f %f %f]\n",
                   mean(0), mean(1), mean(2));
        for (int ii = 0; ii < npti; ++ii)
            point_cloud_vec[i].points_[ii] -= mean;

        for (int ii = 0; ii < npti; ++ii) {
            Eigen::Vector3d p(point_cloud_vec[i].points_[ii]);
            double temp = p.norm();
            if (temp > max_scale)
                max_scale = temp;
        }
        if (max_scale > scale)
            scale = max_scale;
    }

    if (option.use_absolute_scale_) {
        scale_global = 1.0;
        scale_start = scale;
    } else {
        scale_global = scale;
        scale_start = 1.0;
    }
    PrintDebug("normalize points :: global scale : %f\n", scale_global);

    for (int i = 0; i < num; ++i) {
        int npti = static_cast<int>(point_cloud_vec[i].points_.size());
        for (int ii = 0; ii < npti; ++ii) {
            point_cloud_vec[i].points_[ii] /= scale_global;
        }
    }
    return std::make_tuple(pcd_mean_vec, scale_global, scale_start);
}

Eigen::Matrix4d OptimizePairwiseRegistration(
    const std::vector<PointCloud>& point_cloud_vec,
    const std::vector<std::pair<int, int>>& corres, double scale_start,
    const FastGlobalRegistrationOption& option)
{
    PrintDebug("Pairwise rigid pose optimization\n");
    double par = scale_start;
    int numIter = option.iteration_number_;

    int i = 0, j = 1;
    PointCloud point_cloud_copy_j = point_cloud_vec[j];

    if (corres.size() < 10)
        return Eigen::Matrix4d::Identity();

    std::vector<double> s(corres.size(), 1.0);
    Eigen::Matrix4d trans;
    trans.setIdentity();

    for (int itr = 0; itr < numIter; itr++) {
        const int nvariable = 6;
        Eigen::MatrixXd JTJ(nvariable, nvariable);
        Eigen::MatrixXd JTr(nvariable, 1);
        Eigen::MatrixXd J(nvariable, 1);
        JTJ.setZero();
        JTr.setZero();
        double r = 0.0, r2 = 0.0;

        for (int c = 0; c < corres.size(); c++) {
            int ii = corres[c].first;
            int jj = corres[c].second;
            Eigen::Vector3d p, q;
            p = point_cloud_vec[i].points_[ii];
            q = point_cloud_copy_j.points_[jj];
            Eigen::Vector3d rpq = p - q;

            int c2 = c;
            double temp = par / (rpq.dot(rpq) + par);
            s[c2] = temp * temp;

            Eigen::Vector6d Jtr_private;
            float r2_private = 0;
            Jtr_private.setZero();

            J.setZero();
            J(1) = -q(2);
            J(2) = q(1);
            J(3) = -1;
            r = rpq(0);
            JTJ += J * J.transpose() * s[c2];
            JTr += J * r * s[c2];
            r2 += r * r * s[c2];

            Jtr_private += J * r * s[c2];
            r2_private += r * r * s[c2];

            J.setZero();
            J(2) = -q(0);
            J(0) = q(2);
            J(4) = -1;
            r = rpq(1);
            JTJ += J * J.transpose() * s[c2];
            JTr += J * r * s[c2];
            r2 += r * r * s[c2];

            Jtr_private += J * r * s[c2];
            r2_private += r * r * s[c2];

            J.setZero();
            J(0) = -q(1);
            J(1) = q(0);
            J(5) = -1;
            r = rpq(2);
            JTJ += J * J.transpose() * s[c2];
            JTr += J * r * s[c2];
            r2 += r * r * s[c2];
            r2 += (par * (1.0 - sqrt(s[c2])) * (1.0 - sqrt(s[c2])));

            Jtr_private += J * r * s[c2];
            r2_private += r * r * s[c2];
            r2_private += (par * (1.0 - sqrt(s[c2])) * (1.0 - sqrt(s[c2])));

//            if (itr == 0) {
//                printf("cpu- (%d %d): (%f %f %f %f %f %f), %f\n",
//                       ii, jj,
//                       Jtr_private(0), Jtr_private(1), Jtr_private(2),
//                       Jtr_private(3), Jtr_private(4), Jtr_private(5),
//                       r2_private);
//            }
        }
        bool success;
        Eigen::VectorXd result;
        std::tie(success, result) = SolveLinearSystem(-JTJ, JTr);
        Eigen::Matrix4d delta = TransformVector6dToMatrix4d(result);
        trans = delta * trans;
        point_cloud_copy_j.Transform(delta);
        if (itr < 2) {
            std::cout << "cpu JtJ: " << JTJ << std::endl;
            std::cout << "cpu Jtr: " << JTr.transpose() << std::endl;
            std::cout << "cpu rmse: " << r2 << std::endl;
            std::cout << "cpu delta: " << delta << std::endl;
            std::cout << "cpu trans: " << trans <<
                      std::endl;
        }

        // graduated non-convexity.
//        std::cout << "decrease_mu_: " << option.decrease_mu_ << std::endl;
        if (option.decrease_mu_) {
//            PrintInfo("%f %f\n",
//                option.maximum_correspondence_distance_,
//                option.division_factor_);
            if (itr % 4 == 0 && par > option.maximum_correspondence_distance_) {
                par /= option.division_factor_;
                std::cout << "cpu par: " << par << std::endl;
            }
        }
    }
    return trans;
}

// Below line indicates how the transformation matrix aligns two point clouds
// e.g. T * point_cloud_vec[1] is aligned with point_cloud_vec[0].
Eigen::Matrix4d GetTransformationOriginalScale(
    const Eigen::Matrix4d& transformation,
    const std::vector<Eigen::Vector3d>& pcd_mean_vec,
    const double scale_global)
{
    Eigen::Matrix3d R = transformation.block<3, 3>(0, 0);
    Eigen::Vector3d t = transformation.block<3, 1>(0, 3);
    Eigen::Matrix4d transtemp = Eigen::Matrix4d::Zero();
    transtemp.block<3, 3>(0, 0) = R;
    transtemp.block<3, 1>(0, 3) =
        -R*pcd_mean_vec[1] + t*scale_global + pcd_mean_vec[0];
    transtemp(3, 3) = 1;
    return transtemp;
}


int main(int argc, char **argv) {
    open3d::SetVerbosityLevel(open3d::VerbosityLevel::VerboseDebug);

    std::string filepath = "/home/wei/Work/data/stanford/lounge/fragments";
    auto source_origin = CreatePointCloudFromFile(
        filepath + "/fragment_000.ply");
    auto target_origin = CreatePointCloudFromFile(
        filepath + "/fragment_003.ply");
    auto source = open3d::VoxelDownSample(*source_origin, 0.05);
    auto target = open3d::VoxelDownSample(*target_origin, 0.05);

    EstimateNormals(*source, open3d::KDTreeSearchParamHybrid(0.1, 30));
    EstimateNormals(*target, open3d::KDTreeSearchParamHybrid(0.1, 30));

    open3d::Timer timer;
    FastGlobalRegistrationOption option;
    for (int i = 0; i < 10; ++i) {
        timer.Start();
        open3d::cuda::FastGlobalRegistrationCuda fgr;
        fgr.Initialize(*source, *target);

        open3d::cuda::RegistrationResultCuda result;
        for (int iter = 0; iter < 64; ++iter) {
            result = fgr.DoSingleIteration(iter);
        }
        timer.Stop();
        PrintInfo("gpu: %f ms\n", timer.GetDuration());

        timer.Start();
        auto source_features = ComputeFPFHFeature(
            *source, open3d::KDTreeSearchParamHybrid(0.25, 100));
        auto target_features = ComputeFPFHFeature(
            *target, open3d::KDTreeSearchParamHybrid(0.25, 100));
        FastGlobalRegistration(
            *source, *target, *source_features, *target_features);
        timer.Stop();
        PrintInfo("cpu: %f ms\n", timer.GetDuration());


        VisualizeRegistration(*source, *target, result.transformation_);

//        auto res = RegistrationColoredICP(*source, *target, 0.07, result.transformation_);
//        VisualizeRegistration(*source, *target, res.transformation_);
//        auto corres_final = fgr.corres_final_.Download();
//
//
//        std::vector<PointCloud> point_cloud_vec;
//        point_cloud_vec.push_back(*source);
//        point_cloud_vec.push_back(*target);
//
//        double scale_global, scale_start;
//        std::vector<Eigen::Vector3d> pcd_mean_vec;
//        std::tie(pcd_mean_vec, scale_global, scale_start) =
//            NormalizePointCloud(point_cloud_vec, option);
//        std::vector<std::pair<int, int>> corres;
//
//        for (int i = 0; i < corres_final.size(); ++i) {
//            open3d::cuda::Vector2i cfi = corres_final[i];
//            corres.push_back(std::make_pair(cfi(0), cfi(1)));
//        }
//        Eigen::Matrix4d transformation;
//        transformation = OptimizePairwiseRegistration(
//            point_cloud_vec, corres, scale_global, option);
//        auto trans = GetTransformationOriginalScale(
//            transformation, pcd_mean_vec, scale_global).inverse();


//        open3d::cuda::RegistrationResultCuda result;
//        for (int iter = 0; iter < 64; ++iter) {
//            result = fgr.DoSingleIteration(iter);
//        }
//        std::cout << result.transformation_ << std::endl;
//        VisualizeRegistration(*source, *target, trans);

//        auto mutual = fgr.corres_mutual_.Download();
//        auto final = fgr.corres_final_.Download();
//
//        for (auto &f : final) {
//            PrintInfo("(%d %d)\n", f(0), f(1));
//        }
//
//        PrintInfo("Initialize takes %f ms\n", timer.GetDuration());
//        PrintInfo("Reciprocity :%d, final: %d\n", mutual.size(), final.size());

    }

//    std::string base_path = "/home/wei/Work/data/stanford/lounge/";
//    auto rgbd_filenames = ReadDataAssociation(
//        base_path + "data_association.txt");
//
//    open3d::Image source_color, source_depth, target_color, target_depth;
//    open3d::PinholeCameraIntrinsic intrinsic = open3d::PinholeCameraIntrinsic(
//        open3d::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
//    auto rgbd_source = ReadRGBDImage(
//        (base_path + "/" + rgbd_filenames[3].second).c_str(),
//        (base_path + "/" + rgbd_filenames[3].first).c_str(),
//        intrinsic,
//        false);
//    auto rgbd_target = ReadRGBDImage(
//        (base_path + "/" + rgbd_filenames[18].second).c_str(),
//        (base_path + "/" + rgbd_filenames[18].first).c_str(),
//        intrinsic,
//        false);
//
//    auto source_origin = CreatePointCloudFromRGBDImage(*rgbd_source, intrinsic);
//    auto target_origin = CreatePointCloudFromRGBDImage(*rgbd_target, intrinsic);
//    open3d::EstimateNormals(*source_origin);
//    open3d::EstimateNormals(*target_origin);
//
//    std::string filepath = "/home/wei/Work/data/stanford/lounge/fragments";
//    source_origin = CreatePointCloudFromFile(filepath + "/fragment_003.ply");
//    target_origin = CreatePointCloudFromFile(filepath + "/fragment_012.ply");
//
//    auto source = open3d::VoxelDownSample(*source_origin, 0.05);
//    auto target = open3d::VoxelDownSample(*target_origin, 0.05);
//
//    Timer timer;
//    timer.Start();
//    auto source_feature = PreprocessPointCloud(*source);
//    auto target_feature = PreprocessPointCloud(*target);
//    timer.Stop();
//    PrintInfo("Feature extraction takes %f ms\n", timer.GetDuration());
//
//    KDTreeFlann source_feature_tree(*source_feature);
//    KDTreeFlann target_feature_tree(*target_feature);
//
//    for (int i = 0; i < 100; ++i) {
//        std::vector<int> correspondences;
//        std::vector<int> indices(1);
//        std::vector<double> dist(1);
//        timer.Start();
//        for (int i = 0; i < source_feature->Num(); ++i) {
//            target_feature_tree.SearchKNN(
//                Eigen::VectorXd(source_feature->data_.col(i)),
//                1,
//                indices,
//                dist);
//            correspondences.push_back(indices.size() > 0 ? indices[0] : -1);
//        }
//        timer.Stop();
//        PrintInfo("cpu knn takes %f ms\n", timer.GetDuration());
//
//        timer.Start();
//        open3d::cuda::NNCuda nn;
//        nn.NNSearch(source_feature->data_, target_feature->data_);
//        timer.Stop();
//        PrintInfo("cuda knn takes %f ms\n", timer.GetDuration());
//
//        auto correspondences_cuda = nn.nn_idx_.Download();
//        int inconsistency = 0;
//        for (int i = 0; i < source_feature->Num(); ++i) {
//            int correspondence_cpu = correspondences[i];
//            int correspondence_cuda = correspondences_cuda(0, i);
//            if (correspondence_cpu != correspondence_cuda) {
//                ++inconsistency;
//            }
//        }
//        PrintInfo("Consistency: %f (%d / %d)\n",
//            1 - (float) inconsistency / source_feature->Num(),
//            source_feature->Num() - inconsistency, source_feature->Num());
//    }
//    return 0;
}