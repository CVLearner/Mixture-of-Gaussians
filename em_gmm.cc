#include "em_gmm.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <random>

//< Uncomment the line below to utilize Math Kernel Library
//#define EIGEN_USE_MKL_ALL
#include <Eigen/Eigen>

namespace {
    //< Constant for numerical stability
    const float eps_covariance = 1e-10;
    const float eps_zero = 1e-10;
    const float eps_log_negative_inf = -1e30;
    const float eps_convergence = 1e-4;
    const float eps_regularize = 1e-30;
}

using namespace std;
using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> RowMatrixXf;

inline float log_sum(const float& log_a, const float& log_b) {
    return log_a < log_b ?
         (log_b + std::log (1.0 + std::exp (log_a - log_b)))
         : (log_a + std::log (1.0 + std::exp (log_b - log_a)));
}

void calculate_log_prob_spherical(
        const RowMatrixXf& mat_data, 
        const VectorXf& vec_nrm2_pts,
        const RowVectorXf& vec_weights,
        const RowMatrixXf& mat_means,
        const RowMatrixXf& mat_diag_covs,
        RowMatrixXf& mat_log_probs) {

    const long num_modes = mat_means.rows();
    const long num_pts = mat_data.rows();
    const long dim = mat_data.cols();

    assert( vec_nrm2_pts.rows() == num_pts &&
            vec_weights.cols() == num_modes && 
            mat_means.cols() == dim &&
            mat_diag_covs.rows() == num_modes &&
            mat_diag_covs.cols() == dim &&
            mat_log_probs.rows() == num_pts && 
            mat_log_probs.cols() == num_modes );

    RowVectorXf vec_nrm2_centers(num_modes);
    for (long c = 0; c < num_modes; c++) {
        vec_nrm2_centers(c) = mat_means.row(c).squaredNorm();
    }

    mat_log_probs.noalias() = mat_data * mat_means.transpose();
    mat_log_probs *= -2;
    #pragma omp parallel for
    for (long c = 0; c < num_modes; c++) {
        mat_log_probs.col(c) += vec_nrm2_pts;
    }
    #pragma omp parallel for
    for (long n = 0; n < num_pts; n++) {
        mat_log_probs.row(n) += vec_nrm2_centers;
    }

    #pragma omp parallel for
    for (long c = 0; c < num_modes; c++) {
        const float cov = mat_diag_covs(c,0);
        const float c1 = log(vec_weights(c) + eps_regularize) 
            - 0.5*dim*log(2*M_PI) - 0.5*dim*log(cov); 
        const float c2 = -0.5f/cov;
        mat_log_probs.col(c) *= c2;
        mat_log_probs.col(c) = (mat_log_probs.col(c).array() + c1).matrix();
    } 
}

void calculate_log_prob_diagonal(
        const RowMatrixXf& mat_data, 
        const RowVectorXf& vec_weights,
        const RowMatrixXf& mat_means,
        const RowMatrixXf& mat_diag_covs,
        RowMatrixXf& mat_log_probs) {

    const long num_modes = mat_means.rows();
    const long num_pts = mat_data.rows();
    const long dim = mat_data.cols();

    assert( vec_weights.cols() == num_modes && 
            mat_means.cols() == dim &&
            mat_diag_covs.rows() == num_modes &&
            mat_diag_covs.cols() == dim &&
            mat_log_probs.rows() == num_pts && 
            mat_log_probs.cols() == num_modes);

    const float c0(-0.5f*dim*log(2*M_PI));
    RowVectorXf vec_c1_cov_prod(num_modes);
    for (long c = 0; c < num_modes; c++) {
        vec_c1_cov_prod(c) = -0.5f*(mat_diag_covs.row(c).array().log().sum());
    }

    #pragma omp parallel for
    for (long n = 0; n < num_pts; n++) {
        RowVectorXf vec_data = mat_data.row(n);
        for (long c = 0; c < num_modes; c++) {
            RowVectorXf delta = (vec_data - mat_means.row(c));
            mat_log_probs(n,c) = -0.5f*(delta.array()*mat_diag_covs.row(c).array().cwiseInverse()).matrix().dot(delta)
                + c0 + vec_c1_cov_prod(c);
        }
    }
}

void em_gmm(
        const float *data, 
        const long num_pts, 
        const long dim,
        const int num_modes,
        float *means, 
        float *diag_covs,
        float *weights,
        bool should_fit_spherical_gaussian) {

    using namespace std;

    assert (num_modes < num_pts && "Not enough data for em");

    RowVectorXf vec_eps_regularize(num_modes);
    vec_eps_regularize.fill(eps_regularize);

    //< K-means to initialize the EM
    std::vector<int> labels(num_pts, -1);

    Map<const RowMatrixXf> mat_data(data, num_pts, dim);
    Map<RowMatrixXf> mat_means(means, num_modes, dim);
    Map<RowMatrixXf> mat_diag_covs(diag_covs, num_modes, dim);
    Map<RowVectorXf> vec_weights(weights, num_modes);

    //< Random init
    random_device rd;
    default_random_engine gen(rd());
    uniform_int_distribution<long> kmeans_seed_dist(0, num_pts-1);
    vector<long> center_indices(num_modes);
    generate(center_indices.begin(), center_indices.end(), [&]{ 
        return kmeans_seed_dist(gen);
    });
    #pragma omp parallel for
    for (int c = 0; c < num_modes; c++) {
        long n = center_indices[c];
        mat_means.row(c) = mat_data.row(n);
    }

    //< kmeans convergence
    const int max_kmeans_iterations = 20;
    bool is_converged = false;
    float eps(0.0f);

    RowMatrixXf mat_distance(num_pts, num_modes);

    VectorXf vec_nrm2_pts(num_pts);
    #pragma omp parallel for
    for (long r = 0; r < num_pts; r++) {
        vec_nrm2_pts(r) = mat_data.row(r).squaredNorm();
    }

    RowVectorXf vec_nrm2_centers(num_modes);

    //< K-means 
    RowMatrixXf mat_saved_means(num_modes, dim);
    RowVectorXf assigned_counts(num_modes);

    int iterations = 0;
    while ((iterations++ < max_kmeans_iterations) && !is_converged) {

        //< save previous centers
        mat_saved_means = mat_means;

        //< calculate point to center L2 distance
        //< (X - C)^2 = X^2 + C^2 - 2*X*C
        #pragma omp parallel for
        for (int c = 0; c < num_modes; c++) {
            vec_nrm2_centers(c) = mat_means.row(c).squaredNorm();
        }
        mat_distance.noalias() = mat_data * mat_means.transpose();
        mat_distance *= -2;
        #pragma omp parallel for
        for (long c = 0; c < num_modes; c++) {
            mat_distance.col(c) += vec_nrm2_pts;
        }
        
        //< nearest centers along one column
        #pragma omp parallel for
        for (long n = 0; n < num_pts; n++) {
            mat_distance.row(n) += vec_nrm2_centers;
            mat_distance.row(n).minCoeff(&labels[n]);
        }

        assigned_counts.fill(0.0f);
        for (long n = 0; n < num_pts; n++) {
            long c = labels[n];
            if (assigned_counts(c) < 1e-3) {
                mat_means.row(c) = mat_data.row(n);
            } else {
                mat_means.row(c) += mat_data.row(n);
            }
            assigned_counts(c) += 1.0f;
        }

        #pragma omp parallel for
        for (long c = 0; c < num_modes; c++) {
            if (assigned_counts(c) > 1e-3) {
                mat_means.row(c) /= assigned_counts(c);
            } 
        }

        //< evaluation
        const float prev_eps = eps;
        eps = (mat_saved_means - mat_means).norm();
        is_converged = (eps < eps_convergence);
        cout << "kmeans " << "[" << iterations << "] " << prev_eps << " " << eps << endl;
    } 

    //< covariances and weights
    vec_weights = assigned_counts / (float)num_pts;
    mat_diag_covs.fill(0);
    for (long n = 0; n < num_pts; n++) {
        long c = labels[n];
        mat_diag_covs.row(c) += mat_data.row(n).array().square().matrix();
    }
    for (long c = 0; c < num_modes; c++) {
        if (assigned_counts(c) > 1e-3) {
            mat_diag_covs.row(c) /= assigned_counts(c);
            mat_diag_covs.row(c) -= mat_means.row(c).array().square().matrix();
        } 
    }

    if (should_fit_spherical_gaussian) {
        for (int c = 0; c < num_modes; c++) {
            const float spherical_val = std::max(mat_diag_covs.row(c).sum()/dim, eps_covariance);
            mat_diag_covs.row(c).fill(spherical_val);
        }
    }

    //< EM
    RowMatrixXf& mat_log_probs = mat_distance;

    const int max_em_iterations = 20;
    is_converged = false; //< use weights as approximated indicator
    float expectation(std::numeric_limits<float>::lowest());

    RowVectorXf vec_evals(num_pts);
    RowVectorXf vec_log_sum_probs(num_pts);
    RowVectorXf vec_occup_eN(num_modes);
    RowMatrixXf mat_occup_eX(num_modes, dim);
    RowMatrixXf mat_occup_eX2(num_modes, dim);

    iterations = 0;
    while ((iterations++ < max_em_iterations) && !is_converged) {

        //< calculate log probabilities
        if (should_fit_spherical_gaussian) {
            calculate_log_prob_spherical(mat_data, vec_nrm2_pts, vec_weights, mat_means, mat_diag_covs, mat_log_probs);
        } else {
            calculate_log_prob_diagonal(mat_data, vec_weights, mat_means, mat_diag_covs, mat_log_probs);
        }

        #pragma omp parallel for
        for (long n = 0; n < num_pts; n++) {
            vec_log_sum_probs(n) = mat_log_probs(n,0);
            for (long c = 1; c < num_modes; c++) {
                if (mat_log_probs(n,c) > eps_log_negative_inf) {
                    vec_log_sum_probs(n) = log_sum(vec_log_sum_probs(n), mat_log_probs(n,c));
                }
            }
        }

        #pragma omp parallel for
        for (long n = 0; n < num_pts; n++) {
            RowVectorXf soft_count = (mat_log_probs.row(n).array() - vec_log_sum_probs(n)).exp().matrix();
            vec_evals(n) = 0;
            for (long c = 0; c < num_modes; c++) {
                if (soft_count(c) > eps_zero) {
                    vec_evals(n) += mat_log_probs(n,c)*soft_count(c);
                }
            }
            mat_log_probs.row(n) = soft_count;
        }

        ////< Occupation counts
        #pragma omp parallel for
        for (long c = 0; c < num_modes; c++) {
            vec_occup_eN(c) = mat_log_probs.col(c).sum();
        }

        mat_occup_eX = mat_log_probs.transpose() * mat_data;
        //< M-Step: update means/diag_covs/weights
        vec_weights = vec_occup_eN / vec_occup_eN.sum();
        vec_occup_eN += vec_eps_regularize;
        #pragma omp parallel for
        for (int c = 0; c < num_modes; c++) {
            if (vec_weights(c) > eps_zero) {
                mat_means.row(c) = mat_occup_eX.row(c)/vec_occup_eN(c);
            }
        }

        VectorXf vec_sum_occup_nrm2_pts = mat_log_probs.transpose() * vec_nrm2_pts;
        if (should_fit_spherical_gaussian) {
            for (int c = 0; c < num_modes; c++) {
                const float spherical_val 
                    = (vec_sum_occup_nrm2_pts(c) - 2*mat_means.row(c).dot(mat_occup_eX.row(c)) + vec_nrm2_centers(c)*vec_occup_eN(c))
                    / (dim*vec_occup_eN(c));
                mat_diag_covs.row(c).fill(spherical_val);
            }
        } else {
            mat_occup_eX2 = mat_log_probs.transpose() * mat_data.array().square().matrix();
            for (long c = 0; c < num_modes; c++) {
                mat_diag_covs.row(c) = ((mat_occup_eX2.row(c)/vec_occup_eN(c)).array() - mat_means.row(c).array().square()).max(eps_covariance).matrix();
            }
        }

        const float prev_expectation = expectation;
        expectation = vec_evals.sum();
        const float scale = 1e5;
        const float delta = exp((expectation - prev_expectation)/scale) - 1;
        is_converged = (iterations > 0 && delta < eps_convergence);
        cout << "em " << "[" << iterations << "] " << delta << " " << expectation << endl;
    } //< em LOOP END
    
} //< function: fit_mixture_model

void likelihood_gmm(
        const float *data, 
        const long num_pts, 
        const long dim,
        const int num_modes,
        const float *means, 
        const float *diag_covs,
        const float *weights,
        float *log_probs, //< num_pts x num_modes
        bool is_spherical_gaussian) {

    Map<const RowMatrixXf> mat_data(data, num_pts, dim);
    Map<const RowMatrixXf> mat_means(means, num_modes, dim);
    Map<const RowMatrixXf> mat_diag_covs(diag_covs, num_modes, dim);
    Map<const RowVectorXf> vec_weights(weights, num_modes);

    VectorXf vec_nrm2_pts(num_pts);
    #pragma omp parallel for
    for (long r = 0; r < num_pts; r++) {
        vec_nrm2_pts(r) = mat_data.row(r).squaredNorm();
    }

    RowVectorXf vec_nrm2_centers(num_modes);

    RowMatrixXf mat_log_probs(num_pts, num_modes);

    //< calculate log probabilities
    if (is_spherical_gaussian) {
        calculate_log_prob_spherical(mat_data, vec_nrm2_pts, vec_weights, mat_means, mat_diag_covs, mat_log_probs);
    } else {
        calculate_log_prob_diagonal(mat_data, vec_weights, mat_means, mat_diag_covs, mat_log_probs);
    }

    std::copy(mat_log_probs.data(), mat_log_probs.data() + num_pts*num_modes, log_probs);
}
