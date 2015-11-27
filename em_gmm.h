/**
 * @file em_gmm.h
 * @brief Expectation-Maximization Algorithm to fit GMM
 * @author Haoxiang Li
 * @version 1.1
 * @date 2013-12-29
 */
#pragma once

/**
 * @brief Fit Mixture of Gaussian given a set of points
 *
 * @param data pointer to the contiguous memory of data points (num_pts rows -by- dim cols)
 * @param num_pts number of points
 * @param dim dimension of data point
 * @param num_modes number of gaussians
 * @param means mean of the learned GMM, one row for one gaussian 
 * @param diag_covs diagonal covariance  of the learn GM<, one row for one gaussian
 * @param weights weights of the Gaussians
 * @param should_fit_spherical_gaussian whether the learned Gaussians are spherical or not, 
 *        spherical Gaussian has identical variance along all dimensions
 */
void em_gmm(
        const float *data, 
        const long num_pts, 
        const long dim,
        const int num_modes,
        float *means, 
        float *diag_covs,
        float *weights,
        bool should_fit_spherical_gaussian = true);

void likelihood_gmm(
        const float *data, 
        const long num_pts, 
        const long dim,
        const int num_modes,
        const float *means, 
        const float *diag_covs,
        const float *weights,
        float *log_probs, //< num_pts x num_modes
        bool is_spherical_gaussian = true);
