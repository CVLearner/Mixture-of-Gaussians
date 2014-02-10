Mixture-of-Gaussians
====================

Fit a Gaussian mixture model given a set of data

# Features:
- Implemented in C++
- Depend on STL and [Eigen](http://eigen.tuxfamily.org) Only
- Can benefit from the using of Intel Math Kernel Library ([through Eigen](http://eigen.tuxfamily.org/dox/TopicUsingIntelMKL.html))

# Please check the test.cc for usage
```
void em_gmm(
        const float *data, 
        const long num_pts, 
        const long dim,
        const int num_modes,
        float *means, 
        float *diag_covs,
        float *weights,
        bool should_fit_spherical_gaussian = true);
```
