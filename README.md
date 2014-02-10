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

# Example

Points sampled from 2-component GMM can be found in sample.h, the mean and variance of the two Gaussians are 
```
    mean of Gaussian1 : [1 2]
    mean of Gaussian2 : [-3 -5]
    diagonal variance of Gaussian1: [2 0.5]
    diagonal variance of Gaussian2: [1 1]
```

The output of test.cc is (can be different due the random initialization of EM):
```
    $./test
    kmeans [1] 0 2.67368
    kmeans [2] 2.67368 0.000552935
    kmeans [3] 0.000552935 0
    em [1] inf -284034
    em [2] 9.53674e-07 -284034
    weights: 0.499979 0.500021
    means: -2.99555 -5.00304 ;
    0.99933 2.00078
    covs: 1.00493 0.9977 ;
    2.01731 0.499287
```
