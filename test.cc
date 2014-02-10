#include <iostream>
#include <vector>
#include <fstream>
#include <random>

#include "em_gmm.h"
#include "sample.h"

int main(int argc, char **argv) {

    using namespace std;

    const long num_pts(1e5);
    const long num_gaussians(2);
    const long dim(2);

    vector<float> weights(num_gaussians);
    vector<float> means(num_gaussians*dim);
    vector<float> diag_covs(num_gaussians*dim);

    em_gmm(sample_data, num_pts, dim, num_gaussians,
            means.data(), diag_covs.data(), 
            weights.data(), false /*diagonal gaussians*/);

    cout << "weights: " << weights[0] << " " << weights[1] << endl;
    cout << "means: " << means[0] << " " << means[1] << " ;\n"
        << means[2] << " " << means[3] << endl;
    cout << "covs: " << diag_covs[0] << " " << diag_covs[1] << " ;\n"
        << diag_covs[2] << " " << diag_covs[3] << endl;

    return 0;
}
