#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip> 

using namespace std;
template <typename T>
using Matrix = vector<vector<T>>;



template<typename T>
Matrix<T> createMatrix(int n, int m, T* data){
    Matrix<T> mat;
    for (int i(0); i < n; i++){
        mat.push_back(vector<T> ());
        for (int j(0); j < m; j++){
            mat[i].push_back((data[m * i + j]));
        }
    }
    return mat;
}



template<typename T>
void printMatrix(Matrix<T> mat){
    int n = mat.size();
    int m = mat[0].size();
    for (int i(0); i < n; i++){
        for (int j(0); j < m; j++){
            cout << mat[i][j] << " ";
        }
        cout << "\n";
    }
}



void compute_W(Matrix<double> &W)
{
    int D = W.size();
    for (double j(0); j < D; j++){
        W[0][j] = 1 / (j + 1);
        W[j][j] = 1 / (j + 1);
    }
    for (double j(2); j < D; j++){
        for (double i(j-1); i > 0; i--){
            W[i][j] = (j - i) / (i + 1) * W[i+1][j];
        }
    }
}



// Recursion function
int recurse(int n,
            vector<double> &x_f, vector<double> &x_b, 
            vector<int> &feature,
            vector<int> &child_left,
            vector<int> &child_right,
            vector<double> &threshold,
            vector<double> &value,
            vector<vector<double>> &W,
            int n_features,
            vector<double> &phi,
            vector<int> &in_SAB,
            vector<int> &in_SA)
{
    int current_feature = feature[n];
    int xf_child(0), xb_child(0);
    // Case 1: Leaf
    if (child_left[n] < 0)
    {
        for (int i(0); i < n_features; i++){
            if (in_SA[i] > 0)
            {
                phi[i] += W[in_SA[n_features]-1][in_SAB[n_features]-1] * value[n];
            }
            else if (in_SAB[i] > 0)
            {
                phi[i] -= W[in_SA[n_features]][in_SAB[n_features]-1] * value[n];
            }
        }
        return 0;
    }
    // Find children associated with xf and xb
    if (x_f[current_feature] <= threshold[n]){
        xf_child = child_left[n];
    } else {xf_child = child_right[n];}
    if (x_b[current_feature] <= threshold[n]){
        xb_child = child_left[n];
    } else {xb_child = child_right[n];}
    // Case 2: Feature encountered before
    if (in_SAB[current_feature] > 0){
        if (in_SA[current_feature] > 0){
            return recurse(xf_child, x_f, x_b, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SAB, in_SA);
        }
        else{
            return recurse(xb_child, x_f, x_b, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SAB, in_SA);
        }
    }
    // Case 3: xf and xb go the same way
    if (xf_child == xb_child){
        return recurse(xf_child, x_f, x_b, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SAB, in_SA);
    }
    // Case 4: xf and xb don't go the same way
    else {
        in_SA[current_feature]++; in_SA[n_features]++;
        in_SAB[current_feature]++; in_SAB[n_features]++;
        recurse(xf_child, x_f, x_b, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SAB, in_SA);
        in_SA[current_feature]--; in_SA[n_features]--;
        recurse(xb_child, x_f, x_b, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SAB, in_SA);
        in_SAB[current_feature]--; in_SAB[n_features]--;
        return 0;
    }
}



Matrix<double> treeSHAP(Matrix<double> &X_f, 
                                Matrix<double> &X_b, 
                                Matrix<int> &feature,
                                Matrix<int> &left_child,
                                Matrix<int> &right_child,
                                Matrix<double> &threshold,
                                Matrix<double> &value,
                                Matrix<double> &W)
{
    // Setup
    int n_features = X_f[0].size();
    int n_trees = feature.size();
    int size_background = X_b.size();
    int size_foreground = X_f.size();

    // Initialize the SHAP values to zero
    Matrix<double> phi_f_b(size_foreground, vector<double> (n_features, 0));
    // Iterate over all foreground instances
    for (int i(0); i < size_foreground; i++){
        // Iterate over all background instances
        for (int j(0); j < size_background; j++){
            // Iterate over all trees in the ensemble
            for (int t(0); t < n_trees; t++){
                // Last index is the size of the set
                vector<int> in_SAB(n_features+1, 0);
                vector<int> in_SA(n_features+1, 0);
                vector<double> phi(n_features, 0);

                // Start the recursion
                recurse(0, X_f[i], X_b[j], feature[t], left_child[t], right_child[t],
                            threshold[t], value[t], W, n_features, phi, in_SAB, in_SA);

                // Add the contribution of the tree and background instance
                for (int f(0); f < n_features; f++){
                    phi_f_b[i][f] += phi[f];
                }
            }
        }
        // Rescale w.r.t the number of background instances
        for (int f(0); f < n_features; f++){
            phi_f_b[i][f] /= size_background;
        }
    }
    return phi_f_b;
}



extern "C"
int main_treeshap(int Nx, int Nz, int Nt, int d, int depth, double* foreground, double* background,
                  double* threshold_, double* value_, int* feature_, int* left_child_, int* right_child_,
                  double* result) {
    
    // Load data instances
    Matrix<double> X_f = createMatrix<double>(Nx, d, foreground);
    Matrix<double> X_b = createMatrix<double>(Nz, d, background);

    // Load tree structure
    Matrix<double> threshold = createMatrix<double>(Nt, depth, threshold_);
    Matrix<double> value = createMatrix<double>(Nt, depth, value_);
    Matrix<int> feature = createMatrix<int>(Nt, depth, feature_);
    Matrix<int> left_child  = createMatrix<int>(Nt, depth, left_child_);
    Matrix<int> right_child = createMatrix<int>(Nt, depth, right_child_);

    // Precompute the SHAP weights
    Matrix<double> W(d, vector<double> (d));
    compute_W(W);

    Matrix<double> phi = treeSHAP(X_f, X_b, feature, left_child, right_child, 
                                            threshold, value, W);
    
    // Save the results
    for (unsigned int i(0); i < phi.size(); i++){
        for (int j(0); j < d; j++){
            result[i * d + j] = phi[i][j];
        }
    }
    return 0;
}
