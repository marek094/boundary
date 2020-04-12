#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <cmath>
#include <algorithm>

using namespace std;



using space_t = vector<pair<int, vector<float>>>;

template<typename T>
bool bin_read(istream& is, T& out) {
    return (bool) is.read(reinterpret_cast<char*>(&out), sizeof(T));
}


space_t parse_cifar_bin(std::string path) {
    auto result = space_t{};

    ifstream fs(path, ios::binary);

    int dimension;
    bin_read(fs, dimension);
    // cout << dimension << endl;

    int clss;
    while (bin_read(fs, clss)) {
        auto feat = vector<float>{};
        feat.reserve(dimension);
        float num;
        for (int i = 0; i < dimension; ++i) {
            bin_read(fs, num);
            feat.push_back(num);
        }
        result.emplace_back(clss, move(feat));
    }

    return result;
}


float dist(const std::vector<float>& a, const std::vector<float> b) {
    assert(a.size() == b.size());
    float res = 0;
    for (int i=0; i<a.size(); i++) {
        float df = a[i]-b[i];
        res += df*df;
    }
    return std::sqrt(res);
}


void compute(space_t& space) {
    for (int i=0; i < space.size(); i++) {
        auto& [clss_a, feat_a] = space[i];
        for (int j=0; j<i; j++) {
            auto& [clss_b, feat_b] = space[j];
            if (clss_a == clss_b) {
                // cout << (+1.0/+0.0) << " ";
                cout << (1e6*1.0) << " ";
            } else {
                cout << dist(feat_a, feat_b) << " ";
            }
        }
        cout << endl;
    }
}

int main(int argc, char** argv) {
    std::ios_base::sync_with_stdio(false);
    auto args = std::vector<std::string>(argv+1, argv+argc);
    if (args.size() >= 1) {
        auto space = parse_cifar_bin(args[0]);
        compute(space);

        return 0;
    }
    return 1;
}