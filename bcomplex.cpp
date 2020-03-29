#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <cassert>
#include <cmath>
#include <algorithm>

using namespace std;

using space_t = std::vector<std::vector<float>>;
using csv_t = std::unordered_map<int, space_t>;

csv_t parse_cifar_csv(std::filesystem::path path) {
    auto result = csv_t{};
    auto csv_file = ifstream{path};

    string line;
    getline(csv_file, line);
    while (getline(csv_file, line)) {
        auto ss = stringstream{line};
        auto feat = vector<float>{};
        feat.reserve(128);
        float feat_coord;
        char sep;
        int clss; 
        ss >> clss;
        while (ss >> sep >> feat_coord) {
            feat.push_back(feat_coord);
        }
        result[clss].emplace_back(move(feat));
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


vector<vector<float>> compute_outer(const space_t& space_a, const space_t& space_b) {
    vector<vector<float>> outer;
    outer.reserve(space_a.size());
    for (int i=0; i<space_a.size(); i++) {
        vector<float> row;
        row.reserve(space_b.size());
        for (int j=0; j < space_b.size(); ++j) {
            row.push_back(dist(space_a[i], space_b[j]));
        }
        outer.emplace_back(move(row));
    }
    return outer;
}

vector<vector<float>> compute_inner(const space_t& space_a, const space_t& space_b) {
    vector<vector<float>> inner(space_a.size(), vector<float>(space_a.size(), 1.0/0.0));

    for (int k=0; k<space_b.size(); k++) {
        vector<float> distances;
        distances.reserve(space_b.size());
        for (int i=0; i<space_a.size(); i++) {
            float d = dist(space_a[i], space_b[k]);
            for (int j=0; j<distances.size(); j++) {
                float common_distance = d+distances[j];
                inner[i][j] = min(inner[i][j], common_distance);
            }
            distances.push_back(d);
        }
    }

    for (int i=0; i<space_a.size(); i++) {
        for (int j=0; j<i; j++) {
            float d = dist(space_a[i], space_a[j]);
            inner[i][j] = max(inner[i][j], d);
        }
    }

    return inner;
}

void compute(space_t& space_a, space_t& space_b) {
    // assert(clsses.size() == 2); // TODO: multiple

    auto outer = compute_outer(space_a, space_b);
    auto inner_a = compute_inner(space_a, space_b);
    auto inner_b = compute_inner(space_b, space_a);

}

int main(int argc, char** argv) {
    std::ios_base::sync_with_stdio(false);
    auto args = std::vector<std::string>(argv+1, argv+argc);
    if (args.size() >= 3) {
        int clssa = std::stoi(args[1]);
        int clssb = std::stoi(args[2]);

        auto spaces = parse_cifar_csv(args[0]);
        compute(spaces[clssa], spaces[clssb]);
        // std::cout << spacess[clssa].size() << std::endl;

        return 0;
    }
    return 1;
}