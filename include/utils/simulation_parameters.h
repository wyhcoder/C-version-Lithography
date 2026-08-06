#pragma once

#include <unordered_map>
#include <string>

namespace litho {

struct SystemParams {
    double wavelength_nm;
    double pixel_size_nm;
    int grid_size_pixel;
};

struct MaskParams {
    std::string image_name;
    int         row_grid_size;
};

struct OpticsParams {
    double                        na;
    double                        refractive_index;
    std::unordered_map<int, double> aberrations;   // e.g. {"z4": 0.0}
};

struct SourceParams1 {
    std::string type;        // conventional / annular / dipole / quasar
    double      sigma_in;
    double      sigma_out;
};

struct ResistParams {
    std::string model;       // threshold / sigmoid ...
    double      threshold;
    int         alpha;
};

struct SimulationParameters {
    SystemParams system;
    MaskParams   mask;
    OpticsParams optics;
    SourceParams1 source;
    ResistParams resist;

    static SimulationParameters from_yaml(const std::string& file_path);
};

}  // namespace litho
