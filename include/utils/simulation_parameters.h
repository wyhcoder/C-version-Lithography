#pragma once

#include <unordered_map>
#include <string>

namespace litho {

struct SystemParams {
    double wavelength_nm;
    double pixel_size_nm;
    int grid_size_pixel; // 光学网格及补齐后 mask 的边长（像素）；直接采用此值，奇偶尺寸自动处理。
};

struct MaskParams {
    std::string image_name;
    int         row_grid_size; // 原始图案的边长说明（像素）；实际成像尺寸由 system.grid_size_pixel 决定。
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
