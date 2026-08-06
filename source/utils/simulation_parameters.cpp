#include "simulation_parameters.h"

#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace litho {

template <typename T>
static T req(const YAML::Node& n, const std::string& key) {
    if (!n || !n[key])
        throw std::runtime_error("YAML missing required field: " + key);
    return n[key].as<T>();
}

SimulationParameters SimulationParameters::from_yaml(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    SimulationParameters p;

    // ── system ────────────────────────────────────────────────────
    {
        auto n = root["system"];
        p.system.wavelength_nm = req<double>(n, "wavelength_nm");
        p.system.pixel_size_nm = req<double>(n, "pixel_size_nm");
        p.system.grid_size_pixel = req<int>(n, "grid_size_pixel");
    }

    // ── mask ──────────────────────────────────────────────────────
    {
        auto n = root["mask"];
        p.mask.image_name = req<std::string>(n, "image_name");
        p.mask.row_grid_size = req<int>(n, "row_grid_size");
    }

    // ── optics ────────────────────────────────────────────────────
    {
        auto n = root["optics"];
        p.optics.na               = req<double>(n, "na");
        p.optics.refractive_index = req<double>(n, "refractive_index");
        if (n["aberrations"] && n["aberrations"].IsMap()) {
            for (auto it : n["aberrations"]) {
                p.optics.aberrations[it.first.as<int>()] =
                    it.second.as<double>();
            }
        }
    }

    // ── source ────────────────────────────────────────────────────
    {
        auto n = root["source"];
        p.source.type      = req<std::string>(n, "type");
        p.source.sigma_in  = req<double>(n, "sigma_in");
        p.source.sigma_out = req<double>(n, "sigma_out");
    }

    // ── resist ────────────────────────────────────────────────────
    {
        auto n = root["resist"];
        p.resist.model     = req<std::string>(n, "model");
        p.resist.threshold = req<double>(n, "threshold");
        p.resist.alpha     = req<int>(n, "alpha");
    }

    return p;
}

}  // namespace litho
