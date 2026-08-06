#include "simulation_parameters.h"

#include <iostream>
#include <string>

using namespace litho;

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../config.yaml";
    std::cout << "Loading: " << path << "\n";

    SimulationParameters p;
    try {
        p = SimulationParameters::from_yaml(path);
    } catch (const std::exception& e) {
        std::cerr << "Parse failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n[system]"
              << "\n  wavelength_nm = " << p.system.wavelength_nm
              << "\n  pixel_size_nm = " << p.system.pixel_size_nm << "\n";

    std::cout << "\n[mask]"
              << "\n  image_name    = " << p.mask.image_name << "\n";

    std::cout << "\n[optics]"
              << "\n  na               = " << p.optics.na
              << "\n  refractive_index = " << p.optics.refractive_index
              << "\n  aberrations      = {";
    bool first = true;
    for (auto& kv : p.optics.aberrations) {
        if (!first) std::cout << ", ";
        std::cout << kv.first << ": " << kv.second;
        first = false;
    }
    std::cout << "}\n";

    std::cout << "\n[source]"
              << "\n  type      = " << p.source.type
              << "\n  sigma_in  = " << p.source.sigma_in
              << "\n  sigma_out = " << p.source.sigma_out << "\n";

    std::cout << "\n[resist]"
              << "\n  model     = " << p.resist.model
              << "\n  threshold = " << p.resist.threshold
              << "\n  alpha     = " << p.resist.alpha << "\n";

    std::cout << "\nOK\n";
    return 0;
}
