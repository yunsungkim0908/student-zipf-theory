#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <algorithm>
#include <random>
#include <vector>
#include <sstream>
#include <omp.h>

#include <fstream>
#include <iostream>
#include <cstdint>

#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>

#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>

#include <typeinfo>
#include <chrono>
#include <thread>

#include "grammar.h"
#include "varied_ability.h"

using namespace grammar;
using namespace boost;
namespace po = boost::program_options;

double VARIED_ABILITY = -1.0;

int generator(po::variables_map &vm){
    int ability_bins = vm["ability-bins"].as<int>();
    int num_fixed_ability = vm["num-fixed-ability"].as<int>();
    double ability_lim = vm["ability-lim"].as<double>();
    bool load_dist = (vm.count("load-dist") != 0);
    bool dist_only = (vm.count("dist-only") != 0);

    if (ability_bins != 0 && num_fixed_ability != 0 &&
            (ability_bins-1) % (num_fixed_ability-1) != 0){
        std::cout << "ERROR: (num_ability_bins-1) must be divisible by (num_fixed_ability-1)!" << std::endl;
        return 0;
    }
    std::string type = vm["type"].as<std::string>();
    std::string dirname = std::string(vm.count("dist-only") == 0 ? "out/" : "out/dist_only/");
    dirname += vm["label"].as<std::string>();

    grammar::VariedAbilityModel *gram;
    if (load_dist){
        gram = new grammar::VariedAbilityModel(dirname);
    }
    else {
        gram = new grammar::VariedAbilityModel(
            vm["num-choices"].as<int>(),
            vm["length"].as<int>(),
            ability_bins,
            vm["uniformity"].as<double>(),
            ability_lim
        );

        std::cout << "dumping dists..." << std::endl;
        mkdir(dirname.c_str(), 0775);
        gram->dump_dists(dirname);
        std::cout << "Done." << std::endl;
    }

    if (!dist_only){
        str_count_map* counts = gram->sample_trajectories(
            VARIED_ABILITY,
            vm["num-traj"].as<int>(),
            vm["per-thread"].as<int>());
        gram->dump_count_map(counts, dirname, VARIED_ABILITY, false);

        if (num_fixed_ability != 0){
            double ability_step = (1 - 2*ability_lim)/double(num_fixed_ability-1);
            for (int i = 0; i < num_fixed_ability; i++){
                double ability = ability_lim + i*ability_step;
                std::cout << "fixed ability: " << ability << " (" << i+1 << "/" << num_fixed_ability << ")" << std::endl;
                counts = gram->sample_trajectories(
                    ability,
                    vm["num-traj"].as<int>(),
                    vm["per-thread"].as<int>());
                gram->dump_count_map(counts, dirname, ability, false);
            }
        }
    } else
        std::cout << "[NOTE] Sample distribution only." << std::endl;
    delete gram;
    return 0;
}

int main(int argc, char *argv[]){
    po::options_description desc("Options");
    desc.add_options()
        ("help", "help message")
        ("dist-only", "sample distributions only")
        ("load-dist", "load distributions from label dir")
        ("label", po::value<std::string>()->required(), "label for binary output")
        ("type", po::value<std::string>()->default_value(""), "type of grammar")
        ("num-choices", po::value<int>()->default_value(5), "number of choices")
        ("length", po::value<int>()->default_value(10), "length of choices")
        ("ability-bins", po::value<int>()->default_value(0), "number of discrete ability bins (default: unif[0,1].)")
        ("uniformity", po::value<double>()->default_value(-1.0), "uniformity param (default: unif[0,1])")
        ("ability-lim", po::value<double>()->default_value(0.2), "upper and lower ability bounds (default: 0.2 -> [0.2,0.8].)")
        ("num-fixed-ability", po::value<int>()->default_value(2), "number of fixed ability values to evaluate (default: 2)")
        ("num-traj", po::value<int>()->default_value(1000), "number of trajectories sampled")
        ("per-thread", po::value<int>()->default_value(1000), "number of trajectories sampled per thread");
    po::variables_map vm;    
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")){
        std::cout << desc << std::endl;
        return 1;
    }

    generator(vm);

    return 0;
}
