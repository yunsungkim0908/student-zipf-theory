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

int ability_sampler(po::variables_map &vm){
    std::string load_dir = vm["load-dir"].as<std::string>();
    double ability = vm["ability"].as<double>();
    grammar::VariedAbilityModel *gram;
    gram = new grammar::VariedAbilityModel(load_dir);
    
    str_count_map* counts = gram->sample_trajectories(
                ability,
                vm["num-traj"].as<int>(),
                vm["num-traj"].as<int>());
    if (vm.count("print-samples")){
        gram->print_samples(counts);
        return 0;
    }
    int sample_no = vm["sample-no"].as<int>();
    std::string sample_dir = load_dir + "/sample_" + std::to_string(sample_no);
    if (ability > 0){
        std::ostringstream sample_dir_stream;
        sample_dir_stream << std::fixed;
        sample_dir_stream << std::setprecision(2);
        sample_dir_stream << "_";
        sample_dir_stream << ability;
        sample_dir += sample_dir_stream.str();
    }
    mkdir(sample_dir.c_str(), 0775);
    std::cerr << sample_dir << std::endl;

    gram->dump_count_map(counts, sample_dir, -1, false);
    delete gram;
    return 0;
}

int main(int argc, char *argv[]){
    po::options_description desc("Options");
    desc.add_options()
        ("help", "help message")
        ("load-dir", po::value<std::string>()->required(), "grammar directory")
        ("ability", po::value<double>()->default_value(-1.0), "ability")
        ("print-samples", "just print samples without storing")
        ("sample-no", po::value<int>()->default_value(0), "sample number")
        ("type", po::value<std::string>()->default_value(std::string("vam")), "type of grammar")
        ("num-traj", po::value<int>()->default_value(100), "number of trajectories sampled");
    po::variables_map vm;    
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")){
        std::cerr << desc << std::endl;
        return 1;
    }

    ability_sampler(vm);

    return 0;
}
