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
#include "stateful.h"
#include "simple.h"
#include "utils.h"

using namespace grammar;
using namespace boost;
namespace po = boost::program_options;

int ability_sampler(po::variables_map &vm){
    std::string type = vm["type"].as<std::string>();
    std::string load_dir = vm["load-dir"].as<std::string>();
    grammar::Grammar *gram;
    if (type.compare("unif_ability_stateful") == 0)
        gram = new grammar::UnifAbilityStatefulGrammar(load_dir);
    
    str_count_map* counts = gram->sample_trajectories(
                vm["num-traj"].as<int>(),
                vm["num-traj"].as<int>());
    if (vm.count("print-samples")){
        gram->print_samples(counts);
        return 0;
    }
    int sample_no = vm["sample-no"].as<int>();
    std::string sample_dir = load_dir + "/sample_" + std::to_string(sample_no);
    mkdir(sample_dir.c_str(), 0775);
    std::cerr << sample_dir << std::endl;

    gram->dump_count_map(counts, sample_dir, false);
    delete gram;
    return 0;
}

int main(int argc, char *argv[]){
    po::options_description desc("Options");
    desc.add_options()
        ("help", "help message")
        ("type", po::value<std::string>()->required(), "type of grammar")
        ("print-samples", "just print samples without storing")
        ("load-dir", po::value<std::string>()->required(), "grammar directory")
        ("sample-no", po::value<int>()->default_value(0), "sample number")
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
