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

using namespace grammar;
using namespace boost;
namespace po = boost::program_options;

int ability_sampler(po::variables_map &vm){
    std::string type = vm["type"].as<std::string>();
    grammar::Grammar *gram;
    if (type.compare("base") == 0){
        gram = new grammar::Grammar(
                vm["num-choices"].as<int>(),
                vm["length"].as<int>());
    } else if (type.compare("ability") == 0){
        gram = new grammar::AbilityGrammar(
                vm["num-choices"].as<int>(),
                vm["length"].as<int>(),
                vm["ability"].as<double>());
    } else if (type.compare("stateful") == 0){
        gram = new grammar::StatefulGrammar(
                vm["num-choices"].as<int>(),
                vm["length"].as<int>(),
                vm["num-types"].as<int>());
    } else if (type.compare("ability_stateful") == 0){
        gram = new grammar::AbilityStatefulGrammar(
                vm["num-choices"].as<int>(),
                vm["length"].as<int>(),
                vm["num-types"].as<int>(),
                vm["ability"].as<double>());
    } else if (type.compare("unif_ability_stateful") == 0){
        gram = new grammar::UnifAbilityStatefulGrammar(
                vm["num-choices"].as<int>(),
                vm["length"].as<int>(),
                vm["num-types"].as<int>(),
                vm["ability-bins"].as<int>(),
                vm["uniformity"].as<double>());
    } else if (type.compare("monkey") == 0){
        gram = new grammar::MonkeyGrammar( vm["num-choices"].as<int>());
    } else if (type.compare("simple") == 0){
        gram = new grammar::SimpleGrammar(
                vm["num-choices"].as<int>(),
                vm["length"].as<int>());
    } else if (type.compare("ability_simple") == 0){
        gram = new grammar::AbilitySimpleGrammar(
                vm["num-choices"].as<int>(),
                vm["length"].as<int>(),
                vm["ability"].as<double>());
    } else if (type.compare("unif_ability_simple") == 0){
        gram = new grammar::UnifAbilitySimpleGrammar(
                vm["num-choices"].as<int>(),
                vm["length"].as<int>(),
                vm["ability-bins"].as<int>());
    }

    std::string dirname = std::string(vm.count("dist-only") == 0 ? "out/" : "out/dist_only/");
    dirname += vm["label"].as<std::string>();
    mkdir(dirname.c_str(), 0775);
    gram->dump_dists(dirname);
    std::cout << "Done." << std::endl;
    if (vm.count("dist-only") == 0){
        str_count_map* counts = gram->sample_trajectories(
                    vm["num-traj"].as<int>(),
                    vm["per-thread"].as<int>());
        gram->dump_count_map(counts, dirname, true);
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
        ("type", po::value<std::string>()->required(), "type of grammar")
        ("load-dir", po::value<std::string>()->default_value(std::string("")), "load values from this directory")
        ("label", po::value<std::string>()->required(), "label for binary output")
        ("length", po::value<int>()->default_value(10), "length of choices")
        ("num-types", po::value<int>()->default_value(5), "number of types")
        ("ability", po::value<double>()->default_value(0.5), "ability param")
        ("uniformity", po::value<double>()->default_value(-1.0), "uniformity param (default: unif[0,1])")
        ("ability-bins", po::value<int>()->default_value(0), "number of discrete ability bins (default: unif[0,1].)")
        ("num-choices", po::value<int>()->default_value(5), "number of choices")
        ("num-traj", po::value<int>()->default_value(1000), "number of trajectories sampled")
        ("per-thread", po::value<int>()->default_value(1000), "number of trajectories sampled per thread");
    po::variables_map vm;    
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")){
        std::cout << desc << std::endl;
        return 1;
    }

    ability_sampler(vm);

    return 0;
}
