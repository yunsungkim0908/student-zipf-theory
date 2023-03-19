#include <fstream>
#include <algorithm>
#include <gsl/gsl_sort.h>
#include "simple.h"

using namespace grammar;

SimpleGrammar::SimpleGrammar(int choices, int length){
    this->dist = new double[choices];
    this->r = gsl_rng_alloc(gsl_rng_mt19937);
    std::random_device rd;
    gsl_rng_set(this->r, rd());
    double alpha[choices];
    std::fill_n(alpha, choices, 1.0);

    gsl_ran_dirichlet(this->r, choices, alpha, this->dist);
    gsl_sort(&(this->dist[1]), 1, choices-1); 
    std::reverse(&(this->dist[1]), &(this->dist[1])+choices-1);
    std::vector<double> probs;
    probs.insert(probs.end(), this->dist, &(this->dist[choices]));
    this->distribution = new categorical(probs.begin(), probs.end());

    this->choices = choices;
    this->length = length;
    this->_use_dists = false;
}

SimpleGrammar::~SimpleGrammar(){
    delete r;
    delete distribution;
    delete[] dist;
}

void SimpleGrammar::sample_single_trajectory(std::mt19937 &generator,
                            std::unordered_map<std::string, int> &local_counts){
    std::stringstream ss;
    for (int i = 0; i < this->length; i++)
        ss << (*this->distribution)(generator) << "|";
    local_counts[ss.str()] += 1;
};

void SimpleGrammar::dump_dists(std::string dirname){
    std::string size_filename = dirname + std::string("/sizes.bin");
    std::string dist_filename = dirname + std::string("/dists.bin");
    std::ofstream size_file(size_filename, std::ios::binary);
    std::ofstream dist_file(dist_filename, std::ios::binary);
    std::cout << "Dumping distributions..." << std::endl;
    if (size_file.is_open()){
        size_file.write((const char*) &(this->choices), int(sizeof(int)/sizeof(char)));
        size_file.close();
    }
    if (dist_file.is_open()){
        dist_file.write((const char*) this->dist, this->choices * int((sizeof(double)/sizeof(char))));
        dist_file.close();
    }
    std::cout << "Done." << std::endl;
}

AbilitySimpleGrammar::AbilitySimpleGrammar(int choices, int length, double ability){
    this->dist = new double[choices];
    this->r = gsl_rng_alloc(gsl_rng_mt19937);
    std::random_device rd;
    gsl_rng_set(this->r, rd());
    double alpha[choices-1];
    std::fill_n(alpha, choices, 1.0);

    gsl_ran_dirichlet(this->r, choices-1, alpha, &(this->dist[1]));
    std::vector<double> probs;
    for (int i = 0; i < choices; i++)
        this->dist[i] = (i == 0) ? ability : this->dist[i]*(1-ability);
    // sort wrong probabilities only
    gsl_sort(&(this->dist[1]), 1, choices-1); 
    std::reverse(&(this->dist[1]), &(this->dist[1])+choices-1);
    probs.insert(probs.end(), dist, &(dist[choices]));
    this->distribution = new categorical(probs.begin(), probs.end());

    this->choices = choices;
    this->length = length;
    this->ability = ability;
}

void AbilitySimpleGrammar::dump_dists(std::string dirname){
    SimpleGrammar::dump_dists(dirname);
    std::string ability_filename = dirname + std::string("/ability.bin");
    std::ofstream ability_file(ability_filename, std::ios::binary);
    if (ability_file.is_open()){
        ability_file.write((const char*) &this->ability, int(sizeof(double)/sizeof(char)));
        ability_file.close();
    }
}

UnifAbilitySimpleGrammar::UnifAbilitySimpleGrammar(int choices, int length, int ability_bins){
    this->dist = new double[choices-1];
    this->r = gsl_rng_alloc(gsl_rng_mt19937);
    std::random_device rd;
    gsl_rng_set(this->r, rd());
    double alpha[choices-1];
    std::fill_n(alpha, choices-1, 1.0);

    gsl_ran_dirichlet(this->r, choices-1, alpha, dist);
    gsl_sort(this->dist, 1, choices-1); 
    std::reverse(this->dist, this->dist+choices-1);
    std::vector<double> probs;
    probs.insert(probs.end(), dist, &(dist[choices-1]));
    this->distribution = new categorical(probs.begin(), probs.end());

    std::cout << ((ability_bins == 0) ? "continuous " : "discrete ") << "ability" << std::endl;

    this->ability_bins = ability_bins;
    this->choices = choices;
    this->length = length;
}

void UnifAbilitySimpleGrammar::sample_single_trajectory(std::mt19937 &generator,
                            std::unordered_map<std::string, int> &local_counts){
    double ability = this->uniform_sampler(generator);
    double ability_bins = (double) this->ability_bins;
    if (this->ability_bins != 0)
        ability = double(int(ability*ability_bins)+1)/(ability_bins+1);
    std::stringstream ss;
    for (int i = 0; i < this->length; i++){
        int val = (this->uniform_sampler(generator) < ability)
                    ? 0 : (*this->distribution)(generator)+1;
        ss << val << "|";
    }
    local_counts[ss.str()] += 1;
};

void UnifAbilitySimpleGrammar::dump_dists(std::string dirname){
    SimpleGrammar::dump_dists(dirname);
    std::string ability_bins_filename = dirname + std::string("/ability_bins.bin");
    std::ofstream ability_bins_file(ability_bins_filename, std::ios::binary);
    if (ability_bins_file.is_open()){
        ability_bins_file.write((const char*) &this->ability_bins, int(sizeof(int)/sizeof(char)));
        ability_bins_file.close();
    }
}

