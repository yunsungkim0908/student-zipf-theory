#include <cmath>
#include <fstream>
#include <algorithm>
#include <gsl/gsl_sort.h>
#include "stateful.h"
#include "utils.h"
#include "omp.h"

using namespace grammar;

StatefulGrammar::StatefulGrammar(int choices, int length, int num_types){
    double*** dists = new double**[length];
    for (int i = 0; i < length; i++){
        dists[i] = new double*[num_types];
        for (int t = 0; t < num_types; t++)
            dists[i][t] = new double[choices];
    }
    this->dist_size = choices;
    this->r = gsl_rng_alloc(gsl_rng_mt19937);
    std::random_device rd;
    std::mt19937 generator(rd());
    gsl_rng_set(this->r, generator());
    double alpha[choices];
    std::fill_n(alpha, choices, 1.0);

    std::discrete_distribution<int>*** distributions = new std::discrete_distribution<int>**[length];
    for (int i = 0; i < length; i++){
        distributions[i] = new std::discrete_distribution<int>*[num_types];
        for (int t = 0; t < num_types; t++){
            gsl_ran_dirichlet(this->r, choices, alpha, dists[i][t]);
            // sort all 'wrong' probabilities, leave the 'correct' probabilities in the beginning
            gsl_sort(&(dists[i][t][1]), 1, choices-1); 
            std::reverse(&(dists[i][t][1]), &(dists[i][t][1])+choices-1);
            std::vector<double> probs;
            probs.insert(probs.end(), &(dists[i][t][0]), &(dists[i][t][choices]));
            distributions[i][t] = new std::discrete_distribution<int>(probs.begin(), probs.end());
        }
    }

    if (num_types > 1){
        unsigned long long size = (unsigned long long) pow(choices, length);
        this->type_map = new int[size];
        std::uniform_int_distribution<> uniform (0, num_types-1);
        std::cout << "Generating types" << std::endl;
        omp_set_num_threads(100);
#pragma omp parallel for
        for (unsigned long long i = 0; i < size; i++){
            this->type_map[i] = uniform(generator);
        }
    }

    this->dists = dists;
    this->length = length;
    this->choices = choices;
    this->num_types = num_types;
    this->distributions = distributions;
}

StatefulGrammar::~StatefulGrammar(){
    for (int i = 0; i < this->length; i++){
        for (int j = 0; j < this->num_types; j++){
            delete distributions[i][j];
            delete dists[i][j];
        }
        delete distributions[i];
        delete dists[i];
    }
    delete distributions;
    delete dists;
    delete type_map;
    delete r;
}

void StatefulGrammar::dump_dists(std::string dirname){
    std::string size_filename = dirname + std::string("/sizes.bin");
    std::string dist_filename = dirname + std::string("/dists.bin");
    std::string typemap_filename = dirname + std::string("/typemap.bin");
    std::ofstream dist_file(dist_filename, std::ios::binary);
    std::ofstream size_file(size_filename, std::ios::binary);
    std::ofstream typemap_file(size_filename, std::ios::binary);
    std::cout << "Dumping distributions..." << std::endl;
    if (size_file.is_open()){
        size_file.write((const char*) &(this->length), int(sizeof(int)/sizeof(char)));
        size_file.write((const char*) &(this->num_types), int(sizeof(int)/sizeof(char)));
        size_file.write((const char*) &(this->dist_size), int(sizeof(int)/sizeof(char)));
        size_file.write((const char*) &(this->choices), int(sizeof(int)/sizeof(char)));
        size_file.close();
    }
    if (dist_file.is_open()){
        for (int i=0; i < this->length; i++){
            for (int j=0; j < this->num_types; j++){
                dist_file.write((const char*) this->dists[i][j], this->dist_size * int(sizeof(double)/sizeof(char)));
            }
        }
        dist_file.close();
    }
    if (this->num_types > 1 && typemap_file.is_open()){
        unsigned long long size = (unsigned long long) pow(choices, length);
        typemap_file.write((const char*) this->type_map, size * int(sizeof(int)/sizeof(char)));
    }
}

StatefulGrammar::StatefulGrammar(std::string dirname){
}

void StatefulGrammar::sample_single_trajectory(std::mt19937 &generator,
                                            std::unordered_map<std::string, int> &local_counts){
    unsigned long long curr = 0;
    std::stringstream ss;
    for (int i = 0; i < this->length; i++){
        int type = (this->num_types == 1) ? 0 : this->type_map[curr];
        int val = (*(this->distributions[i][type]))(generator);
        curr = curr*this->choices + val;
        ss << val << "|";
    }
    local_counts[ss.str()] += 1;
}

AbilityStatefulGrammar::AbilityStatefulGrammar(int choices, int length, int num_types, double ability){
    double*** dists = new double**[length];
    for (int i = 0; i < length; i++){
        dists[i] = new double*[num_types];
        for (int t = 0; t < num_types; t++)
            dists[i][t] = new double[choices];
    }
    this->dist_size = choices;
    this->r = gsl_rng_alloc(gsl_rng_mt19937);
    std::random_device rd;
    std::mt19937 generator(rd());
    gsl_rng_set(this->r, generator());
    double alpha[choices-1];
    std::fill_n(alpha, choices-1, 1.0);

    std::discrete_distribution<int>*** distributions = new std::discrete_distribution<int>**[length];
    for (int i = 0; i < length; i++){
        distributions[i] = new std::discrete_distribution<int>*[num_types];
        for (int t = 0; t < num_types; t++){
            gsl_ran_dirichlet(this->r, choices-1, alpha, &(dists[i][t][1]));
            for (int j = 0; j < choices; j++)
                dists[i][t][j] = (j == 0) ? ability : (1-ability)*dists[i][t][j];
            gsl_sort(&(dists[i][t][1]), 1, choices-1); 
            std::reverse(&(dists[i][t][1]), &(dists[i][t][1])+choices-1);
            std::vector<double> probs;
            probs.insert(probs.end(), &(dists[i][t][0]), &(dists[i][t][choices]));
            distributions[i][t] = new std::discrete_distribution<int>(probs.begin(), probs.end());
        }
    }

    if (num_types > 1){
        unsigned long long size = (unsigned long long) pow(choices, length);
        this->type_map = new int[size];
        std::uniform_int_distribution<> uniform (0, num_types-1);
        std::cout << "Generating types" << std::endl;
        omp_set_num_threads(100);
#pragma omp parallel for
        for (unsigned long long i = 0; i < size; i++){
            this->type_map[i] = uniform(generator);
        }
    }

    this->dists = dists;
    this->length = length;
    this->choices = choices;
    this->num_types = num_types;
    this->distributions = distributions;
}

void AbilityStatefulGrammar::dump_dists(std::string dirname){
    StatefulGrammar::dump_dists(dirname);
    std::string ability_filename = dirname + std::string("/ability.bin");
    std::ofstream ability_file(ability_filename, std::ios::binary);
    if (ability_file.is_open()){
        ability_file.write((const char*) &this->ability, int(sizeof(double)/sizeof(char)));
        ability_file.close();
    }
}

UnifAbilityStatefulGrammar::UnifAbilityStatefulGrammar(int choices, int length, int num_types,
                                                       int ability_bins, double uniformity){
    this->r = gsl_rng_alloc(gsl_rng_mt19937);
    std::random_device rd;
    std::mt19937 generator(rd());
    gsl_rng_set(this->r, generator());

    if (uniformity < 0.0){
        std::cout << "[NOTE] Uniformity chosen at random..." << std::endl;
        uniformity = this->uniform_sampler(generator);
    }
    std::cout << "Uniformity: " << uniformity << std::endl;

    double*** err_dists = new double**[length];
    for (int i = 0; i < length; i++){
        err_dists[i] = new double*[num_types];
        for (int t = 0; t < num_types; t++)
            err_dists[i][t] = new double[choices-1];
    }
    this->dist_size = choices-1;

    double alpha[choices-1];
    std::fill_n(alpha, choices-1, 1.0);

    std::discrete_distribution<int>*** distributions = new std::discrete_distribution<int>**[length];
    for (int i = 0; i < length; i++){
        distributions[i] = new std::discrete_distribution<int>*[num_types];
        for (int t = 0; t < num_types; t++){
            gsl_ran_dirichlet(this->r, choices-1, alpha, err_dists[i][t]);
            gsl_sort(err_dists[i][t], 1, choices-1); 
            std::reverse(err_dists[i][t], err_dists[i][t]+choices-1);
            std::vector<double> probs;
            for (int c = 0; c < choices-1; c++){
                double unif_prob = 1.0/double(this->dist_size);
                err_dists[i][t][c] = unif_prob + (err_dists[i][t][c] - unif_prob)*(1.0-uniformity);
                probs.insert(probs.end(), err_dists[i][t][c]);
            }
            probs.insert(probs.end(), &(err_dists[i][t][0]), &(err_dists[i][t][choices-1]));
            distributions[i][t] = new std::discrete_distribution<int>(probs.begin(), probs.end());
        }
    }

    if (num_types > 1){
        unsigned long long size = (unsigned long long) pow(choices, length);
        this->type_map = new int[size];
        std::uniform_int_distribution<> uniform (0, num_types-1);
        std::cout << "Generating types" << std::endl;
        omp_set_num_threads(100);
#pragma omp parallel for
        for (unsigned long long i = 0; i < size; i++){
            this->type_map[i] = uniform(generator);
        }
    }

    this->dists = err_dists;
    this->length = length;
    this->choices = choices;
    this->num_types = num_types;
    this->uniformity = uniformity;
    this->ability_bins = ability_bins;
    this->distributions = distributions;
}

void UnifAbilityStatefulGrammar::sample_single_trajectory(std::mt19937 &generator,
                                            std::unordered_map<std::string, int> &local_counts){
    double ability = this->uniform_sampler(generator);
    double ability_bins = (double) this->ability_bins;
    if (this->ability_bins != 0)
        ability = double(int(ability*ability_bins)+1)/(ability_bins+1);
    double unif_prob = 1.0/double(this->choices);
    ability = unif_prob + (ability - unif_prob)*(1.0-this->uniformity);
    unsigned long long curr = 0;
    std::stringstream ss;
    for (int i = 0; i < this->length; i++){
        int val;
        if (this->uniform_sampler(generator) < ability)
            val = 0; // correct (=0) with probability "ability"
        else{
            int type = (this->num_types == 1) ? 0 : this->type_map[curr];
            val = (*(this->distributions[i][type]))(generator)+1;
        }
        curr = curr*this->choices + val;
        ss << val << "|";
    }
    local_counts[ss.str()] += 1;
}

void UnifAbilityStatefulGrammar::dump_dists(std::string dirname){
    StatefulGrammar::dump_dists(dirname);
    std::string ability_bins_filename = dirname + std::string("/ability_bins.bin");
    std::string uniformity_filename = dirname + std::string("/uniformity.bin");
    std::ofstream ability_bins_file(ability_bins_filename, std::ios::binary);
    std::ofstream uniformity_file(uniformity_filename, std::ios::binary);
    if (ability_bins_file.is_open()){
        ability_bins_file.write((const char*) &this->ability_bins, int(sizeof(int)/sizeof(char)));
        ability_bins_file.close();
    }
    if (uniformity_file.is_open()){
        uniformity_file.write((const char*) &this->uniformity, int(sizeof(double)/sizeof(char)));
        uniformity_file.close();
    }
}

UnifAbilityStatefulGrammar::UnifAbilityStatefulGrammar(std::string dirname)
{
    assert(dir_exists(dirname.c_str()) > 0);
    std::string size_filename = dirname + std::string("/sizes.bin");
    std::string dist_filename = dirname + std::string("/dists.bin");
    std::string typemap_filename = dirname + std::string("/typemap.bin");
    std::ifstream dist_file(dist_filename, std::ios::binary);
    std::ifstream size_file(size_filename, std::ios::binary);
    std::ifstream typemap_file(size_filename, std::ios::binary);
    std::cout << "Loading distributions from: " << dirname << std::endl;
    if (size_file.is_open()){
        size_file.read((char*) &(this->length), int(sizeof(int)/sizeof(char)));
        size_file.read((char*) &(this->num_types), int(sizeof(int)/sizeof(char)));
        size_file.read((char*) &(this->dist_size), int(sizeof(int)/sizeof(char)));
        size_file.close();
    }
    this->choices = this->dist_size+1;

    if (this->num_types > 1){
        unsigned long long size = (unsigned long long) pow(this->choices, this->length);
        this->type_map = new int[size];
        if (this->num_types > 1 && typemap_file.is_open())
            typemap_file.read((char*) this->type_map, size * int(sizeof(int)/sizeof(char)));
        else
            std::fill_n(this->type_map, size, 0);
    }

    if (dist_file.is_open()){
        this->dists = new double**[this->length];
        for (int i=0; i < this->length; i++){
            this->dists[i] = new double*[this->num_types];
            for (int t=0; t < this->num_types; t++){
                this->dists[i][t] = new double[this->dist_size];
                dist_file.read((char*) this->dists[i][t], this->dist_size * int(sizeof(double)/sizeof(char)));
            }
        }
        dist_file.close();
    }
    std::discrete_distribution<int>*** distributions = new std::discrete_distribution<int>**[length];
    for (int i = 0; i < length; i++){
        distributions[i] = new std::discrete_distribution<int>*[this->num_types];
        for (int t = 0; t < this->num_types; t++){
            std::vector<double> probs;
            probs.insert(probs.end(), this->dists[i][t], &(this->dists[i][t][this->dist_size]));
            distributions[i][t] = new std::discrete_distribution<int>(probs.begin(), probs.end());
        }
    }
    this->distributions = distributions;

    std::string ability_bins_filename = dirname + std::string("/ability_bins.bin");
    std::string uniformity_filename = dirname + std::string("/uniformity.bin");
    std::ifstream ability_bins_file(ability_bins_filename, std::ios::binary);
    std::ifstream uniformity_file(uniformity_filename, std::ios::binary);
    if (ability_bins_file.is_open()){
        ability_bins_file.read((char*) &this->ability_bins, int(sizeof(int)/sizeof(char)));
        ability_bins_file.close();
    }
    if (uniformity_file.is_open()){
        uniformity_file.read((char*) &this->uniformity, int(sizeof(double)/sizeof(char)));
        uniformity_file.close();
    }
    std::cout << "Done." << std::endl;
}

