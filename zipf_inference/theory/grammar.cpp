#include <gsl/gsl_rng.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_randist.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <omp.h>

#include <fstream>
#include <iostream>
#include <cstdint>

#include <typeinfo>
#include <chrono>
#include <thread>

#include "grammar.h"
#include "utils.h"

using namespace grammar;

Grammar::Grammar(int choices, int length){
    double** dists = new double* [length];
    for (int i=0; i < length; i++)
        dists[i] = new double[choices];

    this->r = gsl_rng_alloc(gsl_rng_mt19937);
    std::random_device rd;
    gsl_rng_set(this->r, rd());

    double alpha[choices];
    std::fill_n(alpha, choices, 1.0);

    std::discrete_distribution<int>** distributions = new std::discrete_distribution<int>*[length];
    for (int i = 0; i < length; i++){
        gsl_ran_dirichlet(this->r, choices, alpha, dists[i]);
        // Sort everything including correct probability
        gsl_sort(dists[i], 1, choices); 
        std::reverse(dists[i], dists[i]+choices);
        std::vector<double> probs;
        probs.insert(probs.end(), &(dists[i][0]), &(dists[i][choices]));
        distributions[i] = new std::discrete_distribution<int> (probs.begin(), probs.end());
    }

    this->dists = dists;
    this->length = length;
    this->choices = choices;
    this->distributions = distributions;
}

Grammar::~Grammar(){
    if (!this->_use_dists)
        return;

    for (int i = 0; i < this->length; i++){
        delete distributions[i];
        delete dists[i];
    }
    delete distributions;
    delete dists;
    delete r;
}

void Grammar::sample_single_trajectory(std::mt19937 &generator,
                                        std::unordered_map<std::string, int> &local_counts){
    std::stringstream ss;
    for (int k=0; k<this->length; k++){
        int val = (*(this->distributions[k]))(generator);
        ss << val << "|";
    }
    local_counts[ss.str()] += 1;
}

str_count_map* Grammar::sample_trajectories(int num_traj, int per_thread) {

    std::random_device rd;
    std::mt19937 generator(rd());

    std::cerr << "Generating trajectories..." << std::endl;

    str_count_map* counts = new str_count_map();
    int traj_cnt = 0;
    omp_set_num_threads(200);
    indicators::BlockProgressBar* traj_bar = getProgressBar();
#pragma omp parallel for
    for (int i=0; i<num_traj/per_thread; i++){
        str_count_map local_counts;
        for (int j=0; j<per_thread; j++){
            this->sample_single_trajectory(generator, local_counts);
        }
#pragma omp critical
        {
            for (auto it : local_counts){
                (*counts)[it.first] += it.second;
            }
            traj_cnt += per_thread;
            float prog = 100.0*traj_cnt/float(num_traj);
            traj_bar->set_progress(prog);
        }
    }
    delete traj_bar;
    std::cerr << "\nDone" << std::endl;
    return counts;
}

std::multiset<uint32_t>* Grammar::get_sorted_counts(str_count_map* counts){
    std::cerr << "\nSorting counts..." << std::endl;
    std::multiset<uint32_t> *ordered_cnts = new std::multiset<uint32_t>();
    // Hide cursor
    indicators::show_console_cursor(true);

    indicators::BlockProgressBar* insert_bar = getProgressBar();

    int idx = 0;
    for (auto it : *counts) {
        ordered_cnts->insert((uint32_t) it.second);
        insert_bar->set_progress(100.0f*idx/float(counts->size()));
        idx += 1;
    }
    insert_bar->set_progress(100);
    indicators::show_console_cursor(true);

    std::cerr << "\nDone." << std::endl;
    
    delete insert_bar;
    return ordered_cnts;
}

void Grammar::print_samples(str_count_map* counts){
    std::cout << "\n\nSamples:" << std::endl;
    for (auto it : *counts){
        for (int i = 0; i < it.second; i++)
            std::cout << it.first << std::endl;
    }
}

void Grammar::dump_dists(std::string dirname){
    std::string size_filename = dirname + std::string("/sizes.bin");
    std::string dist_filename = dirname + std::string("/dists.bin");
    std::ofstream dist_file(dist_filename, std::ios::binary);
    std::ofstream size_file(size_filename, std::ios::binary);
    std::cerr << "Dumping distributions..." << std::endl;
    if (size_file.is_open()){
        size_file.write((const char*) &(this->length), int(sizeof(int)/sizeof(char)));
        size_file.write((const char*) &(this->choices), int(sizeof(int)/sizeof(char)));
        size_file.close();
    }
    if (dist_file.is_open()){
        for (int i=0; i < this->length; i++){
            dist_file.write((const char*) this->dists[i], this->choices * int((sizeof(double)/sizeof(char))));
        }
        dist_file.close();
    }
    std::cerr << "Done." << std::endl;
}

void Grammar::dump_count_map(str_count_map* counts, std::string dirname, bool counts_only){
    std::string keys_filename = dirname + std::string("/keys.txt");
    std::string vals_filename = dirname + std::string("/vals.bin");
    std::cerr << "Dumping count map..." << std::endl;
    std::ofstream keys_file(keys_filename);
    std::ofstream vals_file(vals_filename, std::ios::binary);
    if (keys_file.is_open() && vals_file.is_open()){
        for (auto it: *counts){
            if (!counts_only)
                keys_file << it.first << "\n";
            vals_file.write((const char*) &(it.second), int(sizeof(int)/sizeof(char)));
        }
        keys_file.close();
        vals_file.close();
    }
    std::cerr << "Done." << std::endl;
    delete counts;
}

void Grammar::dump_counts(std::multiset<uint32_t>* counts, std::string dirname, std::string fname){
    std::vector<uint32_t> counts_vec(counts->rbegin(), counts->rend());
    std::stringstream ss;
    ss << dirname << "/" << fname << ".bin";
    std::string out_filename = ss.str();
    std::ofstream out_file (out_filename, std::ios::binary); 
    std::cerr << "Dumping Counts..." << std::endl;
    if (out_file.is_open()){
        out_file.write((const char*) &(*counts_vec.begin()), counts_vec.size()*int(sizeof(uint32_t)/sizeof(char)));
        out_file.close();
    }
    delete counts;
    std::cerr << "Done." << std::endl;
}

AbilityGrammar::AbilityGrammar(int choices, int length, double ability){
    double** dists = new double* [length];
    for (int i=0; i < length; i++)
        dists[i] = new double[choices];
    this->r = gsl_rng_alloc(gsl_rng_mt19937);
    double alpha[choices-1];
    std::fill_n(alpha, choices-1, 1);

    std::discrete_distribution<int>** distributions = new std::discrete_distribution<int>*[length];
    for (int i = 0; i < length; i++){
        gsl_ran_dirichlet(this->r, choices-1, alpha, &(dists[i][1]));
        for (int j = 0; j < choices; j++)
            dists[i][j] = (j == 0) ? ability : dists[i][j]*(1-ability);
        gsl_sort(&(dists[i][1]), 1, choices-1); 
        std::reverse(&(dists[i][1]), &(dists[i][1])+choices-1);
        std::vector<double> probs (dists[i], dists[i] + choices);
        distributions[i] = new std::discrete_distribution<int> (probs.begin(), probs.end());
    }

    this->dists = dists;
    this->length = length;
    this->choices = choices;
    this->distributions = distributions;
}

MonkeyGrammar::MonkeyGrammar(int choices){
    double* dist = new double[choices];
    this->r = gsl_rng_alloc(gsl_rng_mt19937);
    double alpha[choices-1];
    std::fill_n(alpha, choices-1, 1);
    gsl_ran_dirichlet(this->r, choices, alpha, dist);
    // Leave correct probability in the front
    gsl_sort(&(dist[1]), 1, choices-1); 
    std::reverse(&(dist[1]), &(dist[1])+choices-1);
    std::vector<double> probs (dist, dist+choices);
    categorical* distribution = new categorical(probs.begin(), probs.end());

    this->dist = dist;
    this->distribution = distribution;
    this->choices = choices;
}

void MonkeyGrammar::sample_single_trajectory(std::mt19937 &generator,
                                std::unordered_map<std::string, int> &local_counts){
    std::stringstream ss;
    while (true){
        int val = (*(this->distribution))(generator);
        ss << val << "|";
        if (val == 0)
            break;
    }
    local_counts[ss.str()] += 1;
}
void MonkeyGrammar::dump_dists(std::string dirname){
    std::string dist_filename = dirname + std::string("/dists.bin");
    std::ofstream dist_file(dist_filename, std::ios::binary);
    std::cerr << "Dumping distributions..." << std::endl;
    if (dist_file.is_open()){
        dist_file.write((const char*) this->dists, this->choices * int((sizeof(double)/sizeof(char))));
        dist_file.close();
    }
    std::cerr << "Done." << std::endl;
}

