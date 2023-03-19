#include <cmath>
#include <fstream>
#include <gsl/gsl_sort.h>
#include <sstream>

#include "varied_ability.h"
#include "omp.h"
#include "utils.h"

using namespace grammar;

VariedAbilityModel::VariedAbilityModel(int choices, int length, int ability_bins,
                                        double uniformity, double ability_lim){
    this->r = gsl_rng_alloc(gsl_rng_mt19937);
    std::random_device rd;
    std::mt19937 generator(rd());
    gsl_rng_set(this->r, generator());

    if (uniformity < 0.0){
        std::cout << "[NOTE] Uniformity chosen at random..." << std::endl;
        uniformity = this->uniform_sampler(generator);
    }
    std::cout << "Uniformity: " << uniformity << std::endl;

    double** err_dists = new double*[length];
    for (int i = 0; i < length; i++){
        err_dists[i] = new double[choices-1];
    }
    this->dist_size = choices-1;

    double alpha[choices-1];
    std::fill_n(alpha, choices-1, 1.0);

    std::discrete_distribution<int>** distributions = new std::discrete_distribution<int>*[length];
    for (int i = 0; i < length; i++){
        gsl_ran_dirichlet(this->r, choices-1, alpha, err_dists[i]);
        gsl_sort(err_dists[i], 1, choices-1); 
        std::reverse(err_dists[i], err_dists[i]+choices-1);
        std::vector<double> probs;
        for (int c = 0; c < choices-1; c++){
            double unif_prob = 1.0/double(this->dist_size);
            err_dists[i][c] = unif_prob + (err_dists[i][c] - unif_prob)*(1.0-uniformity);
            probs.insert(probs.end(), err_dists[i][c]);
        }
        probs.insert(probs.end(), &(err_dists[i][0]), &(err_dists[i][choices-1]));
        distributions[i] = new std::discrete_distribution<int>(probs.begin(), probs.end());
    }

    this->length = length;
    this->choices = choices;
    this->uniformity = uniformity;
    this->ability_bins = ability_bins;
    this->dists = err_dists;
    this->ability_lim = ability_lim;
    this->distributions = distributions;
}

VariedAbilityModel::VariedAbilityModel(std::string dirname){
    assert(dir_exists(dirname.c_str()) > 0);
    std::string size_filename = dirname + std::string("/sizes.bin");
    std::string dist_filename = dirname + std::string("/dists.bin");
    std::ifstream dist_file(dist_filename, std::ios::binary);
    std::ifstream size_file(size_filename, std::ios::binary);
    std::cout << "Loading distributions from: " << dirname << std::endl;
    if (size_file.is_open()){
        size_file.read((char*) &(this->length), int(sizeof(int)/sizeof(char)));
        size_file.read((char*) &(this->dist_size), int(sizeof(int)/sizeof(char)));
        size_file.read((char*) &(this->choices), int(sizeof(int)/sizeof(char)));
        size_file.close();
    }
    this->choices = this->dist_size + 1;

    if (dist_file.is_open()){
        this->dists = new double*[this->length];
        for (int i=0; i < this->length; i++){
            this->dists[i] = new double[this->dist_size];
            dist_file.read((char*) this->dists[i], this->dist_size * int(sizeof(double)/sizeof(char)));
        }
        dist_file.close();
    }

    std::discrete_distribution<int>** distributions = new std::discrete_distribution<int>*[length];
    for (int i = 0; i < this->length; i++){
        std::vector<double> probs;
        probs.insert(probs.end(), this->dists[i], &(this->dists[i][this->dist_size]));
        distributions[i] = new std::discrete_distribution<int>(probs.begin(), probs.end());
    }
    this->distributions = distributions;

    std::string ability_bins_filename = dirname + std::string("/ability_bins.bin");
    std::string ability_lim_filename = dirname + std::string("/ability_lim.bin");
    std::string uniformity_filename = dirname + std::string("/uniformity.bin");
    std::ifstream ability_bins_file(ability_bins_filename, std::ios::binary);
    std::ifstream ability_lim_file(ability_bins_filename, std::ios::binary);
    std::ifstream uniformity_file(uniformity_filename, std::ios::binary);
    if (ability_bins_file.is_open()){
        ability_bins_file.read((char*) &this->ability_bins, int(sizeof(int)/sizeof(char)));
        ability_bins_file.close();
    }
    if (ability_lim_file.is_open()){
        ability_lim_file.read((char*) &this->ability_lim, int(sizeof(int)/sizeof(char)));
        ability_lim_file.close();
    }
    if (uniformity_file.is_open()){
        uniformity_file.read((char*) &this->uniformity, int(sizeof(double)/sizeof(char)));
        uniformity_file.close();
    }
    std::cout << "Done." << std::endl;
}

void VariedAbilityModel::sample_single_trajectory(
        double ability,
        std::mt19937 &generator,
        std::unordered_map<std::string, int> &local_counts){

    if (ability < 0){
        // random ability
        double buff = this->ability_lim;
        ability = this->uniform_sampler(generator);
        double ability_bins = (double) this->ability_bins;
        if (this->ability_bins != 0)
            ability = double(int(ability*ability_bins))/(ability_bins-1);
        // limit ability to [buff, 1-buff]
        ability = buff + ability*(1 - 2*buff);
    }

    unsigned long long curr = 0;
    std::stringstream ss;
    for (int i = 0; i < this->length; i++){
        int val;
        if (this->uniform_sampler(generator) < ability)
            val = 0; // correct (=0) with probability "ability"
        else{
            val = (*(this->distributions[i]))(generator)+1;
        }
        curr = curr*this->choices + val;
        ss << val << "|";
    }
    local_counts[ss.str()] += 1;
}

str_count_map* VariedAbilityModel::sample_trajectories(double ability, int num_traj, int per_thread) {

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
            this->sample_single_trajectory(ability, generator, local_counts);
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

void VariedAbilityModel::dump_count_map(str_count_map* counts, std::string dirname, double ability, bool counts_only){
    std::string name_header = "";
    if (ability >= 0){
        std::ostringstream name_header_stream;
        name_header_stream << std::fixed;
        name_header_stream << std::setprecision(2);
        name_header_stream << ability;
        name_header = name_header_stream.str() + std::string("_");
    }

    std::string keys_filename = dirname + std::string("/") + name_header + std::string("keys.txt");
    std::string vals_filename = dirname + std::string("/") + name_header + std::string("vals.bin");
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

void VariedAbilityModel::dump_dists(std::string dirname){
    std::string size_filename = dirname + std::string("/sizes.bin");
    std::string dist_filename = dirname + std::string("/dists.bin");
    std::string ability_bins_filename = dirname + std::string("/ability_bins.bin");
    std::string ability_lim_filename = dirname + std::string("/ability_lim.bin");
    std::string uniformity_filename = dirname + std::string("/uniformity.bin");

    std::ofstream dist_file(dist_filename, std::ios::binary);
    std::ofstream size_file(size_filename, std::ios::binary);
    std::ofstream ability_bins_file(ability_bins_filename, std::ios::binary);
    std::ofstream ability_lim_file(ability_lim_filename, std::ios::binary);
    std::ofstream uniformity_file(uniformity_filename, std::ios::binary);

    std::cout << "Dumping distributions..." << std::endl;

    if (size_file.is_open()){
        size_file.write((const char*) &(this->length), int(sizeof(int)/sizeof(char)));
        size_file.write((const char*) &(this->dist_size), int(sizeof(int)/sizeof(char)));
        size_file.write((const char*) &(this->choices), int(sizeof(int)/sizeof(char)));
        size_file.close();
    }
    if (dist_file.is_open()){
        for (int i=0; i < this->length; i++){
            dist_file.write((const char*) this->dists[i], this->dist_size * int(sizeof(double)/sizeof(char)));
        }
        dist_file.close();
    }
    if (ability_bins_file.is_open()){
        ability_bins_file.write((const char*) &this->ability_bins, int(sizeof(int)/sizeof(char)));
        ability_bins_file.close();
    }
    if (ability_lim_file.is_open()){
        ability_lim_file.write((const char*) &this->ability_lim, int(sizeof(double)/sizeof(char)));
        ability_lim_file.close();
    }
    if (uniformity_file.is_open()){
        uniformity_file.write((const char*) &this->uniformity, int(sizeof(double)/sizeof(char)));
        uniformity_file.close();
    }
}
