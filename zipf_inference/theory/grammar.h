#include <unordered_map>
#include <stdio.h>
#include <iostream>
#include <random>
#include <set>

#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <sys/stat.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#ifndef grammar_h
#define grammar_h

typedef std::uniform_real_distribution<double> unif_dist;
typedef std::discrete_distribution<int> categorical;
typedef std::unordered_map<std::string, int> str_count_map;

namespace grammar{

class Grammar{
    public:
        Grammar() = default;
        Grammar(int choices, int length);
        virtual ~Grammar();
        virtual void sample_single_trajectory(std::mt19937 &generator,
                                    std::unordered_map<std::string, int> &local_counts);
        str_count_map* sample_trajectories(int num_traj, int per_thread);
        void print_samples(str_count_map* counts);
        void dump_count_map(str_count_map* counts, std::string label, bool counts_only);
        void dump_counts(std::multiset<uint32_t> *counts, std::string label, std::string fname);
        std::multiset<uint32_t>*  get_sorted_counts(str_count_map* counts);
        virtual void dump_dists(std::string label);
    protected:
        gsl_rng *r;
        categorical** distributions;
        std::unordered_map<std::string, int> counter; 
        int      choices;
        int      length;
        double** dists;
        bool     _use_dists = true;
};

class AbilityGrammar: public Grammar{
    public:
        AbilityGrammar(int choices, int length, double ability);
};

class MonkeyGrammar: public Grammar{
    public:
        MonkeyGrammar(int choices);
        virtual void sample_single_trajectory(std::mt19937 &generator,
                                    std::unordered_map<std::string, int> &local_counts);
        virtual void dump_dists(std::string label);

    protected:
        double* dist;
        categorical* distribution;
};

}

#endif
