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

#include "grammar.h"

#ifndef varied_ability_h
#define varied_ability_h

typedef std::uniform_real_distribution<double> unif_dist;
typedef std::discrete_distribution<int> categorical;
typedef std::unordered_map<std::string, int> str_count_map;

namespace grammar{

class VariedAbilityModel: public Grammar{

    public:
        VariedAbilityModel(int choices, int length, int ability_bins,
                            double uniformity, double ability_lim);
        VariedAbilityModel(std::string dirname);
        void sample_single_trajectory(
                double ability,
                std::mt19937 &generator,
                std::unordered_map<std::string, int> &local_counts);
        str_count_map* sample_trajectories(double ability, int num_traj, int per_thread);
        virtual void dump_count_map(str_count_map* counts, std::string label, double ability, bool counts_only);
        virtual void dump_dists(std::string label);

    protected:
        unif_dist uniform_sampler = unif_dist(0.0, 1.0);
        int      ability_bins;
        int      dist_size;
        double   uniformity;
        double   ability_lim;
};

}
#endif
