#include <unordered_map>
#include <stdio.h>
#include <iostream>
#include <random>
#include <set>

#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "grammar.h"

#ifndef stateful_h
#define stateful_h

namespace grammar{

class StatefulGrammar: public Grammar{
    public:
        StatefulGrammar() = default;
        StatefulGrammar(int choices, int length, int num_types);
        StatefulGrammar(std::string dirname);
        virtual ~StatefulGrammar();
        virtual void dump_dists(std::string label);
        virtual void sample_single_trajectory(std::mt19937 &generator,
                                    std::unordered_map<std::string, int> &local_counts);
    protected:
        gsl_rng *r;
        std::discrete_distribution<int>*** distributions;
        std::unordered_map<std::string, int> counter; 
        int*      type_map;
        int       dist_size;
        int       choices;
        int       length;
        int       num_types;
        double*** dists;
};

class AbilityStatefulGrammar: public StatefulGrammar{
    public:
        AbilityStatefulGrammar(int choices, int length, int num_types, double ability);
        virtual void dump_dists(std::string label);

    protected:
        double ability;
};

class UnifAbilityStatefulGrammar: public StatefulGrammar{
    public:
        UnifAbilityStatefulGrammar(int choices, int length, int num_types, int ability_bins, double uniformity);
        UnifAbilityStatefulGrammar(std::string dirname);
        virtual void sample_single_trajectory(std::mt19937 &generator,
                                    std::unordered_map<std::string, int> &local_counts);
        virtual void dump_dists(std::string label);

    protected:
        unif_dist uniform_sampler = unif_dist(0.0, 1.0);
        int       ability_bins;
        double    uniformity;
};

}

#endif
