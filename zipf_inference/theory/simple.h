#include "grammar.h"

namespace grammar{

class SimpleGrammar: public Grammar{
    /*
     * Grammar where the distribution of choices is identical across all steps
     */

    public:
        SimpleGrammar(){this->_use_dists = false;};
        SimpleGrammar(int choices, int length);
        virtual ~SimpleGrammar();
        virtual void sample_single_trajectory(std::mt19937 &generator,
                                    std::unordered_map<std::string, int> &local_counts);
        virtual void dump_dists(std::string label);

    protected:
        categorical* distribution;
        double*      dist;
};

class AbilitySimpleGrammar: public SimpleGrammar{

    public:
        AbilitySimpleGrammar(int choices, int length, double ability);
        void dump_dists(std::string label);

    protected:
        double ability;
};

class UnifAbilitySimpleGrammar: public SimpleGrammar{

    public:
        UnifAbilitySimpleGrammar(int choices, int length, int ability_bins);
        virtual void sample_single_trajectory(std::mt19937 &generator,
                                    std::unordered_map<std::string, int> &local_counts);
        void dump_dists(std::string label);

    protected:
        unif_dist uniform_sampler = unif_dist(0.0, 1.0);
        int       ability_bins;
};

}
