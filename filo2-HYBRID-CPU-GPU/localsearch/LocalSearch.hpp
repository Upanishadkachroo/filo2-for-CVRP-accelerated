#ifndef _FILO2_LOCALSEARCH_HPP_
#define _FILO2_LOCALSEARCH_HPP_

#include <algorithm>
#include <cfloat>
#include <random>
#include <iostream>
#include <string>
#include <unordered_map>

#include "../instance/Instance.hpp"
#include "../movegen/MoveGenerators.hpp"
#include "../solution/Solution.hpp"
#include "AbstractOperator.hpp"
#include "EjectionChain.hpp"
#include "OneOneExchange.hpp"
#include "OneZeroExchange.hpp"
#include "RevThreeOneExchange.hpp"
#include "RevThreeThreeExchange.hpp"
#include "RevThreeTwoExchange.hpp"
#include "RevThreeZeroExchange.hpp"
#include "RevTwoOneExchange.hpp"
#include "RevTwoTwoExchange.hpp"
#include "RevTwoZeroExchange.hpp"
#include "SplitExchange.hpp"
#include "TailsExchange.hpp"
#include "ThreeOneExchange.hpp"
#include "ThreeThreeExchange.hpp"
#include "ThreeTwoExchange.hpp"
#include "ThreeZeroExchange.hpp"
#include "TwoOneExchange.hpp"
#include "TwoOptExchange.hpp"
#include "TwoTwoExchange.hpp"
#include "TwoZeroExchange.hpp"

namespace cobra {

    // Supported local search operators.
    enum Operator {
        E10,
        E11,
        E20,
        E21,
        E22,
        E30,
        E31,
        E32,
        E33,
        SPLIT,
        TAILS,
        TWOPT,
        EJCH,
        RE20,
        RE21,
        RE22B,
        RE22S,
        RE30,
        RE31,
        RE32B,
        RE32S,
        RE33B,
        RE33S,
    };

    // Helper to convert operator enum to string.
    static std::string getOperatorName(Operator op) {
        switch(op) {
            case E10:   return "E10";
            case E11:   return "E11";
            case E20:   return "E20";
            case E21:   return "E21";
            case E22:   return "E22";
            case E30:   return "E30";
            case E31:   return "E31";
            case E32:   return "E32";
            case E33:   return "E33";
            case SPLIT: return "SPLIT";
            case TAILS: return "TAILS";
            case TWOPT: return "TWOPT";
            case EJCH:  return "EJCH";
            case RE20:  return "RE20";
            case RE21:  return "RE21";
            case RE22B: return "RE22B";
            case RE22S: return "RE22S";
            case RE30:  return "RE30";
            case RE31:  return "RE31";
            case RE32B: return "RE32B";
            case RE32S: return "RE32S";
            case RE33B: return "RE33B";
            case RE33S: return "RE33S";
            default:    return "UNKNOWN";
        }
    }

    // General VND interface.
    class VariableNeighborhoodDescentInterface : private NonCopyable<VariableNeighborhoodDescentInterface> {
    public:
        virtual void apply(Solution& solution) = 0;
    };

    // VND interface implementation as RVND.
    template <bool handle_partial_solutions = false>
    class RandomizedVariableNeighborhoodDescent : public VariableNeighborhoodDescentInterface {
    public:
        RandomizedVariableNeighborhoodDescent(const Instance& instance_, MoveGenerators& moves_, const std::vector<Operator>& operator_list,
                                              std::mt19937& rand_engine_, double tolerance = 0.01)
            : instance(instance_), moves(moves_), rand_engine(rand_engine_) {

            // Lambda to create operator and store pair.
            auto add_operator = [&](Operator op, AbstractOperator* ptr) {
                operators.push_back({op, ptr});
            };

            // Same initialization as before.
            OperatorInitTable[E10] = [this, tolerance]() {
                return new CommonOperator<OneZeroExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[E11] = [this, tolerance]() {
                return new CommonOperator<OneOneExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[E20] = [this, tolerance]() {
                return new CommonOperator<TwoZeroExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[E21] = [this, tolerance]() {
                return new CommonOperator<TwoOneExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[E22] = [this, tolerance]() {
                return new CommonOperator<TwoTwoExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[E30] = [this, tolerance]() {
                return new CommonOperator<ThreeZeroExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[E31] = [this, tolerance]() {
                return new CommonOperator<ThreeOneExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[E32] = [this, tolerance]() {
                return new CommonOperator<ThreeTwoExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[E33] = [this, tolerance]() {
                return new CommonOperator<ThreeThreeExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[SPLIT] = [this, tolerance]() {
                return new CommonOperator<SplitExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[TAILS] = [this, tolerance]() {
                return new CommonOperator<TailsExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[TWOPT] = [this, tolerance]() {
                return new CommonOperator<TwoOptExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[EJCH] = [this, tolerance]() {
                if constexpr (handle_partial_solutions) {
                    std::cout << "EjectionChain is not supported for partial solutions\n";
                    exit(1);
                }
                return new CommonOperator<EjectionChain<25>, false>(instance, moves, tolerance);
            };
            OperatorInitTable[RE20] = [this, tolerance]() {
                return new CommonOperator<RevTwoZeroExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[RE21] = [this, tolerance]() {
                return new CommonOperator<RevTwoOneExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[RE22B] = [this, tolerance]() {
                return new CommonOperator<RevTwoTwoExchange<true>, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[RE22S] = [this, tolerance]() {
                return new CommonOperator<RevTwoTwoExchange<false>, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[RE30] = [this, tolerance]() {
                return new CommonOperator<RevThreeZeroExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[RE31] = [this, tolerance]() {
                return new CommonOperator<RevThreeOneExchange, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[RE32B] = [this, tolerance]() {
                return new CommonOperator<RevThreeTwoExchange<true>, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[RE32S] = [this, tolerance]() {
                return new CommonOperator<RevThreeTwoExchange<false>, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[RE33B] = [this, tolerance]() {
                return new CommonOperator<RevThreeThreeExchange<true>, handle_partial_solutions>(instance, moves, tolerance);
            };
            OperatorInitTable[RE33S] = [this, tolerance]() {
                return new CommonOperator<RevThreeThreeExchange<false>, handle_partial_solutions>(instance, moves, tolerance);
            };

            // Build operators.
            for (auto op : operator_list) {
                auto ptr = OperatorInitTable[op]();
                operators.push_back({op, ptr});
            }

            // Shuffle the operators to prepare for randomization.
            shuffleOperators();
        }

        virtual ~RandomizedVariableNeighborhoodDescent() {
            for (auto& pair : operators) {
                delete pair.second;
            }
        }

        void apply(Solution& solution) override {

            // Shuffle again to randomize order each call.
            shuffleOperators();

            // Apply each operator once.
            for (auto& pair : operators) {
                AbstractOperator* op_ptr = pair.second;
                op_ptr->apply_rough_best_improvement(solution);
            }

            assert(solution.is_feasible());
        }

    private:
        void shuffleOperators() {
            std::shuffle(operators.begin(), operators.end(), rand_engine);
        }

        const Instance& instance;
        MoveGenerators& moves;
        std::mt19937& rand_engine;
        std::unordered_map<Operator, std::function<AbstractOperator*()>> OperatorInitTable;
        std::vector<std::pair<Operator, AbstractOperator*>> operators;
    };

    // Links together VNDs.
    class VariableNeighborhoodDescentComposer {

    public:
        VariableNeighborhoodDescentComposer(double tolerance_) : tolerance(tolerance_){};

        void append(VariableNeighborhoodDescentInterface* vnd) {
            tiers.push_back(vnd);
        }

        void sequential_apply(Solution& solution) {

        __again__:
            for (auto n = 0u; n < tiers.size(); n++) {
                const auto curr_cost = solution.get_cost();
                tiers[n]->apply(solution);
                if (n > 0 && solution.get_cost() + tolerance < curr_cost) {
                    goto __again__;
                }
            }
        }

    private:
        const double tolerance;
        std::vector<VariableNeighborhoodDescentInterface*> tiers;
    };

}  // namespace cobra

#endif
