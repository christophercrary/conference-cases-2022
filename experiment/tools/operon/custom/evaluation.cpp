// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <doctest/doctest.h>
#include <interpreter/dispatch_table.hpp>
#include <thread>

#include "core/dataset.hpp"
#include "interpreter/interpreter.hpp"
#include "core/pset.hpp"
#include "operators/creator.hpp"
#include "operators/evaluator.hpp"
#include "parser/infix.hpp"
#include "nanobench.h"
#include <fstream>
#include <chrono>
#include "taskflow/taskflow.hpp"

namespace Operon {
namespace Test {
    std::size_t TotalNodes(const std::vector<Tree>& trees) {
#ifdef _MSC_VER
        auto totalNodes = std::reduce(trees.begin(), trees.end(), 0UL, [](size_t partial, const auto& t) { return partial + t.Length(); });
#else
        auto totalNodes = std::transform_reduce(trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });
#endif
        return totalNodes;
    }

    namespace nb = ankerl::nanobench;

    template <typename T>
    void Evaluate(tf::Executor& executor, std::vector<Tree> const& trees, Dataset const& ds, Range range)
    {
        DispatchTable ft;
        Interpreter interpreter(ft);
        tf::Taskflow taskflow;
        taskflow.for_each(trees.begin(), trees.end(), [&](auto const& tree) { interpreter.Evaluate<T>(tree, ds, range); });
        executor.run(taskflow).wait();
    }

    // used by some Langdon & Banzhaf papers as benchmark for measuring GPops/s
    TEST_CASE("Evaluation performance")
    {
       
        size_t n = 1000;
        size_t maxLength = 100;
        size_t maxDepth = 1000;

        Operon::RandomGenerator rd(1234);
        
        auto ds = Dataset("../data/single_val.csv", true);

        auto target = "y";
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

        Range range = { 0, ds.Rows() };
        //Range range = { 0, 10000 };






        PrimitiveSet pset;

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = BalancedTreeCreator { pset, inputs };

        std::vector<Tree> trees(n);

        //Evaluate<Operon::Scalar>(executor, trees, ds, range); 


        auto test = [&](tf::Executor& executor, nb::Bench& b, PrimitiveSetConfig cfg, const std::string& name) {
            pset.SetConfig(cfg);
            for (auto t : { NodeType::Add, NodeType::Sub, NodeType::Div, NodeType::Mul }) {
                pset.SetMinMaxArity(t, 2, 2);
            }
            std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

            auto totalOps = TotalNodes(trees) * range.Size();
            printf("\ntotal nodes: %ld\n", totalOps);
            b.batch(totalOps);
            b.run(name, [&]() { Evaluate<Operon::Scalar>(executor, trees, ds, range); });
        };

        SUBCASE("arithmetic") {
            // single-thread
            nb::Bench b;
            b.title("arithmetic").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + exp") {
            nb::Bench b;
            b.title("arithmetic + exp").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Exp, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + log") {
            nb::Bench b;
            b.title("arithmetic + log").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Log, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + sin") {
            nb::Bench b;
            b.title("arithmetic + sin").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Sin, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + cos") {
            nb::Bench b;
            b.title("arithmetic + cos").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Cos, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + tan") {
            nb::Bench b;
            b.title("arithmetic + tan").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Tan, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + sqrt") {
            nb::Bench b;
            b.title("arithmetic + sqrt").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Sqrt, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + cbrt") {
            nb::Bench b;
            b.title("arithmetic + cbrt").relative(true).performanceCounters(true).minEpochIterations(5);
            for (size_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Cbrt, fmt::format("N = {}", i));
            }
        }
    }


void gen(std::string const& typeName, char const* mustacheTemplate,
         ankerl::nanobench::Bench const& bench) {

    std::ofstream templateOut("mustache.template." + typeName);
    templateOut << mustacheTemplate;

    std::ofstream renderOut("mustache.render." + typeName);
    ankerl::nanobench::render(mustacheTemplate, bench, renderOut);
}



    TEST_CASE("Node Evaluations")
    {

        auto ds = Dataset("../data/single_val2.csv", true);

        auto target = "y";
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto const& v) { return v.Name != target; });
        Range range = { 0, ds.Rows() };
        size_t num_vars = ds.Cols() - 1; //-1 because there is a target variable

        auto problem = Problem(ds).Inputs(inputs).Target(target).TrainingRange(range).TestRange(range);
        //problem.GetPrimitiveSet().SetConfig(Operon::PrimitiveSet::Arithmetic);

        //std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        //auto creator = BalancedTreeCreator { problem.GetPrimitiveSet(), inputs };


        std::vector<std::string> *input_strs = new std::vector<std::string>();

        //read in input strings, which are in infix notation
        std::ifstream file("../text/test2.txt");

        std::cout<<"\nstring parsing\n";
        std::string str; 
        while (std::getline(file, str))
        {
            std::cout<<str<<std::endl;
            input_strs->push_back(str);    
        }


        std::vector<Tree> trees; //dont allocate this with new. 

        Hasher<HashFunction::XXHash> hasher;


        for (size_t j = 0; j < input_strs->size(); j++)
        {
            robin_hood::unordered_flat_map<std::string, Operon::Hash> vars_map;
            std::unordered_map<Operon::Hash, std::string> vars_names;
            for (size_t i = 0; i <= num_vars; ++i) {
                auto name = fmt::format("v{}", i);
                auto hash = hasher(reinterpret_cast<uint8_t const*>(name.data()), name.size() * sizeof(char) / sizeof(uint8_t));
                vars_map[name] = hash;
                vars_names[hash] = name;
            }

           // DispatchTable ft1;
            auto tree = Operon::InfixParser::Parse(input_strs->at(j), vars_map);
           // std::cout<<"\nafter parser\n";
            //fmt::print("\nTREE: \n{}\n", Operon::InfixFormatter::Format(tree, vars_names));
            trees.push_back(tree);

        }

         nb::Bench outer_b; 
        //b.title("Evaluator performance").relative(true).performanceCounters(true).minEpochIterations(10);

        auto totalNodes = TotalNodes(trees);
        //printf("\ntotal Nodes %d\n", totalNodes); //added Brit C.
        Operon::Vector<Operon::Scalar> buf(range.Size());

        Operon::RandomGenerator rd(1234); //doesnt seem like this is used in the evaluate function, at least for MSE (which is good)
 

        //look



        //std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

        std::vector<Individual> individuals(trees.size()); //create individuals out of trees for GP alg
        for (size_t i = 0; i < individuals.size(); ++i) {
            individuals[i].Genotype = trees.at(i);
        }


        auto test = [&](nb::Bench &b, std::string const& name, EvaluatorBase&& evaluator, int epochs, int epoch_iterations) {
            evaluator.SetLocalOptimizationIterations(0);
            evaluator.SetBudget(std::numeric_limits<size_t>::max());
            tf::Executor executor(std::thread::hardware_concurrency());
            tf::Taskflow taskflow;

            std::vector<Operon::Vector<Operon::Scalar>> slots(executor.num_workers());
            double sum{0};
            taskflow.transform_reduce(individuals.begin(), individuals.end(), sum, std::plus<>{}, [&](Operon::Individual& ind) {
                auto id = executor.this_worker_id();
                if (slots[id].size() < range.Size()) { slots[id].resize(range.Size()); }
                auto res =  evaluator(rd, ind, slots[id]).front();
                printf("result: %f", res);
                return res;
            });

            auto start = std::chrono::high_resolution_clock::now(); //epochs is number of measurements to perform
            b.batch(totalNodes * range.Size()).epochs(epochs).epochIterations(epoch_iterations).run(name, [&]() {
                sum = 0;
                executor.run(taskflow).wait();
                return sum;
            });

            
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()) / 1e6;
            double node_evals_psec = b.batch() * static_cast<double>(b.epochs() * b.epochIterations()) / duration;
            fmt::print("node evals / s: {:L}\n", node_evals_psec);
          //  printf("node evals from nb: %f",b.results()[1]);
        };

        Interpreter interpreter;
        test(outer_b, "mse", Operon::Evaluator<Operon::MSE, false>(problem, interpreter), 5, 2);
        gen("json", ankerl::nanobench::templates::json(), outer_b);




    }

    TEST_CASE("Evaluator performance")
    {
       // const size_t n         = 100;
       // const size_t maxLength = 100;
       // const size_t maxDepth  = 1000;

        //Operon::RandomGenerator rd(1234);
        auto ds = Dataset("../data/single_val2.csv", true);

        auto target = "y";
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto const& v) { return v.Name != target; });
        Range range = { 0, ds.Rows() };
        size_t num_vars = ds.Cols() - 1; //-1 because there is a target variable

        auto problem = Problem(ds).Inputs(inputs).Target(target).TrainingRange(range).TestRange(range);
        //problem.GetPrimitiveSet().SetConfig(Operon::PrimitiveSet::Arithmetic);

        //std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        //auto creator = BalancedTreeCreator { problem.GetPrimitiveSet(), inputs };


        std::vector<std::string> *input_strs = new std::vector<std::string>();

        //read in input strings, which are in infix notation
        std::ifstream file("../text/test2.txt");

        std::cout<<"\nstring parsing\n";
        std::string str; 
        while (std::getline(file, str))
        {
            std::cout<<str<<std::endl;
            input_strs->push_back(str);    
        }


        std::vector<Tree> trees; //dont allocate this with new. 

        Hasher<HashFunction::XXHash> hasher;


        for (size_t j = 0; j < input_strs->size(); j++)
        {
            robin_hood::unordered_flat_map<std::string, Operon::Hash> vars_map;
            std::unordered_map<Operon::Hash, std::string> vars_names;
            for (size_t i = 0; i <= num_vars; ++i) {
                auto name = fmt::format("v{}", i);
                auto hash = hasher(reinterpret_cast<uint8_t const*>(name.data()), name.size() * sizeof(char) / sizeof(uint8_t));
                vars_map[name] = hash;
                vars_names[hash] = name;
            }

           // DispatchTable ft1;
            auto tree = Operon::InfixParser::Parse(input_strs->at(j), vars_map);
           // std::cout<<"\nafter parser\n";
            //fmt::print("\nTREE: \n{}\n", Operon::InfixFormatter::Format(tree, vars_names));
            trees.push_back(tree);

        }
        //std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

        std::vector<Individual> individuals(trees.size()); //create individuals out of trees for GP alg
        for (size_t i = 0; i < individuals.size(); ++i) {
            individuals[i].Genotype = trees.at(i);
        }

        DispatchTable dt;
        Interpreter interpreter(dt);

        nb::Bench b; //is this being used?
        b.title("Evaluator performance").relative(true).performanceCounters(true).minEpochIterations(10);

        auto totalNodes = TotalNodes(trees);
        //printf("\ntotal Nodes %d\n", totalNodes); //added Brit C.
        Operon::Vector<Operon::Scalar> buf(range.Size());

        Operon::RandomGenerator rd(1234); //doesnt seem like this is used in the evaluate function, at least for MSE (which is good)
 

        //looks like this is running evalution on all trees using nano bench, which I am assuming is multithreaded
     /*
        auto test = [&](std::string const& name, EvaluatorBase&& evaluator) {
            evaluator.SetLocalOptimizationIterations(0);
            evaluator.SetBudget(std::numeric_limits<size_t>::max());
            b.batch(totalNodes * range.Size()).run(name, [&]() {
                return std::transform_reduce(individuals.begin(), individuals.end(), 0.0, std::plus<>{}, [&](auto& ind) { return evaluator(rd, ind, buf).front(); });
            });
        };
    */
/*
        test("r-squared",      Operon::Evaluator<Operon::R2, false>(problem, interpreter));
        test("r-squared + ls", Operon::Evaluator<Operon::R2, true>(problem, interpreter));
        test("nmse",           Operon::Evaluator<Operon::NMSE, false>(problem, interpreter));
        test("nmse + ls",      Operon::Evaluator<Operon::NMSE, true>(problem, interpreter));
        test("mae",            Operon::Evaluator<Operon::MAE, false>(problem, interpreter));
        test("mae + ls",       Operon::Evaluator<Operon::MAE, true>(problem, interpreter));
*/ 

        Operon::Evaluator<Operon::MSE, false> evaluator1 (problem, interpreter);
            evaluator1.SetLocalOptimizationIterations(0);
            evaluator1.SetBudget(std::numeric_limits<size_t>::max());
          //  auto res = std::transform_reduce(individuals.begin(), individuals.end(), 0.0, std::plus<>{}, [&](auto& ind) { return evaluator(rd, ind, buf).front(); });
            //printf("\nres: %f\n", res);

        auto threads = std::thread::hardware_concurrency();
        tf::Taskflow taskflow;
        tf::Executor executor(threads);

        ENSURE(executor.num_workers() > 0);
        printf("\n num threads: %ld\n", executor.num_workers());
        std::vector<Operon::Vector<Operon::Scalar>> slots(executor.num_workers());


        auto trainSize = ds.Rows();

        //this looks like the same thing as above, without nano bench. Make sure its multithreaded
        //int conc = 0;
        taskflow.emplace(
            [&](tf::Subflow& subflow) {
                subflow.for_each_index(0ul, trees.size(), 1ul, [&](size_t i) {
                    auto id = executor.this_worker_id();
                    //printf("\n id: %d, i: %lu\n", id, i);
                    //printf("conc: %d\n", conc++);
                    // make sure the worker has a large enough buffer

                    for(int j = 0; j < 10000000; j++);

                    if (slots[id].size() < trainSize) { slots[id].resize(trainSize); }
                    individuals[i].Fitness = evaluator1(rd, individuals[i], slots[id]);
                });});

        auto start = std::chrono::high_resolution_clock::now();
        executor.run(taskflow).wait();
        //executor.wait_for_all();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        float node_evals_psec = ((float)totalNodes) *(float)range.Size() * 1000000/ ((float)duration.count());


    //    for (size_t i =0 ;i < trees->size(); i++)
    //    {
    //        printf("\nindiv fitness: %f\n", individuals[i].Fitness.front());
   //     }


        printf("\nNode evaluations per second : %f\n", node_evals_psec);

       // test("mse",            Operon::Evaluator<Operon::MSE, false>(problem, interpreter));
        //test("mse + ls",       Operon::Evaluator<Operon::MSE, true>(problem, interpreter));


        //making lambda for entire test structure













    }
} // namespace Test
} // namespace Operon

