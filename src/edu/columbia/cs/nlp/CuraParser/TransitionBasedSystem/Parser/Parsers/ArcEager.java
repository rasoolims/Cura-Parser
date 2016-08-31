/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers;

import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Structures.Pair;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.State;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Features.FeatureExtractor;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.Actions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class ArcEager extends ShiftReduceParser {
    public void leftArc(State state, int dependency) throws Exception {
        state.addArc(state.pop(), state.bufferHead(), dependency);
    }

    public void rightArc(State state, int dependency) throws Exception {
        state.addArc(state.bufferHead(), state.peek(), dependency);
        state.push(state.bufferHead());
        state.incrementBufferHead();
        if (!state.isEmptyFlag() && state.bufferEmpty())
            state.setEmptyFlag(true);
    }

    public boolean canDo(Actions action, State state) {
        if (action == Actions.Shift) { //shift
            return !(!state.bufferEmpty() && state.bufferHead() == state.rootIndex && !state.stackEmpty()) && !state
                    .bufferEmpty() && !state.isEmptyFlag();
        } else if (action == Actions.RightArc) { //right arc
            if (state.stackEmpty())
                return false;
            return !(!state.bufferEmpty() && state.bufferHead() == state.rootIndex) && !state.bufferEmpty() && !state
                    .stackEmpty();

        } else if (action == Actions.LeftArc) { //left arc
            if (state.stackEmpty() || state.bufferEmpty())
                return false;

            if (!state.stackEmpty() && state.peek() == state.rootIndex)
                return false;

            return state.peek() != state.rootIndex && !state.hasHead(state.peek()) && !state.stackEmpty();
        } else if (action == Actions.Reduce) { //reduce
            return !state.stackEmpty() && state.hasHead(state.peek()) || !state.stackEmpty() && state.stackSize() ==
                    1 && state.bufferSize() == 0 && state.peek() == state.rootIndex;
        } else if (action == Actions.Unshift) { //unshift
            return !state.stackEmpty() && !state.hasHead(state.peek()) && state.isEmptyFlag();
        }
        return false;
    }

    public Configuration staticOracle(GoldConfiguration goldConfiguration, Configuration configuration, int depSize) throws Exception {
        int top = -1;
        int first = -1;
        HashMap<Integer, Pair<Integer, Integer>> goldDependencies = goldConfiguration.getGoldDependencies();
        HashMap<Integer, HashSet<Integer>> reversedDependencies = goldConfiguration.getReversedDependencies();

        State state = configuration.state;
        if (!state.stackEmpty())
            top = state.peek();
        if (!state.bufferEmpty())
            first = state.bufferHead();

        if (!configuration.state.isTerminalState()) {
            Configuration newConfig = configuration.clone();

            if (first > 0 && goldDependencies.containsKey(first) && goldDependencies.get(first).first == top) {
                int dependency = goldDependencies.get(first).second;
                double score = 0;
                rightArc(newConfig.state, dependency);
                newConfig.addAction(3 + dependency);
                newConfig.addScore(score);
            } else if (top > 0 && goldDependencies.containsKey(top) && goldDependencies.get(top).first == first) {
                int dependency = goldDependencies.get(top).second;
                double score = 0;
                leftArc(newConfig.state, dependency);
                newConfig.addAction(3 + depSize + dependency);
                newConfig.addScore(score);
            } else if (top >= 0 && state.hasHead(top)) {
                if (reversedDependencies.containsKey(top)) {
                    if (reversedDependencies.get(top).size() == state.valence(top)) {
                        reduce(newConfig.state);
                        newConfig.addAction(1);
                        newConfig.addScore(0);
                    } else {
                        double score = 0;
                        shift(newConfig.state);
                        newConfig.addAction(0);
                        newConfig.addScore(score);
                    }
                } else {
                    double score = 0;
                    reduce(newConfig.state);
                    newConfig.addAction(1);
                    newConfig.addScore(score);
                }
            } else if (state.bufferEmpty() && state.stackSize() == 1 && state.peek() == state.rootIndex) {
                double score = 0;
                reduce(newConfig.state);
                newConfig.addAction(1);
                newConfig.addScore(score);
            } else {
                double score = 0;
                shift(newConfig.state);
                newConfig.addAction(0);
                newConfig.addScore(score);
            }
            return newConfig;
        }
        return configuration;
    }

    public Configuration zeroCostDynamicOracle(GoldConfiguration goldConfiguration, HashMap<Configuration, Double>
            oracles, HashMap<Configuration, Double> newOracles, MLPNetwork network, int labelNullIndex, ArrayList<Integer> dependencyRelations)
            throws Exception {
        double bestScore = Double.NEGATIVE_INFINITY;
        Configuration bestScoringOracle = null;

        for (Configuration configuration : oracles.keySet()) {
            if (!configuration.state.isTerminalState()) {
                State currentState = configuration.state;
                double[] features = FeatureExtractor.extractFeatures(configuration, labelNullIndex, this);
                double[] scores = network.output(features, new double[network.getNumOutputs()]);

                int accepted = 0;
                // I only assumed that we need zero cost ones
                if (goldConfiguration.actionCost(Actions.Shift, -1, currentState, this) == 0) {
                    Configuration newConfig = configuration.clone();
                    double score = scores[0];
                    shift(newConfig.state);
                    newConfig.addAction(0);
                    newConfig.addScore(score);
                    newOracles.put(newConfig, (double) 0);

                    if (newConfig.getScore(true) > bestScore) {
                        bestScore = newConfig.getScore(true);
                        bestScoringOracle = newConfig;
                    }
                    accepted++;
                }
                if (canDo(Actions.RightArc, currentState)) {
                    for (int dependency : dependencyRelations) {
                        if (goldConfiguration.actionCost(Actions.RightArc, dependency, currentState, this) == 0) {
                            Configuration newConfig = configuration.clone();
                            double score = scores[2 + dependency];
                            rightArc(newConfig.state, dependency);
                            newConfig.addAction(3 + dependency);
                            newConfig.addScore(score);
                            newOracles.put(newConfig, (double) 0);

                            if (newConfig.getScore(true) > bestScore) {
                                bestScore = newConfig.getScore(true);
                                bestScoringOracle = newConfig;
                            }
                            accepted++;
                        }
                    }
                }
                if (canDo(Actions.LeftArc, currentState)) {
                    for (int dependency : dependencyRelations) {
                        if (goldConfiguration.actionCost(Actions.LeftArc, dependency, currentState, this) == 0) {
                            Configuration newConfig = configuration.clone();
                            double score = scores[2 + dependencyRelations.size() + dependency];
                            leftArc(newConfig.state, dependency);
                            newConfig.addAction(3 + dependencyRelations.size() + dependency);
                            newConfig.addScore(score);
                            newOracles.put(newConfig, (double) 0);

                            if (newConfig.getScore(true) > bestScore) {
                                bestScore = newConfig.getScore(true);
                                bestScoringOracle = newConfig;
                            }
                            accepted++;
                        }
                    }
                }
                if (goldConfiguration.actionCost(Actions.Reduce, -1, currentState, this) == 0) {
                    Configuration newConfig = configuration.clone();
                    double score = scores[1];
                    reduce(newConfig.state);
                    newConfig.addAction(1);
                    newConfig.addScore(score);
                    newOracles.put(newConfig, (double) 0);

                    if (newConfig.getScore(true) > bestScore) {
                        bestScore = newConfig.getScore(true);
                        bestScoringOracle = newConfig;
                    }
                    accepted++;
                }
            } else {
                newOracles.put(configuration, oracles.get(configuration));
            }
        }

        return bestScoringOracle;
    }
}
