/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package YaraParser.TransitionBasedSystem.Trainer;

import YaraParser.Accessories.Options;
import YaraParser.Accessories.Pair;
import YaraParser.Learning.NeuralNetwork.MLPNetwork;
import YaraParser.Structures.NeuralTrainingInstance;
import YaraParser.TransitionBasedSystem.Configuration.BeamElement;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Configuration.State;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;
import YaraParser.TransitionBasedSystem.Parser.ArcEager.Actions;
import YaraParser.TransitionBasedSystem.Parser.ArcEager.ArcEager;

import java.util.*;

public class ArcEagerBeamTrainer {
    Options options;
    private String updateMode;
    private ArrayList<Integer> dependencyRelations;

    public ArcEagerBeamTrainer(String updateMode, Options options, ArrayList<Integer> dependencyRelations) {
        this.updateMode = updateMode;
        this.options = options;
        this.dependencyRelations = dependencyRelations;
    }

    public ArrayList<NeuralTrainingInstance> getNextInstances(ArrayList<GoldConfiguration> trainData, int start, int end, double dropoutProb)
            throws Exception {
        ArrayList<NeuralTrainingInstance> instances = new ArrayList<>();
        for (int i = start; i < end; i++) {
            addInstance(trainData.get(i), instances, dropoutProb);
        }
        return instances;
    }

    private void addInstance(GoldConfiguration goldConfiguration, ArrayList<NeuralTrainingInstance> instances, double dropoutProb) throws Exception {
        Configuration initialConfiguration = new Configuration(goldConfiguration.getSentence(), options.rootFirst);
        Configuration firstOracle = initialConfiguration.clone();
        ArrayList<Configuration> beam = new ArrayList<Configuration>(options.beamWidth);
        beam.add(initialConfiguration);

        HashMap<Configuration, Double> oracles = new HashMap<>();

        oracles.put(firstOracle, 0.0);

        Configuration bestScoringOracle = null;

        while (!ArcEager.isTerminal(beam) && beam.size() > 0) {
            /**
             *  generating new oracles
             *  it keeps the oracles which are in the terminal state
             */
            HashMap<Configuration, Double> newOracles = new HashMap<>();

            Configuration currentConfig = null;
            for (Configuration conf : oracles.keySet()) {
                currentConfig = conf;
                break;
            }

            int[] baseFeatures = FeatureExtractor.extractBaseFeatures(currentConfig);
            int[] label = new int[2 * (dependencyRelations.size() + 1)];
            if (!ArcEager.canDo(Actions.LeftArc, currentConfig.state)) {
                for (int i = 2; i < 2 + dependencyRelations.size(); i++)
                    label[i + dependencyRelations.size()] = -1;
            }
            if (!ArcEager.canDo(Actions.RightArc, currentConfig.state)) {
                for (int i = 2; i < 2 + dependencyRelations.size(); i++)
                    label[i] = -1;
            }
            if (!ArcEager.canDo(Actions.Shift, currentConfig.state)) {
                label[0] = -1;
            }
            if (!ArcEager.canDo(Actions.Reduce, currentConfig.state)) {
                label[1] = -1;
            }

            bestScoringOracle = staticOracle(goldConfiguration, oracles, newOracles);
            oracles = newOracles;
            int action = bestScoringOracle.actionHistory.get(bestScoringOracle.actionHistory.size() - 1);
            if (action >= 2)
                action -= 1;

            /** todo
             for (int i = 0; i < baseFeatures.length; i++) {
             if (i < 19 && maps.rareWords.contains(baseFeatures[i]))
             if (randGen.nextDouble() <= dropoutProb && baseFeatures[i] != 1)
             baseFeatures[i] = 0;
             }
             **/

            label[action] = 1;
            instances.add(new NeuralTrainingInstance(baseFeatures, label));
            beam = new ArrayList<>(options.beamWidth);
            beam.add(bestScoringOracle);
        }
    }

    private Configuration staticOracle(GoldConfiguration goldConfiguration, HashMap<Configuration, Double> oracles,
                                       HashMap<Configuration, Double> newOracles) throws Exception {
        Configuration bestScoringOracle = null;
        int top = -1;
        int first = -1;
        HashMap<Integer, Pair<Integer, Integer>> goldDependencies = goldConfiguration.getGoldDependencies();
        HashMap<Integer, HashSet<Integer>> reversedDependencies = goldConfiguration.getReversedDependencies();

        for (Configuration configuration : oracles.keySet()) {
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
                    ArcEager.rightArc(newConfig.state, dependency);
                    newConfig.addAction(3 + dependency);
                    newConfig.addScore(score);
                } else if (top > 0 && goldDependencies.containsKey(top) && goldDependencies.get(top).first == first) {
                    int dependency = goldDependencies.get(top).second;
                    double score = 0;
                    ArcEager.leftArc(newConfig.state, dependency);
                    newConfig.addAction(3 + dependencyRelations.size() + dependency);
                    newConfig.addScore(score);
                } else if (top >= 0 && state.hasHead(top)) {
                    if (reversedDependencies.containsKey(top)) {
                        if (reversedDependencies.get(top).size() == state.valence(top)) {
                            ArcEager.reduce(newConfig.state);
                            newConfig.addAction(1);
                            newConfig.addScore(0);
                        } else {
                            double score = 0;
                            ArcEager.shift(newConfig.state);
                            newConfig.addAction(0);
                            newConfig.addScore(score);
                        }
                    } else {
                        double score = 0;
                        ArcEager.reduce(newConfig.state);
                        newConfig.addAction(1);
                        newConfig.addScore(score);
                    }
                } else if (state.bufferEmpty() && state.stackSize() == 1 && state.peek() == state.rootIndex) {
                    double score = 0;
                    ArcEager.reduce(newConfig.state);
                    newConfig.addAction(1);
                    newConfig.addScore(score);
                } else {
                    double score = 0;
                    ArcEager.shift(newConfig.state);
                    newConfig.addAction(0);
                    newConfig.addScore(score);
                }
                bestScoringOracle = newConfig;
                newOracles.put(newConfig, (double) 0);
            } else {
                newOracles.put(configuration, oracles.get(configuration));
            }
        }
        return bestScoringOracle;
    }

    private Configuration zeroCostDynamicOracle(GoldConfiguration goldConfiguration, HashMap<Configuration, Double>
            oracles, HashMap<Configuration, Double> newOracles, MLPNetwork network) throws Exception {
        double bestScore = Double.NEGATIVE_INFINITY;
        Configuration bestScoringOracle = null;

        for (Configuration configuration : oracles.keySet()) {
            if (!configuration.state.isTerminalState()) {
                State currentState = configuration.state;
                int[] features = FeatureExtractor.extractBaseFeatures(configuration);
                double[] scores = network.output(features, new int[network.getSoftmaxLayerDim()]);

                int accepted = 0;
                // I only assumed that we need zero cost ones
                if (goldConfiguration.actionCost(Actions.Shift, -1, currentState) == 0) {
                    Configuration newConfig = configuration.clone();
                    double score = scores[0];
                    ArcEager.shift(newConfig.state);
                    newConfig.addAction(0);
                    newConfig.addScore(score);
                    newOracles.put(newConfig, (double) 0);

                    if (newConfig.getScore(true) > bestScore) {
                        bestScore = newConfig.getScore(true);
                        bestScoringOracle = newConfig;
                    }
                    accepted++;
                }
                if (ArcEager.canDo(Actions.RightArc, currentState)) {
                    for (int dependency : dependencyRelations) {
                        if (goldConfiguration.actionCost(Actions.RightArc, dependency, currentState) == 0) {
                            Configuration newConfig = configuration.clone();
                            double score = scores[2 + dependency];
                            ArcEager.rightArc(newConfig.state, dependency);
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
                if (ArcEager.canDo(Actions.LeftArc, currentState)) {
                    for (int dependency : dependencyRelations) {
                        if (goldConfiguration.actionCost(Actions.LeftArc, dependency, currentState) == 0) {
                            Configuration newConfig = configuration.clone();
                            double score = scores[2 + dependencyRelations.size() + dependency];
                            ArcEager.leftArc(newConfig.state, dependency);
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
                if (goldConfiguration.actionCost(Actions.Reduce, -1, currentState) == 0) {
                    Configuration newConfig = configuration.clone();
                    double score = scores[1];
                    ArcEager.reduce(newConfig.state);
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

    private void beamSortOneThread(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver, MLPNetwork network) throws Exception {
        for (int b = 0; b < beam.size(); b++) {
            Configuration configuration = beam.get(b);
            State currentState = configuration.state;
            double prevScore = configuration.score;
            boolean canShift = ArcEager.canDo(Actions.Shift, currentState);
            boolean canReduce = ArcEager.canDo(Actions.Reduce, currentState);
            boolean canRightArc = ArcEager.canDo(Actions.RightArc, currentState);
            boolean canLeftArc = ArcEager.canDo(Actions.LeftArc, currentState);

            int[] labels = new int[network.getSoftmaxLayerDim()];
            if (!canShift) labels[0] = -1;
            if (!canReduce) labels[1] = -1;
            if (!canRightArc)
                for (int i = 0; i < dependencyRelations.size(); i++)
                    labels[2 + i] = -1;
            if (!canLeftArc)
                for (int i = 0; i < dependencyRelations.size(); i++)
                    labels[dependencyRelations.size() + 2 + i] = -1;
            int[] features = FeatureExtractor.extractBaseFeatures(configuration);
            double[] scores = network.output(features, labels);

            if (canShift) {
                double score = scores[0];
                double addedScore = score + prevScore;
                beamPreserver.add(new BeamElement(addedScore, b, 0, -1));

                if (beamPreserver.size() > options.beamWidth)
                    beamPreserver.pollFirst();
            }
            if (canReduce) {
                double score = scores[1];
                double addedScore = score + prevScore;
                beamPreserver.add(new BeamElement(addedScore, b, 1, -1));

                if (beamPreserver.size() > options.beamWidth)
                    beamPreserver.pollFirst();
            }

            if (canRightArc) {
                for (int dependency : dependencyRelations) {
                    double score = scores[2 + dependency];
                    double addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));

                    if (beamPreserver.size() > options.beamWidth)
                        beamPreserver.pollFirst();
                }
            }
            if (canLeftArc) {
                for (int dependency : dependencyRelations) {
                    double score = scores[2 + dependencyRelations.size() + dependency];
                    double addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));

                    if (beamPreserver.size() > options.beamWidth)
                        beamPreserver.pollFirst();
                }
            }
        }
    }

}