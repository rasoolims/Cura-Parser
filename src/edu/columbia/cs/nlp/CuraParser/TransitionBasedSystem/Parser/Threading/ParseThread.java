/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */


package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Threading;

import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Structures.Pair;
import edu.columbia.cs.nlp.CuraParser.Structures.Sentence;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.BeamElement;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.State;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Features.FeatureExtractor;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.Actions;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;

import java.util.ArrayList;
import java.util.TreeSet;
import java.util.concurrent.Callable;

public class ParseThread implements Callable<Pair<Configuration, Integer>> {
    MLPNetwork network;

    ArrayList<Integer> dependencyRelations;
    Sentence sentence;
    boolean rootFirst;
    int beamWidth;
    GoldConfiguration goldConfiguration;
    boolean partial;
    int labelNullIndex;
    int id;
    ShiftReduceParser parser;

    public ParseThread(int id, MLPNetwork network, Sentence sentence,
                       boolean rootFirst, int beamWidth, GoldConfiguration goldConfiguration, boolean partial, int labelNullIndex,
                       ShiftReduceParser parser) {
        this.id = id;
        this.network = network;
        this.dependencyRelations = network.getDepLabels();
        this.sentence = sentence;
        this.rootFirst = rootFirst;
        this.beamWidth = beamWidth;
        this.goldConfiguration = goldConfiguration;
        this.partial = partial;
        this.labelNullIndex = labelNullIndex;
        this.parser = parser;
    }

    @Override
    public Pair<Configuration, Integer> call() throws Exception {
        if (!partial)
            return parse();
        else return new Pair<>(parsePartial(), id);
    }

    Pair<Configuration, Integer> parse() throws Exception {
        Configuration initialConfiguration = new Configuration(sentence, rootFirst);

        ArrayList<Configuration> beam = new ArrayList<Configuration>(beamWidth);
        beam.add(initialConfiguration);

        while (!parser.isTerminal(beam)) {
            if (beamWidth != 1) {
                TreeSet<BeamElement> beamPreserver = new TreeSet<BeamElement>();
                for (int b = 0; b < beam.size(); b++) {
                    Configuration configuration = beam.get(b);
                    State currentState = configuration.state;
                    double prevScore = configuration.score;
                    boolean canShift = parser.canDo(Actions.Shift, currentState);
                    boolean canReduce = parser.canDo(Actions.Reduce, currentState);
                    boolean canRightArc = parser.canDo(Actions.RightArc, currentState);
                    boolean canLeftArc = parser.canDo(Actions.LeftArc, currentState);
                    double[] labels = new double[network.getNumOutputs()];
                    if (!canShift) labels[0] = -1;
                    if (!canReduce) labels[1] = -1;
                    if (!canRightArc)
                        for (int i = 0; i < dependencyRelations.size(); i++)
                            labels[2 + i] = -1;
                    if (!canLeftArc)
                        for (int i = 0; i < dependencyRelations.size(); i++)
                            labels[dependencyRelations.size() + 2 + i] = -1;
                    double[] features = FeatureExtractor.extractFeatures(configuration, labelNullIndex, parser);
                    double[] scores = network.output(features, labels);
                    if (!canShift
                            && !canReduce
                            && !canRightArc
                            && !canLeftArc) {
                        beamPreserver.add(new BeamElement(prevScore, b, 4, -1));

                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }

                    if (canShift) {
                        double score = scores[0];
                        double addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 0, -1));

                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }

                    if (canReduce) {
                        double score = scores[1];
                        double addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 1, -1));

                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }

                    if (canRightArc) {
                        for (int dependency : dependencyRelations) {
                            double score = scores[2 + dependency];
                            double addedScore = score + prevScore;
                            beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));

                            if (beamPreserver.size() > beamWidth)
                                beamPreserver.pollFirst();
                        }
                    }

                    if (canLeftArc) {
                        for (int dependency : dependencyRelations) {
                            double score = scores[2 + dependencyRelations.size() + dependency];
                            double addedScore = score + prevScore;
                            beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));

                            if (beamPreserver.size() > beamWidth)
                                beamPreserver.pollFirst();
                        }
                    }
                }

                ArrayList<Configuration> repBeam = new ArrayList<Configuration>(beamWidth);
                for (BeamElement beamElement : beamPreserver.descendingSet()) {
                    if (repBeam.size() >= beamWidth)
                        break;
                    int b = beamElement.number;
                    int action = beamElement.action;
                    int label = beamElement.label;
                    double score = beamElement.score;

                    Configuration newConfig = beam.get(b).clone();

                    if (action == 0) {
                        parser.shift(newConfig.state);
                        newConfig.addAction(0);
                    } else if (action == 1) {
                        parser.reduce(newConfig.state);
                        newConfig.addAction(1);
                    } else if (action == 2) {
                        parser.rightArc(newConfig.state, label);
                        newConfig.addAction(3 + label);
                    } else if (action == 3) {
                        parser.leftArc(newConfig.state, label);
                        newConfig.addAction(3 + dependencyRelations.size() + label);
                    } else if (action == 4) {
                        parser.unShift(newConfig.state);
                        newConfig.addAction(2);
                    }
                    newConfig.setScore(score);
                    repBeam.add(newConfig);
                }
                beam = repBeam;
            } else {
                Configuration configuration = beam.get(0);
                State currentState = configuration.state;
                boolean canShift = parser.canDo(Actions.Shift, currentState);
                boolean canReduce = parser.canDo(Actions.Reduce, currentState);
                boolean canRightArc = parser.canDo(Actions.RightArc, currentState);
                boolean canLeftArc = parser.canDo(Actions.LeftArc, currentState);
                double[] labels = new double[network.getNumOutputs()];
                if (!canShift) labels[0] = -1;
                if (!canReduce) labels[1] = -1;
                if (!canRightArc)
                    for (int i = 0; i < dependencyRelations.size(); i++)
                        labels[2 + i] = -1;
                if (!canLeftArc)
                    for (int i = 0; i < dependencyRelations.size(); i++)
                        labels[dependencyRelations.size() + 2 + i] = -1;
                double[] features = FeatureExtractor.extractFeatures(configuration, labelNullIndex, parser);
                double[] scores = network.output(features, labels);
                double bestScore = Double.NEGATIVE_INFINITY;
                int bestAction = -1;


                if (!canShift
                        && !canReduce
                        && !canRightArc
                        && !canLeftArc) {

                    if (!currentState.stackEmpty()) {
                        parser.unShift(currentState);
                        configuration.addAction(2);
                    } else if (!currentState.bufferEmpty() && currentState.stackEmpty()) {
                        parser.shift(currentState);
                        configuration.addAction(0);
                    }
                }

                if (canShift) {
                    double score = scores[0];
                    if (score > bestScore) {
                        bestScore = score;
                        bestAction = 0;
                    }
                }
                if (canReduce) {
                    double score = scores[1];
                    if (score > bestScore) {
                        bestScore = score;
                        bestAction = 1;
                    }
                }
                if (canRightArc) {
                    for (int dependency : dependencyRelations) {
                        double score = scores[2 + dependency];
                        if (score > bestScore) {
                            bestScore = score;
                            bestAction = 3 + dependency;
                        }
                    }
                }
                if (parser.canDo(Actions.LeftArc, currentState)) {
                    for (int dependency : dependencyRelations) {
                        double score = scores[2 + dependencyRelations.size() + dependency];
                        if (score > bestScore) {
                            bestScore = score;
                            bestAction = 3 + dependencyRelations.size() + dependency;
                        }
                    }
                }

                if (bestAction != -1) {
                    if (bestAction == 0) {
                        parser.shift(configuration.state);
                    } else if (bestAction == (1)) {
                        parser.reduce(configuration.state);
                    } else {

                        if (bestAction >= 3 + dependencyRelations.size()) {
                            int label = bestAction - (3 + dependencyRelations.size());
                            parser.leftArc(configuration.state, label);
                        } else {
                            int label = bestAction - 3;
                            parser.rightArc(configuration.state, label);
                        }
                    }
                    configuration.addScore(bestScore);
                    configuration.addAction(bestAction);
                }
                if (beam.size() == 0) {
                    System.out.println("WHY BEAM SIZE ZERO?");
                }
            }
        }

        Configuration bestConfiguration = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (Configuration configuration : beam) {
            if (configuration.getScore(true) > bestScore) {
                bestScore = configuration.getScore(true);
                bestConfiguration = configuration;
            }
        }
        return new Pair<>(bestConfiguration, id);
    }

    public Configuration parsePartial() throws Exception {
        Configuration initialConfiguration = new Configuration(sentence, rootFirst);
        boolean isNonProjective = false;
        if (goldConfiguration.isNonprojective()) {
            isNonProjective = true;
        }

        ArrayList<Configuration> beam = new ArrayList<>(beamWidth);
        beam.add(initialConfiguration);

        while (!parser.isTerminal(beam)) {
            TreeSet<BeamElement> beamPreserver = new TreeSet<>();

            parsePartialWithOneThread(beam, beamPreserver, isNonProjective, goldConfiguration, beamWidth);

            ArrayList<Configuration> repBeam = new ArrayList<>(beamWidth);
            for (BeamElement beamElement : beamPreserver.descendingSet()) {
                if (repBeam.size() >= beamWidth)
                    break;
                int b = beamElement.number;
                int action = beamElement.action;
                int label = beamElement.label;
                double score = beamElement.score;

                Configuration newConfig = beam.get(b).clone();

                if (action == 0) {
                    parser.shift(newConfig.state);
                    newConfig.addAction(0);
                } else if (action == 1) {
                    parser.reduce(newConfig.state);
                    newConfig.addAction(1);
                } else if (action == 2) {
                    parser.rightArc(newConfig.state, label);
                    newConfig.addAction(3 + label);
                } else if (action == 3) {
                    parser.leftArc(newConfig.state, label);
                    newConfig.addAction(3 + dependencyRelations.size() + label);
                } else if (action == 4) {
                    parser.unShift(newConfig.state);
                    newConfig.addAction(2);
                }
                newConfig.setScore(score);
                repBeam.add(newConfig);
            }
            beam = repBeam;
        }

        Configuration bestConfiguration = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (Configuration configuration : beam) {
            if (configuration.getScore(true) > bestScore) {
                bestScore = configuration.getScore(true);
                bestConfiguration = configuration;
            }
        }
        return bestConfiguration;
    }

    private void parsePartialWithOneThread(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver, Boolean
            isNonProjective, GoldConfiguration goldConfiguration, int beamWidth) throws Exception {
        for (int b = 0; b < beam.size(); b++) {
            Configuration configuration = beam.get(b);
            State currentState = configuration.state;
            double prevScore = configuration.score;
            boolean canShift = parser.canDo(Actions.Shift, currentState);
            boolean canReduce = parser.canDo(Actions.Reduce, currentState);
            boolean canRightArc = parser.canDo(Actions.RightArc, currentState);
            boolean canLeftArc = parser.canDo(Actions.LeftArc, currentState);
            double[] labels = new double[network.getNumOutputs()];
            if (!canShift) labels[0] = -1;
            if (!canReduce) labels[1] = -1;
            if (!canRightArc)
                for (int i = 0; i < dependencyRelations.size(); i++)
                    labels[2 + i] = -1;
            if (!canLeftArc)
                for (int i = 0; i < dependencyRelations.size(); i++)
                    labels[dependencyRelations.size() + 2 + i] = -1;
            double[] features = FeatureExtractor.extractFeatures(configuration, labelNullIndex, parser);
            double[] scores = network.output(features, labels);
            if (!canShift
                    && !canReduce
                    && !canRightArc
                    && !canLeftArc) {
                beamPreserver.add(new BeamElement(prevScore, b, 4, -1));

                if (beamPreserver.size() > beamWidth)
                    beamPreserver.pollFirst();
            }

            if (canShift) {
                if (isNonProjective || goldConfiguration.actionCost(Actions.Shift, -1, currentState, parser) == 0) {
                    double score = scores[0];
                    double addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 0, -1));

                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
            }

            if (canReduce) {
                if (isNonProjective || goldConfiguration.actionCost(Actions.Reduce, -1, currentState, parser) == 0) {
                    double score = scores[1];
                    double addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 1, -1));

                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
            }

            if (canRightArc) {
                for (int dependency : dependencyRelations) {
                    if (isNonProjective || goldConfiguration.actionCost(Actions.RightArc, dependency, currentState, parser) == 0) {
                        double score = scores[2 + dependency];
                        double addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));

                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
            }

            if (canLeftArc) {
                for (int dependency : dependencyRelations) {
                    if (isNonProjective || goldConfiguration.actionCost(Actions.LeftArc, dependency, currentState, parser) == 0) {
                        double score = scores[2 + dependencyRelations.size() + dependency];
                        double addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));

                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
            }
        }

        if (beamPreserver.size() == 0) {
            for (int b = 0; b < beam.size(); b++) {
                Configuration configuration = beam.get(b);
                State currentState = configuration.state;
                double prevScore = configuration.score;
                boolean canShift = parser.canDo(Actions.Shift, currentState);
                boolean canReduce = parser.canDo(Actions.Reduce, currentState);
                boolean canRightArc = parser.canDo(Actions.RightArc, currentState);
                boolean canLeftArc = parser.canDo(Actions.LeftArc, currentState);
                double[] labels = new double[network.getNumOutputs()];
                if (!canShift) labels[0] = -1;
                if (!canReduce) labels[1] = -1;
                if (!canRightArc)
                    for (int i = 0; i < dependencyRelations.size(); i++)
                        labels[2 + i] = -1;
                if (!canLeftArc)
                    for (int i = 0; i < dependencyRelations.size(); i++)
                        labels[dependencyRelations.size() + 2 + i] = -1;
                double[] features = FeatureExtractor.extractFeatures(configuration, labelNullIndex, parser);
                double[] scores = network.output(features, labels);
                if (!canShift
                        && !canReduce
                        && !canRightArc
                        && !canLeftArc) {
                    beamPreserver.add(new BeamElement(prevScore, b, 4, -1));

                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }

                if (canShift) {
                    double score = scores[0];
                    double addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 0, -1));

                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }

                if (canReduce) {
                    double score = scores[1];
                    double addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 1, -1));

                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }

                if (canRightArc) {
                    for (int dependency : dependencyRelations) {
                        double score = scores[2 + dependency];
                        double addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));

                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }
                }

                if (canLeftArc) {
                    for (int dependency : dependencyRelations) {
                        double score = scores[2 + dependencyRelations.size() + dependency];
                        double addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));

                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
            }
        }
    }
}
