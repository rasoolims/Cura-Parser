/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Trainer;

import edu.columbia.cs.nlp.YaraParser.Accessories.Options;
import edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.YaraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.YaraParser.Structures.NeuralTrainingInstance;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.BeamElement;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.State;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Features.FeatureExtractor;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Enums.Actions;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Enums.ParserType;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Parsers.ArcEager;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Parsers.ArcStandard;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.*;

public class BeamTrainer {
    final HashSet<Integer> rareWords;
    Options options;
    Random random;
    ShiftReduceParser parser;
    private String updateMode;
    private ArrayList<Integer> dependencyRelations;
    private int labelNullIndex;

    public BeamTrainer(String updateMode, Options options, ArrayList<Integer> dependencyRelations, int labelNullIndex, HashSet<Integer>
            rareWords) throws Exception {
        this.updateMode = updateMode;
        this.options = options;
        this.dependencyRelations = dependencyRelations;
        this.labelNullIndex = labelNullIndex;
        random = new Random();
        this.rareWords = rareWords;
        if (options.parserType == ParserType.ArcEager)
            parser = new ArcEager();
        else if (options.parserType == ParserType.ArcStandard)
            parser = new ArcStandard();
        else
            throw new NotImplementedException();
    }

    public ArrayList<NeuralTrainingInstance> getNextInstances(ArrayList<GoldConfiguration> trainData, int start, int end, double dropWordProb)
            throws Exception {
        ArrayList<NeuralTrainingInstance> instances = new ArrayList<>();
        for (int i = start; i < end; i++) {
            addInstance(trainData.get(i), instances, dropWordProb);
        }
        return instances;
    }

    private void addInstance(GoldConfiguration goldConfiguration, ArrayList<NeuralTrainingInstance> instances, double dropWordProb) throws Exception {
        Configuration initialConfiguration = new Configuration(goldConfiguration.getSentence(), options.rootFirst);
        Configuration firstOracle = initialConfiguration.clone();
        ArrayList<Configuration> beam = new ArrayList<Configuration>(options.beamWidth);
        beam.add(initialConfiguration);

        HashMap<Configuration, Double> oracles = new HashMap<>();

        oracles.put(firstOracle, 0.0);

        Configuration bestScoringOracle = null;

        while (!parser.isTerminal(beam) && beam.size() > 0) {
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

            int[] baseFeatures = FeatureExtractor.extractBaseFeatures(currentConfig, labelNullIndex, parser);
            int[] label = new int[2 * (dependencyRelations.size() + 1)];
            if (!parser.canDo(Actions.LeftArc, currentConfig.state)) {
                for (int i = 2; i < 2 + dependencyRelations.size(); i++)
                    label[i + dependencyRelations.size()] = -1;
            }
            if (!parser.canDo(Actions.RightArc, currentConfig.state)) {
                for (int i = 2; i < 2 + dependencyRelations.size(); i++)
                    label[i] = -1;
            }
            if (!parser.canDo(Actions.Shift, currentConfig.state)) {
                label[0] = -1;
            }
            if (!parser.canDo(Actions.Reduce, currentConfig.state)) {
                label[1] = -1;
            }

            bestScoringOracle = parser.staticOracle(goldConfiguration, oracles, newOracles, dependencyRelations.size());
            oracles = newOracles;
            int action = bestScoringOracle.actionHistory.get(bestScoringOracle.actionHistory.size() - 1);
            if (action >= 2)
                action -= 1;

            int numWordLayers = parser instanceof ArcEager ? 22 : 20;
            for (int i = 0; i < baseFeatures.length; i++) {
                if (i < numWordLayers && rareWords.contains(baseFeatures[i]))
                    if (random.nextDouble() <= dropWordProb && baseFeatures[i] != 1)
                        baseFeatures[i] = IndexMaps.UnknownIndex;
            }

            label[action] = 1;
            instances.add(new NeuralTrainingInstance(baseFeatures, label));
            beam = new ArrayList<>(options.beamWidth);
            beam.add(bestScoringOracle);
        }
    }

    private void beamSortOneThread(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver, MLPNetwork network) throws Exception {
        for (int b = 0; b < beam.size(); b++) {
            Configuration configuration = beam.get(b);
            State currentState = configuration.state;
            double prevScore = configuration.score;
            boolean canShift = parser.canDo(Actions.Shift, currentState);
            boolean canReduce = parser.canDo(Actions.Reduce, currentState);
            boolean canRightArc = parser.canDo(Actions.RightArc, currentState);
            boolean canLeftArc = parser.canDo(Actions.LeftArc, currentState);

            int[] labels = new int[network.getSoftmaxLayerDim()];
            if (!canShift) labels[0] = -1;
            if (!canReduce) labels[1] = -1;
            if (!canRightArc)
                for (int i = 0; i < dependencyRelations.size(); i++)
                    labels[2 + i] = -1;
            if (!canLeftArc)
                for (int i = 0; i < dependencyRelations.size(); i++)
                    labels[dependencyRelations.size() + 2 + i] = -1;
            int[] features = FeatureExtractor.extractBaseFeatures(configuration, labelNullIndex, parser);
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