/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Trainer;

import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.Structures.NeuralTrainingInstance;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.BeamElement;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.State;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Features.FeatureExtractor;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.Actions;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.ParserType;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcEager;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcStandard;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.*;

public class GreedyTrainer {
    final HashSet<Integer> rareWords;
    Options options;
    Random random;
    ShiftReduceParser parser;
    private ArrayList<Integer> dependencyRelations;
    private int labelNullIndex;

    public GreedyTrainer(Options options, ArrayList<Integer> dependencyRelations, int labelNullIndex, HashSet<Integer> rareWords) throws Exception {
        this.options = options;
        this.dependencyRelations = dependencyRelations;
        this.labelNullIndex = labelNullIndex;
        random = new Random();
        this.rareWords = rareWords;
        if (options.generalProperties.parserType == ParserType.ArcEager)
            parser = new ArcEager();
        else if (options.generalProperties.parserType == ParserType.ArcStandard)
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
        Configuration initialConfiguration = new Configuration(goldConfiguration.getSentence(), options.generalProperties.rootFirst);
        Configuration firstOracle = initialConfiguration.clone();
        ArrayList<Configuration> beam = new ArrayList<Configuration>(options.generalProperties.beamWidth);
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

            double[] baseFeatures = FeatureExtractor.extractBaseFeatures(currentConfig, labelNullIndex, parser);
            double[] label = new double[2 * (dependencyRelations.size() + 1)];
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
            beam = new ArrayList<>(options.generalProperties.beamWidth);
            beam.add(bestScoringOracle);
        }
    }
}