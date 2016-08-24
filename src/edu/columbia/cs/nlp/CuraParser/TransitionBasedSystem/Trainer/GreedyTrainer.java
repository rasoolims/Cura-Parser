/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Trainer;

import edu.columbia.cs.nlp.CuraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.CuraParser.Accessories.Evaluator;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPTrainer;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.AveragingOption;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.UpdaterType;
import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.Structures.NeuralTrainingInstance;
import edu.columbia.cs.nlp.CuraParser.Structures.Pair;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.BeamElement;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.State;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Features.FeatureExtractor;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Beam.BeamParser;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.Actions;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.ParserType;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcEager;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcStandard;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.FileOutputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.*;
import java.util.zip.GZIPOutputStream;

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

    public static void trainWithNN(Options options) throws Exception {
        if (options.trainingOptions.trainFile.equals("") || options.generalProperties.modelFile.equals("")) {
            Options.showHelp();
        } else {
            IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled,
                    options.generalProperties.lowercase, options.trainingOptions.clusterFile, options.trainingOptions.minFreq);
            int wDim = options.networkProperties.wDim;
            if (options.trainingOptions.wordEmbeddingFile.length() > 0)
                wDim = maps.readEmbeddings(options.trainingOptions.wordEmbeddingFile);

            CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
            ArrayList<GoldConfiguration> dataSet =
                    reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options.generalProperties.rootFirst,
                            options.generalProperties.lowercase, maps);
            System.out.println("CoNLL data reading done!");

            ArrayList<Integer> dependencyLabels = new ArrayList<>();
            for (int lab = 0; lab < maps.relSize(); lab++)
                dependencyLabels.add(lab);

            System.out.println("size of training data (#sens): " + dataSet.size());
            System.out.println("Embedding dimension " + wDim);
            GreedyTrainer trainer = new GreedyTrainer( options, dependencyLabels,
                    maps.labelNullIndex, maps.rareWords);
            ArrayList<NeuralTrainingInstance> allInstances = trainer.getNextInstances(dataSet, 0, dataSet.size(), 0);
            int numWordLayers = options.generalProperties.parserType == ParserType.ArcEager ? 22 : 20;
            maps.constructPreComputeMap(allInstances, numWordLayers, 10000);

            MLPNetwork mlpNetwork = new MLPNetwork(maps, options, dependencyLabels, wDim, options.networkProperties.posDim,
                    options.networkProperties.depDim, options.generalProperties.parserType);
            MLPNetwork avgMlpNetwork = new MLPNetwork(maps, options, dependencyLabels, wDim, options.networkProperties.posDim,
                    options.networkProperties.depDim, options.generalProperties.parserType);
            maps.emptyEmbeddings();

            MLPTrainer neuralTrainer = new MLPTrainer(mlpNetwork, options);

            double bestModelUAS = 0;
            Random random = new Random();
            int decayStep = (int) (options.trainingOptions.decayStep * allInstances.size() / options.networkProperties.batchSize);
            decayStep = decayStep == 0 ? 1 : decayStep;
            System.out.println("Data has " + allInstances.size() + " instances");
            System.out.println("Decay after every " + decayStep + " batches");
            for (int step = 0; step < options.trainingOptions.trainingIter; step++) {
                List<NeuralTrainingInstance> instances = Utils.getRandomSubset(allInstances, random, options.networkProperties.batchSize);
                try {
                    neuralTrainer.fit(instances, step, step % (Math.max(1, options.trainingOptions.UASEvalPerStep / 10)) == 0 ? true : false);
                } catch (Exception ex) {
                    System.err.println("Exception occurred: " + ex.getMessage());
                    ex.printStackTrace();
                    System.exit(1);
                }
                if (options.updaterProperties.updaterType == UpdaterType.SGD) {
                    if (step % decayStep == 0) {
                        neuralTrainer.setLearningRate(0.96 * neuralTrainer.getLearningRate());
                        System.out.println("The new learning rate: " + neuralTrainer.getLearningRate());
                    }
                }

                if (options.trainingOptions.averagingOption != AveragingOption.NO) {
                    // averaging
                    double ratio = Math.min(0.9999, (double) step / (9 + step));
                    mlpNetwork.averageNetworks(avgMlpNetwork, 1 - ratio, step == 1 ? 0 : ratio);
                }

                if (step % options.trainingOptions.UASEvalPerStep == 0) {
                    if (options.trainingOptions.averagingOption != AveragingOption.ONLY) {
                        bestModelUAS = evaluate(options, mlpNetwork, bestModelUAS);
                    }
                    if (options.trainingOptions.averagingOption != AveragingOption.NO) {
                        avgMlpNetwork.preCompute();
                        bestModelUAS = evaluate(options, avgMlpNetwork, bestModelUAS);
                    }
                }
            }
            neuralTrainer.shutDownLiveThreads();
        }
    }

    private static double evaluate(Options options, MLPNetwork mlpNetwork, double bestModelUAS) throws Exception {
        BeamParser parser = new BeamParser(mlpNetwork, options.generalProperties.numOfThreads, options.generalProperties.parserType);
        parser.parseConll(options.trainingOptions.devPath, options.generalProperties.modelFile + ".tmp", options.generalProperties.rootFirst,
                options.generalProperties.beamWidth, options.generalProperties.lowercase,
                options.generalProperties.numOfThreads,
                false, "");
        Pair<Double, Double> eval = Evaluator.evaluate(options.trainingOptions.devPath, options.generalProperties.modelFile + ".tmp",
                options.generalProperties.punctuations);
        if (eval.first > bestModelUAS) {
            bestModelUAS = eval.first;
            System.out.print("Saving the new model...");
            FileOutputStream fos = new FileOutputStream(options.generalProperties.modelFile);
            GZIPOutputStream gz = new GZIPOutputStream(fos);
            ObjectOutput writer = new ObjectOutputStream(gz);
            writer.writeObject(mlpNetwork);
            writer.writeObject(options);
            writer.close();
            System.out.print("done!\n\n");
        }
        return bestModelUAS;
    }
}