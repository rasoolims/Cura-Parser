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
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Features.FeatureExtractor;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Beam.BeamParser;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.Actions;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.ParserType;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcEager;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcStandard;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class GreedyTrainer {
    final HashSet<Integer> rareWords;
    protected ArrayList<Integer> dependencyRelations;
    protected int labelNullIndex;
    Options options;
    Random random;
    ShiftReduceParser parser;

    public GreedyTrainer(Options options, ArrayList<Integer> dependencyRelations, int labelNullIndex, HashSet<Integer> rareWords) {
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

    public static void trainWithNN(Options options) throws Exception {
        if (options.trainingOptions.trainFile.equals("") || options.generalProperties.modelFile.equals("")) {
            Options.showHelp();
        } else {
            if (options.trainingOptions.pretrainLayers && options.networkProperties.hiddenLayer2Size != 0) {
                trainMultiLayerNetwork(options);
            } else {
                train(options);
            }
        }
    }

    private static void trainMultiLayerNetwork(Options options) throws Exception {
        Options oneLayerOption = options.clone();
        oneLayerOption.networkProperties.hiddenLayer2Size = 0;
        oneLayerOption.generalProperties.beamWidth = 1;
        oneLayerOption.trainingOptions.trainingIter = options.trainingOptions.preTrainingIter;
        String modelFile = options.trainingOptions.preTrainedModelPath.equals("") ? options.generalProperties.modelFile : options.trainingOptions
                .preTrainedModelPath;

        if (options.trainingOptions.preTrainedModelPath.equals("")) {
            System.out.println("First training with one hidden layer!");
            train(oneLayerOption);
        }

        System.out.println("Loading model with one hidden layer!");
        FileInputStream fos = new FileInputStream(modelFile);
        GZIPInputStream gz = new GZIPInputStream(fos);
        ObjectInput reader = new ObjectInputStream(gz);
        MLPNetwork mlpNetwork = (MLPNetwork) reader.readObject();
        reader.close();

        System.out.println("Now Training with two layers!");
        Options twoLayerOptions = options.clone();
        twoLayerOptions.generalProperties.beamWidth = 1;
        MLPNetwork net = constructMlpNetwork(twoLayerOptions);
        // Putting the first layer into it!
        net.layer(0).setLayer(mlpNetwork.layer(0));
        trainNetwork(twoLayerOptions, net);
    }

    private static void train(Options options) throws Exception {
        Options greedyOptions = options.clone();
        greedyOptions.generalProperties.beamWidth = 1;
        MLPNetwork mlpNetwork = constructMlpNetwork(greedyOptions);
        trainNetwork(greedyOptions, mlpNetwork);
    }

    private static void trainNetwork(Options options, MLPNetwork mlpNetwork) throws Exception {
        MLPNetwork avgMlpNetwork = new MLPNetwork(mlpNetwork.maps, options, mlpNetwork.getDepLabels(), mlpNetwork.getwDim(),
                options.networkProperties.posDim, options.networkProperties.depDim, options.generalProperties.parserType);

        GreedyTrainer trainer = new GreedyTrainer(options, mlpNetwork.getDepLabels(), mlpNetwork.maps.labelNullIndex, mlpNetwork.maps.rareWords);
        CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
        ArrayList<GoldConfiguration> dataSet =
                reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options.generalProperties.rootFirst,
                        options.generalProperties.lowercase, mlpNetwork.maps);
        System.out.println("CoNLL data reading done!");
        System.out.println("size of training data (#sens): " + dataSet.size());

        ArrayList<NeuralTrainingInstance> allInstances = trainer.getNextInstances(dataSet, 0, dataSet.size(), 0);
        int numWordLayers = options.generalProperties.parserType == ParserType.ArcEager ? 22 : 20;
        mlpNetwork.maps.constructPreComputeMap(allInstances, numWordLayers, 10000);
        mlpNetwork.resetPreComputeMap();
        avgMlpNetwork.resetPreComputeMap();
        mlpNetwork.maps.emptyEmbeddings();

        MLPTrainer neuralTrainer = new MLPTrainer(mlpNetwork, options);

        double bestModelUAS = 0;
        Random random = new Random();
        System.out.println("Data has " + allInstances.size() + " instances");
        System.out.println("Decay after every " + options.trainingOptions.decayStep + " batches");
        int step;
        for (step = 0; step < options.trainingOptions.trainingIter; step++) {
            List<NeuralTrainingInstance> instances = Utils.getRandomSubset(allInstances, random, options.networkProperties.batchSize);
            try {
                neuralTrainer.fit(instances, step, step % (Math.max(1, options.trainingOptions.UASEvalPerStep / 10)) == 0);
            } catch (Exception ex) {
                System.err.println("Exception occurred: " + ex.getMessage());
                ex.printStackTrace();
                System.exit(1);
            }
            if (options.updaterProperties.updaterType == UpdaterType.SGD) {
                if ((step + 1) % options.trainingOptions.decayStep == 0) {
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

        if (options.trainingOptions.averagingOption != AveragingOption.NO) {
            // averaging
            double ratio = Math.min(0.9999, (double) step / (9 + step));
            mlpNetwork.averageNetworks(avgMlpNetwork, 1 - ratio, step == 1 ? 0 : ratio);
        }

        if (options.trainingOptions.averagingOption != AveragingOption.ONLY) {
            bestModelUAS = evaluate(options, mlpNetwork, bestModelUAS);
        }
        if (options.trainingOptions.averagingOption != AveragingOption.NO) {
            avgMlpNetwork.preCompute();
            bestModelUAS = evaluate(options, avgMlpNetwork, bestModelUAS);
        }
        neuralTrainer.shutDownLiveThreads();
    }

    private static MLPNetwork constructMlpNetwork(Options options) throws Exception {
        IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled,
                options.generalProperties.lowercase, options.trainingOptions.clusterFile, options.trainingOptions.minFreq,
                options.generalProperties.includePosAsUnknown);
        int wDim = options.networkProperties.wDim;
        if (options.trainingOptions.wordEmbeddingFile.length() > 0)
            wDim = maps.readEmbeddings(options.trainingOptions.wordEmbeddingFile);

        ArrayList<Integer> dependencyLabels = new ArrayList<>();
        for (int lab = 0; lab < maps.relSize(); lab++)
            dependencyLabels.add(lab);

        System.out.println("Embedding dimension " + wDim);
        return new MLPNetwork(maps, options, dependencyLabels, wDim, options.networkProperties.posDim,
                options.networkProperties.depDim, options.generalProperties.parserType);
    }

    protected static double evaluate(Options options, MLPNetwork mlpNetwork, double bestModelUAS) throws Exception {
        System.out.println("Evaluating with " + options.generalProperties.beamWidth + " beams!");
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

    public static void outputTrainInstances(Options options) throws Exception {
        Options greedyOptions = options.clone();
        greedyOptions.generalProperties.beamWidth = 1;
        MLPNetwork mlpNetwork = constructMlpNetwork(greedyOptions);
        GreedyTrainer trainer = new GreedyTrainer(options, mlpNetwork.getDepLabels(), mlpNetwork.maps.labelNullIndex, mlpNetwork.maps.rareWords);
        CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
        ArrayList<GoldConfiguration> dataSet =
                reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options.generalProperties.rootFirst,
                        options.generalProperties.lowercase, mlpNetwork.maps);
        System.out.println("CoNLL data reading done!");
        System.out.println("size of training data (#sens): " + dataSet.size());

        ArrayList<NeuralTrainingInstance> allInstances = trainer.getNextInstances(dataSet, 0, dataSet.size(), 0);
        BufferedWriter writer = new BufferedWriter(new FileWriter(options.generalProperties.outputFile));

        for (NeuralTrainingInstance instance : allInstances) {
            writer.write(instance.toString());
        }
        writer.close();
    }

    private void addInstance(GoldConfiguration goldConfiguration, ArrayList<NeuralTrainingInstance> instances, double dropWordProb) throws Exception {
        Configuration initialConfiguration = new Configuration(goldConfiguration.getSentence(), options.generalProperties.rootFirst);
        Configuration firstOracle = initialConfiguration.clone();
        ArrayList<Configuration> beam = new ArrayList<>(1);
        beam.add(initialConfiguration);

        Configuration oracle = firstOracle;

        while (!parser.isTerminal(beam) && beam.size() > 0) {
            double[] baseFeatures = FeatureExtractor.extractFeatures(oracle, labelNullIndex, parser);
            double[] label = new double[2 * (dependencyRelations.size() + 1)];
            if (!options.trainingOptions.considerAllActions && !parser.canDo(Actions.LeftArc, oracle.state)) {
                for (int i = 2; i < 2 + dependencyRelations.size(); i++)
                    label[i + dependencyRelations.size()] = -1;
            }
            if (!options.trainingOptions.considerAllActions && !parser.canDo(Actions.RightArc, oracle.state)) {
                for (int i = 2; i < 2 + dependencyRelations.size(); i++)
                    label[i] = -1;
            }
            if (!options.trainingOptions.considerAllActions && !parser.canDo(Actions.Shift, oracle.state)) {
                label[0] = -1;
            }
            if ((!options.trainingOptions.considerAllActions && !parser.canDo(Actions.Reduce, oracle.state))
                    || options.generalProperties.parserType == ParserType.ArcStandard) {
                label[1] = -1;
            }

            oracle = parser.staticOracle(goldConfiguration, oracle, dependencyRelations.size());
            int action = oracle.actionHistory.get(oracle.actionHistory.size() - 1);
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
            beam = new ArrayList<>(1);
            beam.add(oracle);
        }
    }

    public ArrayList<NeuralTrainingInstance> getNextInstances(ArrayList<GoldConfiguration> trainData, int start, int end, double dropWordProb)
            throws Exception {
        ArrayList<NeuralTrainingInstance> instances = new ArrayList<>();
        for (int i = start; i < end; i++) {
            addInstance(trainData.get(i), instances, dropWordProb);
        }
        return instances;
    }
}