/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.YaraParser;

import edu.columbia.cs.nlp.YaraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.YaraParser.Accessories.Evaluator;
import edu.columbia.cs.nlp.YaraParser.Accessories.Options;
import edu.columbia.cs.nlp.YaraParser.Accessories.Utils;
import edu.columbia.cs.nlp.YaraParser.Learning.Activation.Enums.ActivationType;
import edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.MLPTrainer;
import edu.columbia.cs.nlp.YaraParser.Learning.Updater.Enums.AveragingOption;
import edu.columbia.cs.nlp.YaraParser.Learning.Updater.Enums.UpdaterType;
import edu.columbia.cs.nlp.YaraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.YaraParser.Structures.NeuralTrainingInstance;
import edu.columbia.cs.nlp.YaraParser.Structures.Pair;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Beam.BeamParser;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Enums.ParserType;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Trainer.BeamTrainer;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class YaraParser {
    public static void main(String[] args) throws Exception {
        Options options = Options.processArgs(args);

        if (args.length < 2) {
            options.generalProperties.train = true;
            options.trainingOptions.trainFile = "/Users/msr/Desktop/data/dev_smal.conll";
            options.trainingOptions.devPath = "/Users/msr/Desktop/data/train_smal.conll";
            options.trainingOptions.wordEmbeddingFile = "/Users/msr/Desktop/data/word.embed";
            options.trainingOptions.clusterFile = "/Users/msr/Downloads/trained_freqw+clusters_1k.cbow.en";
            options.generalProperties.modelFile = "/tmp/model";
            options.generalProperties.outputFile = "/tmp/model.out";
            options.generalProperties.labeled = true;
            options.networkProperties.hiddenLayer1Size = 200;
            options.updaterProperties.learningRate = 0.001;
            options.networkProperties.batchSize = 1024;
            options.trainingOptions.trainingIter = 100;
            options.generalProperties.beamWidth = 1;
            options.trainingOptions.useDynamicOracle = false;
            options.generalProperties.numOfThreads = 2;
            options.trainingOptions.decayStep = 10;
            options.trainingOptions.UASEvalPerStep = 10;
            options.updaterProperties.updaterType = UpdaterType.ADAM;
            options.trainingOptions.averagingOption = AveragingOption.BOTH;
            options.networkProperties.activationType = ActivationType.RELU;
            options.generalProperties.parserType = ParserType.ArcStandard;
        }

        if (options.generalProperties.showHelp) {
            Options.showHelp();
        } else {
            System.out.println(options);
            if (options.generalProperties.train) {
                trainWithNN(options);
            } else if (options.generalProperties.parseTaggedFile || options.generalProperties.parseConllFile
                    || options.generalProperties.parsePartialConll) {
                parse(options);
            } else if (options.generalProperties.evaluate) {
                evaluate(options);
            } else {
                Options.showHelp();
            }
        }
        System.exit(0);
    }

    private static void evaluate(Options options) throws Exception {
        if (options.generalProperties.inputFile.equals("") || options.generalProperties.outputFile.equals(""))
            Options.showHelp();
        else {
            Evaluator.evaluate(options.generalProperties.inputFile, options.generalProperties.outputFile, options.generalProperties.punctuations);
        }
    }

    private static void parse(Options options) throws Exception {
        if (options.generalProperties.outputFile.equals("") || options.generalProperties.inputFile.equals("")
                || options.generalProperties.modelFile.equals("")) {
            Options.showHelp();

        } else {
            FileInputStream fos = new FileInputStream(options.generalProperties.modelFile);
            GZIPInputStream gz = new GZIPInputStream(fos);
            ObjectInput reader = new ObjectInputStream(gz);
            MLPNetwork mlpNetwork = (MLPNetwork) reader.readObject();
            Options infoptions = (Options) reader.readObject();
            BeamParser parser = new BeamParser(mlpNetwork, options.generalProperties.numOfThreads, infoptions.generalProperties.parserType);

            if (options.generalProperties.parseTaggedFile)
                parser.parseTaggedFile(options.generalProperties.inputFile,
                        options.generalProperties.outputFile, infoptions.generalProperties.rootFirst, options.generalProperties.beamWidth,
                        infoptions.generalProperties.lowercase,
                        options.separator, options.generalProperties.numOfThreads);
            else if (options.generalProperties.parseConllFile)
                parser.parseConll(options.generalProperties.inputFile, options.generalProperties.outputFile, infoptions.generalProperties
                                .rootFirst, options.generalProperties.beamWidth, infoptions.generalProperties.lowercase,
                        options.generalProperties.numOfThreads, false, options.scorePath);
            else if (options.generalProperties.parsePartialConll)
                parser.parseConll(options.generalProperties.inputFile, options.generalProperties.outputFile, infoptions.generalProperties
                                .rootFirst, options.generalProperties.beamWidth, infoptions.generalProperties.lowercase,
                        options.generalProperties.numOfThreads, true, options.scorePath);
            parser.shutDownLiveThreads();
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
            BeamTrainer trainer = new BeamTrainer(options.trainingOptions.useMaxViol ? "max_violation" : "early", options, dependencyLabels,
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
                    MLPNetwork.averageNetworks(mlpNetwork, avgMlpNetwork, 1 - ratio, step == 1 ? 0 : ratio);
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
