/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package YaraParser.Parser;

import YaraParser.Accessories.CoNLLReader;
import YaraParser.Accessories.Evaluator;
import YaraParser.Accessories.Options;
import YaraParser.Accessories.Pair;
import YaraParser.Learning.Activation.ActivationType;
import YaraParser.Learning.AveragingOption;
import YaraParser.Learning.NeuralNetwork.MLPNetwork;
import YaraParser.Learning.NeuralNetwork.MLPTrainer;
import YaraParser.Learning.Updater.UpdaterType;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.NeuralTrainingInstance;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Parser.KBeamArcEagerParser;
import YaraParser.TransitionBasedSystem.Trainer.ArcEagerBeamTrainer;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class YaraParser {
    public static void main(String[] args) throws Exception {
        Options options = Options.processArgs(args);

        if (args.length < 2) {
            options.train = true;
            options.inputFile = "/Users/msr/Desktop/data/dev_smal.conll";
            options.devPath = "/Users/msr/Desktop/data/train_smal.conll";
            options.wordEmbeddingFile = "/Users/msr/Desktop/data/word.embed";
            // options.clusterFile = "/Users/msr/Desktop/data/brown-rcv1.clean.tokenized-CoNLL03.txt-c1000-freq1.txt";
            options.modelFile = "/tmp/model";
            options.labeled = true;
            options.hiddenLayer1Size = 200;
            options.learningRate = 0.001;
            options.batchSize = 1024;
            options.trainingIter = 3000;
            options.beamWidth = 1;
            options.useDynamicOracle = false;
            options.numOfThreads = 1;
            options.UASEvalPerStep = 10;
            options.updaterType = UpdaterType.ADAM;
            options.averagingOption = AveragingOption.BOTH;
            options.activationType = ActivationType.RELU;
        }

        if (options.showHelp) {
            Options.showHelp();
        } else {
            System.out.println(options);
            if (options.train) {
                trainWithNN(options);
            } else if (options.parseTaggedFile || options.parseConllFile || options.parsePartialConll) {
                parse(options);
            } else if (options.evaluate) {
                evaluate(options);
            } else {
                Options.showHelp();
            }
        }
        System.exit(0);
    }

    private static void evaluate(Options options) throws Exception {
        if (options.goldFile.equals("") || options.predFile.equals(""))
            Options.showHelp();
        else {
            Evaluator.evaluate(options.goldFile, options.predFile, options.punctuations);
        }
    }

    private static void parse(Options options) throws Exception {
        if (options.outputFile.equals("") || options.inputFile.equals("")
                || options.modelFile.equals("")) {
            Options.showHelp();

        } else {
            FileInputStream fos = new FileInputStream(options.modelFile);
            GZIPInputStream gz = new GZIPInputStream(fos);
            ObjectInput reader = new ObjectInputStream(gz);
            MLPNetwork mlpNetwork = (MLPNetwork) reader.readObject();
            Options infoptions = (Options) reader.readObject();
            KBeamArcEagerParser parser = new KBeamArcEagerParser(mlpNetwork, options.numOfThreads);

            if (options.parseTaggedFile)
                parser.parseTaggedFile(options.inputFile,
                        options.outputFile, infoptions.rootFirst, options.beamWidth, infoptions.lowercase,
                        options.separator, options.numOfThreads);
            else if (options.parseConllFile)
                parser.parseConllFile(options.inputFile,
                        options.outputFile, infoptions.rootFirst, options.beamWidth, true, infoptions.lowercase, false, options.scorePath);
            else if (options.parsePartialConll)
                parser.parseConllFile(options.inputFile, options.outputFile, infoptions.rootFirst, options.beamWidth, infoptions.labeled,
                        infoptions.lowercase, true, options.scorePath);
            parser.shutDownLiveThreads();
        }
    }

    public static void trainWithNN(Options options) throws Exception {
        if (options.inputFile.equals("") || options.modelFile.equals("")) {
            Options.showHelp();
        } else {
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, options.clusterFile, 1);
            int wDim = 64;
            if (options.wordEmbeddingFile.length() > 0)
                wDim = maps.readEmbeddings(options.wordEmbeddingFile);

            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<GoldConfiguration> dataSet =
                    reader.readData(Integer.MAX_VALUE, false, options.labeled, options.rootFirst, options.lowercase, maps);
            System.out.println("CoNLL data reading done!");

            ArrayList<Integer> dependencyLabels = new ArrayList<>();
            for (int lab = 0; lab < maps.relSize(); lab++)
                dependencyLabels.add(lab);

            System.out.println("size of training data (#sens): " + dataSet.size());
            System.out.println("Embedding dimension " + wDim);
            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early",
                    options, dependencyLabels, maps);

            MLPNetwork mlpNetwork = new MLPNetwork(maps, options, dependencyLabels, wDim, 32, 32);
            MLPNetwork avgMlpNetwork = new MLPNetwork(maps, options, dependencyLabels, wDim, 32, 32);

            MLPTrainer neuralTrainer = new MLPTrainer(mlpNetwork, options.updaterType, 0.9, options.learningRate, 0.0001, options.numOfThreads,
                    options.dropoutProbForHiddenLayer);


            int step = 0;
            double bestModelUAS = 0;
            ArrayList<NeuralTrainingInstance> allInstances = trainer.getNextInstances(dataSet, 0, dataSet.size(), 0);
            int decayStep = (int) (options.decayStep * allInstances.size() / options.batchSize);
            decayStep = decayStep == 0 ? 1 : decayStep;
            System.out.println("Decay after every " + decayStep + " batches");
            for (int i = 0; i < options.trainingIter; i++) {
                System.out.println("reshuffling data for round " + i);
                Collections.shuffle(allInstances);
                int s = 0;
                int e = Math.min(allInstances.size(), options.batchSize);

                while (true) {
                    step++;
                    List<NeuralTrainingInstance> instances = allInstances.subList(s, e);
                    neuralTrainer.fit(instances, step, step % (options.UASEvalPerStep / 10) == 0 ? true : false);
                    s = e;
                    e = Math.min(allInstances.size(), options.batchSize + e);

                    if (options.updaterType == UpdaterType.SGD) {
                        if (step % decayStep == 0) {
                            neuralTrainer.setLearningRate(0.96 * neuralTrainer.getLearningRate());
                            System.out.println("The new learning rate: " + neuralTrainer.getLearningRate());
                        }
                    }

                    if (options.averagingOption != AveragingOption.NO) {
                        // averaging
                        double ratio = Math.min(0.9999, (double) step / (9 + step));
                        MLPNetwork.averageNetworks(mlpNetwork, avgMlpNetwork, 1 - ratio, step == 1 ? 0 : ratio);
                    }

                    if (step % options.UASEvalPerStep == 0) {
                        if (options.averagingOption != AveragingOption.ONLY) {
                            KBeamArcEagerParser parser = new KBeamArcEagerParser(mlpNetwork, options.numOfThreads);
                            parser.parseConllFile(options.devPath, options.modelFile + ".tmp", options.rootFirst,
                                    options.beamWidth, options.labeled, options.lowercase, false, "");
                            Pair<Double, Double> eval = Evaluator.evaluate(options.devPath, options.modelFile + ".tmp", options.punctuations);
                            if (eval.first > bestModelUAS) {
                                bestModelUAS = eval.first;
                                System.out.print("Saving the new model...");
                                FileOutputStream fos = new FileOutputStream(options.modelFile);
                                GZIPOutputStream gz = new GZIPOutputStream(fos);
                                ObjectOutput writer = new ObjectOutputStream(gz);
                                writer.writeObject(mlpNetwork);
                                writer.writeObject(dependencyLabels);
                                writer.writeObject(options);
                                writer.close();
                                System.out.print("done!\n\n");
                            }
                        }
                        if (options.averagingOption != AveragingOption.NO) {
                            avgMlpNetwork.preCompute();
                            KBeamArcEagerParser parser = new KBeamArcEagerParser(avgMlpNetwork, options.numOfThreads);
                            parser.parseConllFile(options.devPath, options.modelFile + ".tmp", options.rootFirst,
                                    options.beamWidth, options.labeled, options.lowercase, false, "");
                            Pair<Double, Double> eval = Evaluator.evaluate(options.devPath, options.modelFile + ".tmp", options.punctuations);
                            if (eval.first > bestModelUAS) {
                                bestModelUAS = eval.first;
                                System.out.print("Saving the new model...");
                                FileOutputStream fos = new FileOutputStream(options.modelFile);
                                GZIPOutputStream gz = new GZIPOutputStream(fos);
                                ObjectOutput writer = new ObjectOutputStream(gz);
                                writer.writeObject(avgMlpNetwork);
                                writer.writeObject(options);
                                writer.close();
                                System.out.print("done!\n\n");
                            }
                        }
                    }

                    if (s >= allInstances.size())
                        break;
                }
            }
            neuralTrainer.shutDownLiveThreads();
        }
    }
}
