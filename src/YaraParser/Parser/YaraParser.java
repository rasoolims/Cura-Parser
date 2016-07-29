/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package YaraParser.Parser;

import YaraParser.Accessories.CoNLLReader;
import YaraParser.Accessories.Evaluator;
import YaraParser.Accessories.Options;
import YaraParser.Accessories.Pair;
import YaraParser.Learning.AveragedPerceptron;
import YaraParser.Learning.MLPClassifier;
import YaraParser.Learning.MLPNetwork;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.InfStruct;
import YaraParser.Structures.NeuralTrainingInstance;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Parser.KBeamArcEagerParser;
import YaraParser.TransitionBasedSystem.Trainer.ArcEagerBeamTrainer;

import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;

public class YaraParser {
    public static void main(String[] args) throws Exception {
        Options options = Options.processArgs(args);

        if (args.length < 2) {
            options.train = true;
            options.inputFile = "/Users/msr/Desktop/data/dev_smal.delex.conll";
            options.devPath = "/Users/msr/Desktop/data/train_smal.delex.conll";
            options.wordEmbeddingFile = "/Users/msr/Desktop/data/word.embed";
            //  options.clusterFile = "/Users/msr/Desktop/data/brown-rcv1.clean.tokenized-CoNLL03.txt-c1000-freq1.txt";
            options.modelFile = "/tmp/model";
            options.labeled = false;
            options.hiddenLayer1Size = 64;
            options.learningRate = 0.2;
            options.batchSize = 32;
            options.trainingIter = 3000;
            options.beamWidth = 1;
            options.decayStep = 3;
            options.useDynamicOracle = false;
        }

        if (options.showHelp) {
            Options.showHelp();
        } else {
            System.out.println(options);
            if (options.train) {
                // train(options);
                //  createTrainData(options);
                trainWithNN(options);
            } else if (options.parseTaggedFile || options.parseConllFile || options.parsePartialConll) {
                parseNN(options);
                // parse(options);
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
            InfStruct infStruct = new InfStruct(options.modelFile);
            ArrayList<Integer> dependencyLabels = infStruct.dependencyLabels;
            IndexMaps maps = infStruct.maps;


            Options inf_options = infStruct.options;
            AveragedPerceptron averagedPerceptron = new AveragedPerceptron(infStruct);

            int featureSize = averagedPerceptron.featureSize();
            KBeamArcEagerParser parser = new KBeamArcEagerParser(averagedPerceptron, dependencyLabels, featureSize,
                    maps, options.numOfThreads);

            if (options.parseTaggedFile)
                parser.parseTaggedFile(options.inputFile,
                        options.outputFile, inf_options.rootFirst, inf_options.beamWidth, inf_options.lowercase,
                        options.separator, options.numOfThreads);
            else if (options.parseConllFile)
                parser.parseConllFile(options.inputFile,
                        options.outputFile, inf_options.rootFirst, inf_options.beamWidth, true, inf_options
                                .lowercase, options.numOfThreads, false, options.scorePath);
            else if (options.parsePartialConll)
                parser.parseConllFile(options.inputFile,
                        options.outputFile, inf_options.rootFirst, inf_options.beamWidth, options.labeled,
                        inf_options.lowercase, options.numOfThreads, true, options.scorePath);
            parser.shutDownLiveThreads();
        }
    }

    private static void parseNN(Options options) throws Exception {
        FileInputStream fos = new FileInputStream(options.modelFile);
        GZIPInputStream gz = new GZIPInputStream(fos);
        ObjectInput reader = new ObjectInputStream(gz);
        MLPNetwork mlpNetwork = (MLPNetwork) reader.readObject();
        KBeamArcEagerParser.parseNNConllFileNoParallel(mlpNetwork, options.inputFile, options.outputFile,
                options.beamWidth, 1, false, "");
    }

    public static void train(Options options) throws Exception {
        if (options.inputFile.equals("") || options.modelFile.equals("")) {
            Options.showHelp();
        } else {
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, options
                    .clusterFile, 1);
            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled, options
                    .rootFirst, options.lowercase, maps);
            System.out.println("CoNLL data reading done!");

            ArrayList<Integer> dependencyLabels = new ArrayList<Integer>();
            for (int lab : maps.getLabelMap().keySet())
                dependencyLabels.add(lab);

            int featureLength = options.useExtendedFeatures ? 72 : 26;
            if (options.useExtendedWithBrownClusterFeatures || maps.hasClusters())
                featureLength = 153;

            System.out.println("size of training data (#sens): " + dataSet.size());

            HashMap<String, Integer> labels = new HashMap<String, Integer>();
            int labIndex = 0;
            labels.put("sh", labIndex++);
            labels.put("rd", labIndex++);
            labels.put("us", labIndex++);
            for (int label : dependencyLabels) {
                if (options.labeled) {
                    labels.put("ra_" + label, 3 + label);
                    labels.put("la_" + label, 3 + dependencyLabels.size() + label);
                } else {
                    labels.put("ra_" + label, 3);
                    labels.put("la_" + label, 4);
                }
            }

            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early", new
                    AveragedPerceptron(featureLength, dependencyLabels.size()),
                    options, dependencyLabels, featureLength, maps);
            trainer.train(dataSet, options.devPath, options.trainingIter, options.modelFile, options.lowercase,
                    options.punctuations, options.partialTrainingStartingIteration);
        }
    }

    public static void trainWithNN(Options options) throws Exception {
        if (options.inputFile.equals("") || options.modelFile.equals("")) {
            Options.showHelp();
        } else {
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, options
                    .clusterFile, 1);
            int wDim = 64;
            if (options.wordEmbeddingFile.length() > 0)
                wDim = maps.readEmbeddings(options.wordEmbeddingFile);

            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled, options
                    .rootFirst, options.lowercase, maps);
            System.out.println("CoNLL data reading done!");

            ArrayList<Integer> dependencyLabels = new ArrayList<Integer>();
            for (int lab : maps.getLabelMap().keySet())
                dependencyLabels.add(lab);

            int featureLength = options.useExtendedFeatures ? 72 : 26;
            if (options.useExtendedWithBrownClusterFeatures || maps.hasClusters())
                featureLength = 153;

            System.out.println("size of training data (#sens): " + dataSet.size());

            HashMap<String, Integer> labels = new HashMap<String, Integer>();
            int labIndex = 0;
            labels.put("sh", labIndex++);
            labels.put("rd", labIndex++);
            labels.put("us", labIndex++);
            for (int label : dependencyLabels) {
                if (options.labeled) {
                    labels.put("ra_" + label, 3 + label);
                    labels.put("la_" + label, 3 + dependencyLabels.size() + label);
                } else {
                    labels.put("ra_" + label, 3);
                    labels.put("la_" + label, 4);
                }
            }

            System.out.println("Embedding dimension " + wDim);
            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early", new
                    AveragedPerceptron(featureLength, dependencyLabels.size()),
                    options, dependencyLabels, featureLength, maps);

            MLPNetwork mlpNetwork = new MLPNetwork(maps, options, dependencyLabels, wDim);
            MLPNetwork avgMlpNetwork = new MLPNetwork(maps, options, dependencyLabels, wDim);

            MLPClassifier classifier = new MLPClassifier(mlpNetwork, 0.9, options.learningRate, 0.0001);

            int decayStep = (int) (options.decayStep * dataSet.size() / options.batchSize);
            decayStep = decayStep == 0 ? 1 : decayStep;
            System.out.println("Decay after every "+decayStep +" batches");

            int step = 0;
            for (int i = 0; i < options.trainingIter; i++) {
                classifier.confusionMatrix = new int[2 * (dependencyLabels.size() + 1)][2 * (dependencyLabels.size() + 1)];

                System.out.println("reshuffling data for round " + i);
                Collections.shuffle(dataSet);
                int s = 0;
                int e = Math.min(dataSet.size(), options.batchSize);

                while (true) {
                    ArrayList<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, s, e, 0);
                    classifier.fit(instances, step, step % 10 == 0 ? true : false);

                    s = e;
                    e = Math.min(dataSet.size(), options.batchSize + e);

                    step++;
                    if (step % decayStep == 0) {
                        classifier.setLearningRate(0.96 * classifier.getLearningRate());
                        System.out.println("The new learning rate: " + classifier.getLearningRate());
                    }

                    // averaging
                    double ratio = Math.min(0.9999, (double) step / (9 + step));
                    MLPNetwork.averageNetworks(mlpNetwork, avgMlpNetwork, 1 - ratio, step == 1 ? 0 : ratio);

                    if (step % 100 == 0) {
                        KBeamArcEagerParser.parseNNConllFileNoParallel(mlpNetwork, options.devPath, options.modelFile
                                + ".tmp", options.beamWidth, 1, false, "");
                        Pair<Double, Double> eval = Evaluator.evaluate(options.devPath, options.modelFile + ".tmp", options.punctuations);

                        avgMlpNetwork.preCompute();
                        KBeamArcEagerParser.parseNNConllFileNoParallel(avgMlpNetwork, options.devPath, options.modelFile
                                + ".tmp", options.beamWidth, 1, false, "");
                        eval = Evaluator.evaluate(options.devPath, options.modelFile + ".tmp", options.punctuations);
                    }

                    if (s >= dataSet.size())
                        break;
                }

                StringBuilder output = new StringBuilder();
                for (int c = 0; c < classifier.confusionMatrix.length; c++) {
                    for (int j = 0; j < classifier.confusionMatrix.length; j++) {
                        output.append(classifier.confusionMatrix[c][j] + "\t");
                    }
                    output.append("\n");
                }
                System.out.print(output.toString());
            }
        }
    }
}
