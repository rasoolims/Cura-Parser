/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.CuraParser;

import edu.columbia.cs.nlp.CuraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.CuraParser.Accessories.Evaluator;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Learning.Activation.Enums.ActivationType;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPTrainer;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.AveragingOption;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.UpdaterType;
import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.Structures.NeuralTrainingInstance;
import edu.columbia.cs.nlp.CuraParser.Structures.Pair;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Beam.BeamParser;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.ParserType;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Trainer.GreedyTrainer;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class CuraParser {
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
            options.networkProperties.hiddenLayer2Size = 0;
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
                GreedyTrainer.trainWithNN(options);
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


}
