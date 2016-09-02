/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.CuraParser;

import edu.columbia.cs.nlp.CuraParser.Accessories.Evaluator;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Beam.BeamParser;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Trainer.BeamTrainer;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Trainer.GreedyTrainer;

import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.zip.GZIPInputStream;

public class CuraParser {
    public static void main(String[] args) throws Exception {
        Options options = Options.processArgs(args);

        if (options.generalProperties.showHelp) {
            Options.showHelp();
        } else {
            System.out.println(options);
            if (options.generalProperties.train) {
                if (options.trainingOptions.beamTrainingIter == 0)
                    GreedyTrainer.trainWithNN(options);
                else
                    BeamTrainer.trainWithNN(options);
            } else if (options.generalProperties.parseTaggedFile || options.generalProperties.parseConllFile
                    || options.generalProperties.parsePartialConll) {
                parse(options);
            } else if (options.generalProperties.evaluate) {
                evaluate(options);
            } else if (options.generalProperties.output) {
                GreedyTrainer.outputTrainInstances(options);
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
