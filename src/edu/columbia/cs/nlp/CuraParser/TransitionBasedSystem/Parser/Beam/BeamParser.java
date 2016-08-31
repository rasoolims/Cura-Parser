/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Beam;

import edu.columbia.cs.nlp.CuraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.Structures.Pair;
import edu.columbia.cs.nlp.CuraParser.Structures.Sentence;
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
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Threading.BeamScorerThread;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Threading.ParseTaggedThread;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Threading.ParseThread;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Threading.PartialTreeBeamScorerThread;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.TreeSet;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class BeamParser {
    final MLPNetwork network;
    ArrayList<Integer> dependencyRelations;
    IndexMaps maps;
    int numThreads;
    ExecutorService executor;
    CompletionService<ArrayList<BeamElement>> pool;
    ShiftReduceParser parser;

    public BeamParser(MLPNetwork network, int numOfThreads, ParserType parserType) {
        this.dependencyRelations = network.getDepLabels();
        this.network = network;
        this.numThreads = numOfThreads;
        this.maps = network.maps;
        executor = Executors.newFixedThreadPool(numOfThreads);
        pool = new ExecutorCompletionService<>(executor);

        if (parserType == ParserType.ArcEager)
            parser = new ArcEager();
        else if (parserType == ParserType.ArcStandard)
            parser = new ArcStandard();
        else
            throw new NotImplementedException();
    }

    private void parseWithOneThread(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver, int beamWidth) throws Exception {
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
                for (int i = 0; i < maps.relSize(); i++)
                    labels[2 + i] = -1;
            if (!canLeftArc)
                for (int i = 0; i < maps.relSize(); i++)
                    labels[maps.relSize() + 2 + i] = -1;
            double[] features = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
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
                    double score = scores[2 + maps.relSize() + dependency];
                    double addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));

                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
            }
        }
    }

    public Configuration parse(Sentence sentence, boolean rootFirst, int beamWidth, int numOfThreads) throws Exception {
        Configuration initialConfiguration = new Configuration(sentence, rootFirst);

        ArrayList<Configuration> beam = new ArrayList<>(beamWidth);
        beam.add(initialConfiguration);

        while (!parser.isTerminal(beam)) {
            TreeSet<BeamElement> beamPreserver = new TreeSet<>();

            if (numOfThreads == 1) {
                parseWithOneThread(beam, beamPreserver, beamWidth);
            } else {
                for (int b = 0; b < beam.size(); b++) {
                    pool.submit(new BeamScorerThread(true, network, beam.get(b),
                            dependencyRelations, b, rootFirst, maps.labelNullIndex, parser));
                }
                for (int b = 0; b < beam.size(); b++) {
                    for (BeamElement element : pool.take().get()) {
                        beamPreserver.add(element);
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

    private Configuration parsePartial(GoldConfiguration goldConfiguration, boolean rootFirst, int beamWidth) throws Exception {
        Configuration initialConfiguration = new Configuration(goldConfiguration.getSentence(), rootFirst);

        ArrayList<Configuration> beam = new ArrayList<>(beamWidth);
        beam.add(initialConfiguration);

        while (!parser.isTerminal(beam)) {
            TreeSet<BeamElement> beamPreserver = new TreeSet<>();

            for (int b = 0; b < beam.size(); b++) {
                pool.submit(new PartialTreeBeamScorerThread(true, network, goldConfiguration, beam.get(b),
                        dependencyRelations, b, maps.labelNullIndex, parser));
            }
            for (int b = 0; b < beam.size(); b++) {
                for (BeamElement element : pool.take().get()) {
                    beamPreserver.add(element);
                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
            }

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

    public void parseTaggedFile(String inputFile, String outputFile, boolean rootFirst, int beamWidth, boolean
            lowerCased, String separator, int numOfThreads) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
        long start = System.currentTimeMillis();

        ExecutorService executor = Executors.newFixedThreadPool(numOfThreads);
        CompletionService<Pair<String, Integer>> pool = new ExecutorCompletionService<>(executor);


        String line;
        int count = 0;
        int lineNum = 0;
        while ((line = reader.readLine()) != null) {
            pool.submit(new ParseTaggedThread(lineNum++, line, separator, rootFirst, lowerCased, maps, beamWidth, this));

            if (lineNum % 1000 == 0) {
                String[] outs = new String[lineNum];
                for (int i = 0; i < lineNum; i++) {
                    count++;
                    if (count % 100 == 0)
                        System.err.print(count + "...");
                    Pair<String, Integer> result = pool.take().get();
                    outs[result.second] = result.first;
                }

                for (int i = 0; i < lineNum; i++) {
                    if (outs[i].length() > 0) {
                        writer.write(outs[i]);
                    }
                }

                lineNum = 0;
            }
        }

        if (lineNum > 0) {
            String[] outs = new String[lineNum];
            for (int i = 0; i < lineNum; i++) {
                count++;
                if (count % 100 == 0)
                    System.err.print(count + "...");
                Pair<String, Integer> result = pool.take().get();
                outs[result.second] = result.first;
            }

            for (int i = 0; i < lineNum; i++) {

                if (outs[i].length() > 0) {
                    writer.write(outs[i]);
                }
            }
        }

        long end = System.currentTimeMillis();
        System.out.println("\n" + (end - start) + " ms");
        writer.flush();
        writer.close();

        System.out.println("done!");
    }

    public void parseConll(String inputFile, String outputFile, boolean rootFirst, int beamWidth, boolean
            lowerCased, int numThreads, boolean partial, String scorePath) throws Exception {
        CoNLLReader reader = new CoNLLReader(inputFile);

        boolean addScore = false;
        if (scorePath.trim().length() > 0)
            addScore = true;
        ArrayList<Double> scoreList = new ArrayList<>();

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<Pair<Configuration, Integer>> pool = new ExecutorCompletionService<>(executor);

        long start = System.currentTimeMillis();
        int allArcs = 0;
        int size = 0;
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile + ".tmp"));
        int dataCount = 0;

        while (true) {
            ArrayList<GoldConfiguration> data = reader.readData(15000, true, true, rootFirst, lowerCased, maps);
            size += data.size();
            if (data.size() == 0) break;

            int index = 0;
            Configuration[] confs = new Configuration[data.size()];

            if (!partial) {
                for (GoldConfiguration goldConfiguration : data) {
                    ParseThread thread = new ParseThread(index, network, goldConfiguration.getSentence(), rootFirst, beamWidth, goldConfiguration,
                            partial, maps.labelNullIndex, parser);
                    pool.submit(thread);
                    index++;
                }

                for (int i = 0; i < confs.length; i++) {
                    dataCount++;
                    if (dataCount % 100 == 0)
                        System.err.print(dataCount + " ... ");

                    Pair<Configuration, Integer> configurationIntegerPair = pool.take().get();
                    confs[configurationIntegerPair.second] = configurationIntegerPair.first;
                }
            } else {
                for (int i = 0; i < confs.length; i++) {
                    GoldConfiguration goldConfiguration = data.get(i);
                    confs[i] = parsePartial(goldConfiguration, rootFirst, beamWidth);
                }
            }

            for (int j = 0; j < confs.length; j++) {
                Configuration bestParse = confs[j];
                if (addScore) {
                    scoreList.add(bestParse.score / bestParse.sentence.size());
                }
                int[] words = data.get(j).getSentence().getWords();

                allArcs += words.length - 1;

                StringBuilder finalOutput = new StringBuilder();
                for (int i = 0; i < words.length; i++) {
                    int w = i + 1;
                    int head = bestParse.state.getHead(w);
                    int dep = bestParse.state.getDependency(w, maps.labelNullIndex);

                    if (w == bestParse.state.rootIndex && !rootFirst)
                        continue;

                    if (head == bestParse.state.rootIndex)
                        head = 0;

                    String label = head == 0 ? maps.rootString : maps.revLabels[dep];
                    String output = head + "\t" + label + "\n";
                    finalOutput.append(output);
                }
                finalOutput.append("\n");
                writer.write(finalOutput.toString());
            }
        }

        System.err.print("\n");
        long end = System.currentTimeMillis();
        double each = (1.0 * (end - start)) / size;
        double eacharc = (1.0 * (end - start)) / allArcs;

        writer.flush();
        writer.close();

        DecimalFormat format = new DecimalFormat("##.00");

        System.out.print(format.format(eacharc) + " ms for each arc!\n");
        System.out.print(format.format(each) + " ms for each sentence!\n\n");

        BufferedReader gReader = new BufferedReader(new FileReader(inputFile));
        BufferedReader pReader = new BufferedReader(new FileReader(outputFile + ".tmp"));
        BufferedWriter pwriter = new BufferedWriter(new FileWriter(outputFile));

        String line;

        while ((line = pReader.readLine()) != null) {
            String gLine = gReader.readLine();
            if (line.trim().length() > 0) {
                while (gLine.trim().length() == 0)
                    gLine = gReader.readLine();
                String[] ps = line.split("\t");
                String[] gs = gLine.split("\t");
                gs[6] = ps[0];
                gs[7] = ps[1];
                StringBuilder output = new StringBuilder();
                for (String g : gs) {
                    output.append(g).append("\t");
                }
                pwriter.write(output.toString().trim() + "\n");
            } else {
                pwriter.write("\n");
            }
        }
        pwriter.flush();
        pwriter.close();

        if (addScore) {
            BufferedWriter scoreWriter = new BufferedWriter(new FileWriter(scorePath));

            for (int i = 0; i < scoreList.size(); i++)
                scoreWriter.write(scoreList.get(i) + "\n");
            scoreWriter.flush();
            scoreWriter.close();
        }
    }

    public void shutDownLiveThreads() {
        boolean isTerminated = executor.isTerminated();
        while (!isTerminated) {
            executor.shutdownNow();
            isTerminated = executor.isTerminated();
        }
    }
}