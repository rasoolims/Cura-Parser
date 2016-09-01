/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.CuraParser.Accessories;

import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.Structures.Pair;
import edu.columbia.cs.nlp.CuraParser.Structures.Sentence;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.CompactTree;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class CoNLLReader {
    /**
     * An object for reading the CoNLL file
     */
    BufferedReader fileReader;

    /**
     * Initializes the file reader
     *
     * @param filePath Path to the file
     * @throws Exception If the file path is not correct or there are not enough permission to read the file
     */
    public CoNLLReader(String filePath) throws Exception {
        fileReader = new BufferedReader(new FileReader(filePath));
    }

    public static IndexMaps createIndices(String filePath, boolean labeled, boolean lowercased, String clusterFile,
                                          int rareMaxWordCount, boolean includePOSAsWordForUnknown) throws Exception {
        HashMap<String, String> str2clusterMap = new HashMap<>();
        String line;
        if (clusterFile.length() > 0) {
            BufferedReader reader = new BufferedReader(new FileReader(clusterFile));
            while ((line = reader.readLine()) != null) {
                String[] spl = line.trim().split(" ");
                if (spl.length == 2) {
                    str2clusterMap.put(spl[0], spl[1]);
                }
            }
        }

        HashMap<String, Integer> wordMap = new HashMap<>();
        HashMap<String, Integer> depRelationMap = new HashMap<>();
        HashMap<String, Integer> posMap = new HashMap<>();
        String rootString = "ROOT";
        HashMap<String, Integer> wordCount = new HashMap<>();
        HashSet<String> labels = new HashSet<>();
        HashSet<String> tags = new HashSet<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        while ((line = reader.readLine()) != null) {
            String[] spl = line.trim().split("\t");
            if (spl.length > 7) {
                String word = spl[1];
                String pos = spl[3];
                if (lowercased)
                    word = word.toLowerCase();
                if (Utils.isNumeric(word))
                    word = "_NUM_";

                if (str2clusterMap.size() > 0) {
                    if (str2clusterMap.containsKey(word))
                        word = str2clusterMap.get(word);
                    else
                        word = "_unk_";
                }

                if (includePOSAsWordForUnknown) {
                    if (!wordCount.containsKey("pos:" + pos))
                        wordCount.put("pos:" + pos, 1);
                    else
                        wordCount.put("pos:" + pos, wordCount.get("pos:" + pos) + 1);

                }
                if (!word.equals("_unk_")) {
                    if (wordCount.containsKey(word))
                        wordCount.put(word, wordCount.get(word) + 1);
                    else
                        wordCount.put(word, 1);
                }
                String label = spl[7];
                int head = Integer.parseInt(spl[6]);
                if (head == 0)
                    rootString = label;

                if (label.equals("_"))
                    label = "-";
                if (!labeled)
                    label = "~";
                labels.add(label);
                tags.add(pos);
            }
        }

        // We shift everything by two when getting the actual value for the dependency feature.
        depRelationMap.put(rootString, 0);
        int l = 1;
        for (String lab : labels) {
            if (!lab.equals(rootString)) {
                depRelationMap.put(lab, l++);
            }
        }

        posMap.put(rootString, IndexMaps.RootIndex);
        int p = IndexMaps.RootIndex + 1;
        for (String pos : tags)
            if (!pos.equals(rootString))
                posMap.put(pos, p++);

        HashSet<Integer> rareWords = new HashSet<>();

        int wi = IndexMaps.RootIndex;
        wordMap.put(rootString, wi++);
        reader = new BufferedReader(new FileReader(filePath));
        while ((line = reader.readLine()) != null) {
            String[] spl = line.trim().split("\t");
            if (spl.length > 7) {
                String word = spl[1];
                String pos = spl[3];
                if (lowercased)
                    word = word.toLowerCase();
                if (Utils.isNumeric(word))
                    word = "_NUM_";
                if (str2clusterMap.size() > 0) {
                    if (str2clusterMap.containsKey(word))
                        word = str2clusterMap.get(word);
                    else if (includePOSAsWordForUnknown) {
                        word = "pos:" + pos;
                    } else
                        word = "_unk_";
                }

                if (wordCount.containsKey(word) && wordCount.get(word) > rareMaxWordCount && !wordMap.containsKey(word))
                    wordMap.put(word, wi++);
            }
        }

        if (includePOSAsWordForUnknown) {
            // Making sure that all tags are included in the word map.
            for (String pos : tags)
                if (!wordMap.containsKey("pos:" + pos))
                    wordMap.put("pos:" + pos, wi++);

        }

        Object[] wordsSet = str2clusterMap.keySet().toArray().clone();
        for (Object word : wordsSet) {
            if (!wordMap.containsKey(str2clusterMap.get(word)))
                str2clusterMap.remove(word);
        }

        int rare = 0;
        for (String word : wordCount.keySet()) {
            if (wordCount.get(word) <= rareMaxWordCount) {
                rare++;
            } else if (wordCount.get(word) <= 1) {
                rareWords.add(wordMap.get(word));
            }
        }
        System.out.println("#rare_types: " + rare + " out of " + (wordCount.size()));
        return new IndexMaps(rootString, wordMap, posMap, depRelationMap, rareWords, str2clusterMap);
    }

    /**
     * @param limit it is used if we want to read part of the data
     * @return
     */
    public ArrayList<GoldConfiguration> readData(int limit, boolean keepNonProjective, boolean labeled,
                                                 boolean rootFirst, boolean lowerCased, IndexMaps maps) throws Exception {
        ArrayList<GoldConfiguration> configurationSet = new ArrayList<>();
        HashSet<String> oovTypes = new HashSet<>();

        String line;
        ArrayList<Integer> tokens = new ArrayList<>();
        ArrayList<Integer> tags = new ArrayList<>();

        HashMap<Integer, Pair<Integer, Integer>> goldDependencies = new HashMap<>();
        int sentenceCounter = 0;
        while ((line = fileReader.readLine()) != null) {
            line = line.trim();
            if (line.length() == 0) {
                if (tokens.size() >= 1) {
                    sentenceCounter++;
                    if (!rootFirst) {
                        for (int gold : goldDependencies.keySet()) {
                            if (goldDependencies.get(gold).first.equals(0))
                                goldDependencies.get(gold).setFirst(tokens.size() + 1);
                        }
                        tokens.add(IndexMaps.RootIndex);
                        tags.add(IndexMaps.RootIndex);
                    }
                    Sentence currentSentence = new Sentence(tokens, tags);
                    GoldConfiguration goldConfiguration = new GoldConfiguration(currentSentence, goldDependencies);
                    if (keepNonProjective || !goldConfiguration.isNonprojective())
                        configurationSet.add(goldConfiguration);
                    goldDependencies = new HashMap<>();
                    tokens = new ArrayList<>();
                    tags = new ArrayList<>();
                } else {
                    goldDependencies = new HashMap<>();
                    tokens = new ArrayList<>();
                    tags = new ArrayList<>();
                }
                if (sentenceCounter >= limit) {
                    System.out.println("buffer full..." + configurationSet.size());
                    break;
                }
            } else {
                String[] splitLine = line.split("\t");
                if (splitLine.length < 8)
                    throw new Exception("wrong file format");
                int wordIndex = Integer.parseInt(splitLine[0]);
                String word = splitLine[1].trim();
                if (lowerCased)
                    word = word.toLowerCase();
                if (Utils.isNumeric(word))
                    word = "_NUM_";
                String pos = splitLine[3].trim();
                int wi = maps.word2Int(word, pos);
                if (wi == IndexMaps.UnknownIndex)
                    oovTypes.add(word);
                int pi = maps.pos2Int(pos);
                tags.add(pi);
                tokens.add(wi);

                int headIndex = Integer.parseInt(splitLine[6]);
                String relation = splitLine[7];
                if (relation.equals("_"))
                    relation = "-";

                if (!labeled)
                    relation = "~";

                int ri = -1;
                if (headIndex != -1)
                    ri = maps.dep2Int(relation);

                if (headIndex >= 0)
                    goldDependencies.put(wordIndex, new Pair<>(headIndex, ri));
            }
        }
        if (tokens.size() > 0) {
            if (!rootFirst) {
                for (int gold : goldDependencies.keySet()) {
                    if (goldDependencies.get(gold).first.equals(0))
                        goldDependencies.get(gold).setFirst(tokens.size() + 1);
                }
                tokens.add(IndexMaps.RootIndex);
                tags.add(IndexMaps.RootIndex);
            }
            sentenceCounter++;
            Sentence currentSentence = new Sentence(tokens, tags);
            configurationSet.add(new GoldConfiguration(currentSentence, goldDependencies));
        }

        if (configurationSet.size() > 0)
            System.out.println("oov  " + oovTypes.size());
        return configurationSet;
    }

    public ArrayList<CompactTree> readStringData() throws Exception {
        ArrayList<CompactTree> treeSet = new ArrayList<CompactTree>();

        String line;
        ArrayList<String> tags = new ArrayList<String>();

        HashMap<Integer, Pair<Integer, String>> goldDependencies = new HashMap<Integer, Pair<Integer, String>>();
        while ((line = fileReader.readLine()) != null) {
            line = line.trim();
            if (line.length() == 0) {
                if (tags.size() >= 1) {
                    CompactTree goldConfiguration = new CompactTree(goldDependencies, tags);
                    treeSet.add(goldConfiguration);
                }
                tags = new ArrayList<>();
                goldDependencies = new HashMap<>();
            } else {
                String[] splitLine = line.split("\t");
                if (splitLine.length < 8)
                    throw new Exception("wrong file format");
                int wordIndex = Integer.parseInt(splitLine[0]);
                String pos = splitLine[3].trim();
                tags.add(pos);

                int headIndex = Integer.parseInt(splitLine[6]);
                String relation = splitLine[7];
                if (pos.length() > 0)
                    goldDependencies.put(wordIndex, new Pair<>(headIndex, relation));
            }
        }

        if (tags.size() > 0) {
            treeSet.add(new CompactTree(goldDependencies, tags));
        }

        return treeSet;
    }
}
