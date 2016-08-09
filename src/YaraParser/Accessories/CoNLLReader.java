/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package YaraParser.Accessories;

import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.Sentence;
import YaraParser.TransitionBasedSystem.Configuration.CompactTree;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;

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
                                          int rareMaxWordCount) throws Exception {
        HashMap<String, String> str2clusterMap = new HashMap<>();

        HashMap<String, Integer> wordMap = new HashMap<>();
        HashMap<String, Integer> depRelationMap = new HashMap<>();
        HashMap<String, Integer> posMap = new HashMap<>();

        int wc = 3; // 0 for OOV, 1 for null, 2 for ROOT!

        String rootString = "ROOT";

        HashMap<String, Integer> wordCount = new HashMap<>();
        HashSet<String> labels = new HashSet<>();
        HashSet<String> tags = new HashSet<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] spl = line.trim().split("\t");
            if (spl.length > 7) {
                String word = spl[1];
                String pos = spl[3];
                if (lowercased)
                    word = word.toLowerCase();
                if (wordCount.containsKey(word))
                    wordCount.put(word, wordCount.get(word) + 1);
                else
                    wordCount.put(word, 1);
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

        depRelationMap.put(rootString, IndexMaps.LabelRootIndex);
        int l = IndexMaps.LabelRootIndex + 1;
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


        if (clusterFile.length() > 0) {
            reader = new BufferedReader(new FileReader(clusterFile));
            while ((line = reader.readLine()) != null) {
                String[] spl = line.trim().split("\t");
                if (spl.length > 2) {
                    String cluster = spl[0];
                    String word = spl[1];
                    str2clusterMap.put(word, "cluster___" + cluster.substring(0, Math.min(8, cluster.length())));
                }
            }
        }


        HashSet<Integer> rareWords = new HashSet<>();

        // todo indexing may be false
        int wi = IndexMaps.RootIndex;
        wordMap.put(rootString, wi++);
        HashSet<String> addedClusters = new HashSet<>();
        reader = new BufferedReader(new FileReader(filePath));
        while ((line = reader.readLine()) != null) {
            String[] spl = line.trim().split("\t");
            if (spl.length > 7) {
                String word = spl[1];
                if (lowercased)
                    word = word.toLowerCase();
                if (wordCount.get(word) > rareMaxWordCount && !wordMap.containsKey(word)) {
                    wordMap.put(word, wi++);
                } else if (wordCount.get(word) <= rareMaxWordCount && str2clusterMap.containsKey(word)) {
                    String c = str2clusterMap.get(word);

                    if (!wordMap.containsKey(c)) {
                        wordMap.put(c, wi++);
                        addedClusters.add(c);
                    }
                }
            }
        }

        Object[] wordsSet = str2clusterMap.keySet().toArray().clone();
        for (Object word : wordsSet) {
            if (!addedClusters.contains(str2clusterMap.get(word)))
                str2clusterMap.remove(word);
        }

        int word2cluster = 0;
        int rare = 0;
        for (String word : wordCount.keySet()) {
            if (wordCount.get(word) <= rareMaxWordCount) {
                rare++;
                if (str2clusterMap.containsKey(word))
                    word2cluster++;
            }
        }
        System.out.println("#rare_types: " + rare + " out of " + (wordCount.size()));
        System.out.println("#word2cluster: " + word2cluster + " out of " + rare);
        System.out.println("#word2cluster (distinct): " + addedClusters.size());

        TreeMap<Integer, HashSet<Integer>> sortedCounts = new TreeMap<>();
        for (String word : wordCount.keySet()) {
            if (wordMap.containsKey(word) && wordMap.containsKey(wordMap.get(word))) {
                int id = wordMap.get(wordMap.get(word));
                int count = wordCount.get(word);
                if (!sortedCounts.containsKey(count))
                    sortedCounts.put(count, new HashSet<Integer>());
                sortedCounts.get(count).add(id);
            }
        }

        HashMap<Integer, Integer> preComputeMap = new HashMap<>();
        int wCount = 0;
        preComputeMap.put(0, wCount++);
        preComputeMap.put(1, wCount++);
        preComputeMap.put(2, wCount++);

        for (int count : sortedCounts.descendingKeySet()) {
            HashSet<Integer> ids = sortedCounts.get(count);
            for (int id : ids) {
                if (id > 2)
                    preComputeMap.put(id, wCount++);
                if (wCount >= 1000)
                    break;
            }
        }

        return new IndexMaps(rootString, wordMap, posMap, depRelationMap, rareWords, preComputeMap, str2clusterMap);
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
                int wi = maps.word2Int(word);
                if (wi == -1)
                    oovTypes.add(word);

                String pos = splitLine[3].trim();
                int pi = maps.pos2Int(pos);

                tags.add(pi);
                tokens.add(wi);

                int headIndex = Integer.parseInt(splitLine[6]);
                String relation = splitLine[7];
                if (relation.equals("_"))
                    relation = "-";


                if (headIndex == 0)
                    relation = "ROOT";
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
                        goldDependencies.get(gold).setFirst(goldDependencies.size() + 1);
                }
                tokens.add(IndexMaps.RootIndex);
                tags.add(IndexMaps.RootIndex);
            }
            sentenceCounter++;
            Sentence currentSentence = new Sentence(tokens, tags);
            configurationSet.add(new GoldConfiguration(currentSentence, goldDependencies));
        }

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

                if (headIndex == 0) {
                    relation = "ROOT";
                }

                if (pos.length() > 0)
                    goldDependencies.put(wordIndex, new Pair<Integer, String>(headIndex, relation));
            }
        }


        if (tags.size() > 0) {
            treeSet.add(new CompactTree(goldDependencies, tags));
        }

        return treeSet;
    }

}
