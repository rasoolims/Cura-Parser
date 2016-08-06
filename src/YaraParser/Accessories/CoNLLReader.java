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
        HashMap<String, Integer> stringMap = new HashMap<String, Integer>();
        HashMap<Integer, Integer> labelMap = new HashMap<Integer, Integer>();
        HashMap<String, Integer> clusterMap = new HashMap<>();
        HashMap<Integer, Integer> cluster4Map = new HashMap<>();
        HashMap<Integer, Integer> cluster6Map = new HashMap<>();
        HashMap<String, String> str2clusterMap = new HashMap<>();

        HashMap<Integer, Integer> wordMap = new HashMap<Integer, Integer>();
        HashMap<Integer, Integer> depRelationMap = new HashMap<Integer, Integer>();
        HashMap<Integer, Integer> posMap = new HashMap<Integer, Integer>();

        wordMap.put(0, 2);
        int wc = 3; // 0 for OOV, 1 for null, 2 for ROOT!


        String rootString = "ROOT";
        int wi = 1;
        int labelCount = 3;
        if (labeled) {
            wi = 1;
            stringMap.put(rootString, 0);
            labelMap.put(0, 0);
            depRelationMap.put(0, 2);
        } else {
            stringMap.put(rootString, 0);
            wi = 0;
            labelCount = 2;
        }

        HashMap<String, Integer> wordCount = new HashMap<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] spl = line.trim().split("\t");
            if (spl.length > 7) {
                String word = spl[1];
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
                if (!stringMap.containsKey(label)) {
                    labelMap.put(wi, labelCount);
                    depRelationMap.put(wi, labelCount);
                    labelCount++;
                    stringMap.put(label, wi++);
                }
            }
        }

        reader = new BufferedReader(new FileReader(filePath));
        while ((line = reader.readLine()) != null) {
            String[] spl = line.trim().split("\t");
            if (spl.length > 7) {
                String pos = spl[3];
                if (!stringMap.containsKey(pos)) {
                    stringMap.put(pos, wi++);
                }
            }
        }

        posMap.put(0, 2);
        int posCount = 3;// 0 for OOV, 1 for null, 2 for root!
        reader = new BufferedReader(new FileReader(filePath));
        while ((line = reader.readLine()) != null) {
            String[] spl = line.trim().split("\t");
            if (spl.length > 7) {
                String pos = spl[3];
                if (!posMap.containsKey(stringMap.get(pos))) {
                    posMap.put(stringMap.get(pos), posCount++);
                }
            }
        }


        if (clusterFile.length() > 0) {
            reader = new BufferedReader(new FileReader(clusterFile));
            while ((line = reader.readLine()) != null) {
                String[] spl = line.trim().split("\t");
                if (spl.length > 2) {
                    String cluster = spl[0];
                    String word = spl[1];
                    str2clusterMap.put(word, "cluster___" + cluster.substring(0, Math.min(8, cluster.length())));
                    String prefix4 = cluster.substring(0, Math.min(4, cluster.length()));
                    String prefix6 = cluster.substring(0, Math.min(6, cluster.length()));
                    int clusterNum = wi;

                    if (!stringMap.containsKey(cluster)) {
                        clusterMap.put(word, wi);
                    } else {
                        clusterNum = stringMap.get(cluster);
                        clusterMap.put(word, clusterNum);
                    }

                    int pref4Id = wi;
                    if (!stringMap.containsKey(prefix4)) {
                        stringMap.put(prefix4, wi++);
                    } else {
                        pref4Id = stringMap.get(prefix4);
                    }

                    int pref6Id = wi;
                    if (!stringMap.containsKey(prefix6)) {
                        stringMap.put(prefix6, wi++);
                    } else {
                        pref6Id = stringMap.get(prefix6);
                    }

                    cluster4Map.put(clusterNum, pref4Id);
                    cluster6Map.put(clusterNum, pref6Id);
                }
            }
        }

        HashSet<Integer> rareWords = new HashSet<>();

        // todo indexing may be false
        int addedCluster = 0;
        reader = new BufferedReader(new FileReader(filePath));
        while ((line = reader.readLine()) != null) {
            String[] spl = line.trim().split("\t");
            if (spl.length > 7) {
                String word = spl[1];
                if (lowercased)
                    word = word.toLowerCase();
                if (wordCount.get(word) > rareMaxWordCount && !stringMap.containsKey(word)) {
                    stringMap.put(word, wi++);
                } else if (wordCount.get(word) <= rareMaxWordCount && str2clusterMap.containsKey(word)) {
                    String c = str2clusterMap.get(word);

                    if (!wordCount.containsKey(c)) {
                        wordCount.put(c, 1);
                        addedCluster++;
                    } else
                        wordCount.put(c, wordCount.get(c) + 1);

                    if (!stringMap.containsKey(c)) {
                        stringMap.put(c, wi++);
                    }
                }
            }
        }

        reader = new BufferedReader(new FileReader(filePath));
        while ((line = reader.readLine()) != null) {
            String[] spl = line.trim().split("\t");
            if (spl.length > 7) {
                String word = spl[1];
                if (lowercased)
                    word = word.toLowerCase();
                if (wordCount.get(word) > rareMaxWordCount && !wordMap.containsKey(stringMap.get(word))) {
                    if (wordCount.get(word) == 1)
                        rareWords.add(wc);
                    wordMap.put(stringMap.get(word), wc++);
                } else if (wordCount.get(word) <= rareMaxWordCount && str2clusterMap.containsKey(word)) {
                    String c = str2clusterMap.get(word);
                    int key = stringMap.get(c);
                    if (!wordMap.containsKey(key)) {
                        wordMap.put(key, wc++);
                    }
                }
            }
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
        System.out.println("#rare_types: " + rare + " out of " + (wordCount.size() - addedCluster));
        System.out.println("#word2cluster: " + word2cluster + " out of " + rare);
        System.out.println("#word2cluster (distinct): " + addedCluster);

        TreeMap<Integer, HashSet<Integer>> sortedCounts = new TreeMap<>();
        for (String word : wordCount.keySet()) {
            if (stringMap.containsKey(word) && wordMap.containsKey(stringMap.get(word))) {
                int id = wordMap.get(stringMap.get(word));
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

        return new IndexMaps(stringMap, labelMap, rootString, wordMap, posMap, depRelationMap, cluster4Map,
                cluster6Map, clusterMap, rareWords, preComputeMap, str2clusterMap);
    }


    /**
     * @param limit it is used if we want to read part of the data
     * @return
     */
    public ArrayList<GoldConfiguration> readData(int limit, boolean keepNonProjective, boolean labeled,
                                                 boolean rootFirst, boolean lowerCased, IndexMaps maps) throws
            Exception {
        HashMap<String, Integer> wordMap = maps.getStringMap();
        ArrayList<GoldConfiguration> configurationSet = new ArrayList<GoldConfiguration>();
        HashSet<String> oovTypes = new HashSet<>();

        String line;
        ArrayList<Integer> tokens = new ArrayList<Integer>();
        ArrayList<Integer> tags = new ArrayList<Integer>();
        ArrayList<Integer> cluster4Ids = new ArrayList<Integer>();
        ArrayList<Integer> cluster6Ids = new ArrayList<Integer>();
        ArrayList<Integer> clusterIds = new ArrayList<Integer>();

        HashMap<Integer, Pair<Integer, Integer>> goldDependencies = new HashMap<Integer, Pair<Integer, Integer>>();
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
                        tokens.add(0);
                        tags.add(0);
                        cluster4Ids.add(0);
                        cluster6Ids.add(0);
                        clusterIds.add(0);
                    }
                    Sentence currentSentence = new Sentence(tokens, tags, cluster4Ids, cluster6Ids, clusterIds);
                    GoldConfiguration goldConfiguration = new GoldConfiguration(currentSentence, goldDependencies);
                    if (keepNonProjective || !goldConfiguration.isNonprojective())
                        configurationSet.add(goldConfiguration);
                    goldDependencies = new HashMap<Integer, Pair<Integer, Integer>>();
                    tokens = new ArrayList<Integer>();
                    tags = new ArrayList<Integer>();
                    cluster4Ids = new ArrayList<Integer>();
                    cluster6Ids = new ArrayList<Integer>();
                    clusterIds = new ArrayList<Integer>();
                } else {
                    goldDependencies = new HashMap<Integer, Pair<Integer, Integer>>();
                    tokens = new ArrayList<Integer>();
                    tags = new ArrayList<Integer>();
                    cluster4Ids = new ArrayList<Integer>();
                    cluster6Ids = new ArrayList<Integer>();
                    clusterIds = new ArrayList<Integer>();
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
                String pos = splitLine[3].trim();

                int wi = -1;
                if (wordMap.containsKey(word)) {
                    wi = wordMap.get(word);
                } else if (maps.hasClusters()) {
                    wi = maps.clusterIdForWord(word);
                    if (wi == -1)
                        oovTypes.add(word);
                } else {
                    oovTypes.add(word);
                }

                int pi = -1;
                if (wordMap.containsKey(pos))
                    pi = wordMap.get(pos);

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

                int ri = 0;
                if (wordMap.containsKey(relation))
                    ri = wordMap.get(relation);
                if (headIndex == -1)
                    ri = -1;

                int[] ids = maps.clusterId(word);
                clusterIds.add(ids[0]);
                cluster4Ids.add(ids[1]);
                cluster6Ids.add(ids[2]);

                if (headIndex >= 0)
                    goldDependencies.put(wordIndex, new Pair<Integer, Integer>(headIndex, ri));
            }
        }
        if (tokens.size() > 0) {
            if (!rootFirst) {
                for (int gold : goldDependencies.keySet()) {
                    if (goldDependencies.get(gold).first.equals(0))
                        goldDependencies.get(gold).setFirst(goldDependencies.size() + 1);
                }
                tokens.add(0);
                tags.add(0);
                cluster4Ids.add(0);
                cluster6Ids.add(0);
                clusterIds.add(0);
            }
            sentenceCounter++;
            Sentence currentSentence = new Sentence(tokens, tags, cluster4Ids, cluster6Ids, clusterIds);
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
                tags = new ArrayList<String>();
                goldDependencies = new HashMap<Integer, Pair<Integer, String>>();
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
