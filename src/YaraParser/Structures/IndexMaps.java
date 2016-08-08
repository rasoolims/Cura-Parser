/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package YaraParser.Structures;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;

public class IndexMaps implements Serializable {
    public final String rootString;
    public final HashSet<Integer> rareWords;
    public final HashMap<Integer, Integer> preComputeMap;
    public String[] revStrings;
    private HashMap<String, Integer> stringMap;
    private HashMap<Integer, Integer> labelMap;
    // for neural net
    private HashMap<Integer, Integer> wordMap;
    private HashMap<Integer, Integer> posMap;
    private HashMap<Integer, Integer> depRelationMap;
    private HashMap<Integer, double[]> embeddingsDictionary;
    private HashMap<String, String> str2clusterMap;


    public IndexMaps(HashMap<String, Integer> stringMap, HashMap<Integer, Integer> labelMap, String rootString,
                     HashMap<Integer, Integer> wordMap, HashMap<Integer, Integer> posMap, HashMap<Integer, Integer> depRelationMap,
                     HashSet<Integer> rareWords, HashMap<Integer, Integer> preComputeMap,
                     HashMap<String, String> str2clusterMap) {
        this.stringMap = stringMap;
        this.wordMap = wordMap;
        this.posMap = posMap;
        this.depRelationMap = depRelationMap;
        this.labelMap = labelMap;
        this.str2clusterMap = str2clusterMap;

        revStrings = new String[stringMap.size() + 1];
        revStrings[0] = "ROOT";

        for (String word : stringMap.keySet()) {
            revStrings[stringMap.get(word)] = word;
        }
        this.rootString = rootString;
        embeddingsDictionary = new HashMap<>();
        this.rareWords = rareWords;
        this.preComputeMap = preComputeMap;
    }

    public HashMap<String, Integer> getStringMap() {
        return stringMap;
    }


    public HashMap<Integer, Integer> getLabelMap() {
        return labelMap;
    }

    public int getNeuralWordKey(int wordId) {
        int key = 0;
        if (wordMap.containsKey(wordId))
            key = wordMap.get(wordId);
        return key;
    }

    public int getNeuralPOSKey(int posId) {
        int key = 0;
        if (posMap.containsKey(posId))
            key = posMap.get(posId);
        return key;
    }

    public int getNeuralDepRelationKey(int labelId) {
        if (labelId == -1)
            return 1; // null
        return depRelationMap.get(labelId);
    }

    public int vocabSize() {
        return wordMap.size();
    }

    public int posSize() {
        return posMap.size();
    }

    public int relSize() {
        return depRelationMap.size();
    }

    public int readEmbeddings(String path) throws Exception {
        embeddingsDictionary = new HashMap<>();
        int eDim = 64;

        BufferedReader reader = new BufferedReader(new FileReader(path));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] spl = line.trim().split(" ");
            if (stringMap.containsKey(spl[0])) {
                double[] e = new double[spl.length - 1];
                eDim = e.length;
                int wordIndex = stringMap.get(spl[0]);
                for (int i = 0; i < e.length; i++) {
                    e[i] = Double.parseDouble(spl[i + 1]);
                }
                // adding 2 for unknown and null
                embeddingsDictionary.put(wordIndex + 2, e);
            }
        }
        return eDim;
    }


    public double[] embeddings(int wordIndex) {
        return embeddingsDictionary.get(wordIndex);
    }

    public int clusterIdForWord(String word) {
        int id = -1;
        if (str2clusterMap.containsKey(word)) {
            String c = str2clusterMap.get(word);
            if (stringMap.containsKey(c))
                return wordMap.get(stringMap.get(c));
        }
        return id;
    }
}
