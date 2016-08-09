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
    public static final int RootIndex = 2;
    public static final int UnknownIndex = 0;
    public static final int NullIndex = 1;
    public static final int LabelRootIndex = 0;
    public static int LabelUnknownIndex;
    public static int LabelNullIndex;
    public final String rootString;
    public final HashSet<Integer> rareWords;
    public final HashMap<Integer, Integer> preComputeMap;
    public String[] revWords;
    public String[] revLabels;
    public String[] revPos;
    // for neural net
    private HashMap<String, Integer> wordMap;
    private HashMap<String, Integer> posMap;
    private HashMap<String, Integer> depRelationMap;
    private HashMap<Integer, double[]> embeddingsDictionary;
    private HashMap<String, String> str2clusterMap;


    public IndexMaps(String rootString, HashMap<String, Integer> wordMap, HashMap<String, Integer> posMap, HashMap<String, Integer> depRelationMap,
                     HashSet<Integer> rareWords, HashMap<Integer, Integer> preComputeMap,
                     HashMap<String, String> str2clusterMap) {
        this.wordMap = wordMap;
        this.rootString = rootString;
        this.posMap = posMap;
        this.depRelationMap = depRelationMap;
        this.str2clusterMap = str2clusterMap;

        revWords = new String[wordMap.size() + 3];
        revWords[RootIndex] = rootString;
        for (String word : wordMap.keySet()) {
            revWords[wordMap.get(word)] = word;
        }

        revPos = new String[posMap.size() + 3];
        revPos[RootIndex] = rootString;
        for (String pos : posMap.keySet()) {
            revPos[posMap.get(pos)] = pos;
        }

        revLabels = new String[depRelationMap.size()];
        for (String label : depRelationMap.keySet()) {
            revLabels[depRelationMap.get(label)] = label;
        }
        LabelUnknownIndex = depRelationMap.size();
        LabelNullIndex = depRelationMap.size() + 1;

        embeddingsDictionary = new HashMap<>();
        this.rareWords = rareWords;
        this.preComputeMap = preComputeMap;
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
            if (wordMap.containsKey(spl[0])) {
                double[] e = new double[spl.length - 1];
                eDim = e.length;
                int wordIndex = wordMap.get(spl[0]);
                for (int i = 0; i < e.length; i++) {
                    e[i] = Double.parseDouble(spl[i + 1]);
                }
                embeddingsDictionary.put(wordIndex, e);
            }
        }
        return eDim;
    }


    public double[] embeddings(int wordIndex) {
        return embeddingsDictionary.get(wordIndex);
    }

    public int word2Int(String word) {
        if (wordMap.containsKey(word))
            return wordMap.get(word);
        if (str2clusterMap.containsKey(word))
            return wordMap.get(str2clusterMap.get(word));
        return UnknownIndex;
    }

    public int pos2Int(String pos) {
        if (posMap.containsKey(pos))
            return posMap.get(pos);
        return UnknownIndex;
    }

    public int dep2Int(String dep) {
        if (depRelationMap.containsKey(dep))
            return depRelationMap.get(dep);
        return UnknownIndex;
    }
}
