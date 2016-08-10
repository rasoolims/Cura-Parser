package YaraParser.Structures;

import YaraParser.Accessories.Pair;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.TreeMap;

public class IndexMaps implements Serializable {
    public static final int RootIndex = 2;
    public static final int UnknownIndex = 0;
    public static final int NullIndex = 1;
    public static final int LabelRootIndex = 0;
    public static int LabelUnknownIndex;
    public static int LabelNullIndex;
    public final String rootString;
    public final HashSet<Integer> rareWords;
    public HashMap<Integer, Integer>[] preComputeMap;
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
                     HashSet<Integer> rareWords, HashMap<String, String> str2clusterMap) {
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
            int wordIndex = -1;
            if (wordMap.containsKey(spl[0]))
                wordIndex = wordMap.get(spl[0]);
            else if (wordMap.containsKey(spl[0].toLowerCase()))
                wordIndex = wordMap.get(spl[0].toLowerCase());

            if (wordIndex != -1) {
                double[] e = new double[spl.length - 1];
                eDim = e.length;
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

    public void emptyEmbeddings() {
        embeddingsDictionary = null;
    }

    public int word2Int(String word) {
        if (wordMap.containsKey(word))
            return wordMap.get(word);
        if (str2clusterMap.containsKey(word))
            return wordMap.get(str2clusterMap.get(word));
        if (str2clusterMap.containsKey(word.toLowerCase()))
            return wordMap.get(str2clusterMap.get(word.toLowerCase()));
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
        return LabelUnknownIndex;
    }

    public void constructPreComputeMap(List<NeuralTrainingInstance> instances, int numWordLayer, int maxNumber) {
        HashMap<Integer, Integer>[] counts = new HashMap[numWordLayer];
        preComputeMap = new HashMap[numWordLayer];

        for (int i = 0; i < counts.length; i++) {
            counts[i] = new HashMap<>();
            preComputeMap[i] = new HashMap<>();
        }

        for (NeuralTrainingInstance instance : instances) {
            int[] feats = instance.getFeatures();
            for (int i = 0; i < numWordLayer; i++) {
                int f = feats[i];
                if (counts[i].containsKey(f))
                    counts[i].put(f, counts[i].get(f) + 1);
                else
                    counts[i].put(f, 1);
            }
        }

        TreeMap<Integer, HashSet<Pair<Integer, Integer>>> sortedCounts = new TreeMap<>();
        for (int i = 0; i < counts.length; i++) {
            for (int f : counts[i].keySet()) {
                int count = counts[i].get(f);
                if (!sortedCounts.containsKey(count))
                    sortedCounts.put(count, new HashSet<Pair<Integer, Integer>>());
                sortedCounts.get(count).add(new Pair<>(i, f));
            }
        }

        int c = 0;
        int[] slotcounter = new int[preComputeMap.length];

        for (int count : sortedCounts.descendingKeySet()) {
            for (Pair<Integer, Integer> p : sortedCounts.get(count)) {
                c++;
                preComputeMap[p.first].put(p.second, slotcounter[p.first]);
                slotcounter[p.first] += 1;
            }
            if (c >= maxNumber)
                break;
        }

    }
}