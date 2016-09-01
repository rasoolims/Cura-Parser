package edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers;

import java.util.HashMap;
import java.util.Random;
import java.util.Set;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 7:47 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class WordEmbeddingLayer extends EmbeddingLayer {
    /**
     * For using the pre-computation trick for different slots.
     */
    private HashMap<Integer, Integer>[] precomputationMap;

    /**
     * @param nIn    Vocabulary size.
     * @param nOut   Embedding dimension.
     * @param random
     */
    public WordEmbeddingLayer(int nIn, int nOut, Random random, HashMap<Integer, Integer>[] precomputationMap) {
        super(nIn, nOut, random);
        this.precomputationMap = precomputationMap;
    }

    public boolean isFrequent(int index, int wordId) {
        if (precomputationMap == null) return false;
        return precomputationMap[index].containsKey(wordId);
    }

    public int preComputeId(int index, int wordId) {
        return precomputationMap[index].get(wordId);
    }

    public int numOfEmbeddingSlot() {
        return precomputationMap.length;
    }

    public int numOfPrecomputedItems(int index) {
        return precomputationMap[index].size();
    }

    public Set<Integer> preComputedIds(int index) {
        return precomputationMap[index].keySet();
    }

    public void emptyPrecomputedMap() {
        this.precomputationMap = null;
    }

    public final HashMap<Integer, Integer>[] getPrecomputationMap() {
        return precomputationMap;
    }

    public void setPrecomputationMap(HashMap<Integer, Integer>[] precomputationMap) {
        this.precomputationMap = precomputationMap;
    }

    @Override
    public void setLayer(Layer layer) {
        super.setLayer(layer);
        precomputationMap = ((WordEmbeddingLayer) layer).getPrecomputationMap();
    }
}
