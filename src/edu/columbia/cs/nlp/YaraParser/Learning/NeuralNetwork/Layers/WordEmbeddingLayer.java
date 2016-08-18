package edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.Layers;

import java.util.HashMap;
import java.util.Random;

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
    HashMap<Integer, Integer>[] precomputationMap;

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
        return precomputationMap[index].containsKey(wordId);
    }

    public int preComputeId(int index, int wordId) {
        return precomputationMap[index].get(wordId);
    }
}
