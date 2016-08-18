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
     * For using the precomputation trick.
     */
    HashMap<Integer, Integer> precomputationMap;

    /**
     * @param nIn    Vocabulary size.
     * @param nOut   Embedding dimension.
     * @param random
     */
    public WordEmbeddingLayer(int nIn, int nOut, Random random, HashMap<Integer, Integer> precomputationMap) {
        super(nIn, nOut, random);
        this.precomputationMap = precomputationMap;
    }

    public boolean isFrequent(int index) {
        return precomputationMap.containsKey(index);
    }

    public int preComputeId(int index) {
        return precomputationMap.get(index);
    }
}
