package edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.Layers;

import edu.columbia.cs.nlp.YaraParser.Learning.Activation.Identity;
import edu.columbia.cs.nlp.YaraParser.Learning.WeightInit.UniformInit;

import java.util.HashMap;
import java.util.Random;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 7:43 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class EmbeddingLayer extends Layer {
    /**
     * @param nIn  Vocabulary size.
     * @param nOut Embedding dimension.
     */
    public EmbeddingLayer(int nIn, int nOut, Random random) {
        super(new Identity(), nIn, nOut, new UniformInit(random, nOut), null, false);
    }

    public int dim() {
        return nIn();
    }

    public void addPretrainedVectors(HashMap<Integer, double[]> embeddingsDictionary) {
        int numOfPretrained = 0;
        for (int i = 0; i < nOut(); i++) {
            double[] embeddings = embeddingsDictionary.get(i);
            if (embeddings != null) {
                w[i] = embeddings;
                numOfPretrained++;
            }
        }
        System.out.println("num of pre-trained embedding " + numOfPretrained + " out of " + nOut());
    }
}
