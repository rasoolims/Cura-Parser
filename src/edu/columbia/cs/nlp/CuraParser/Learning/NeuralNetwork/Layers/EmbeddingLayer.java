package edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers;

import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Learning.Activation.Identity;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.UniformInit;

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
     * @param nIn Embedding dimension.
     * @param nOut Vocabulary size.
     */
    public EmbeddingLayer(int nIn, int nOut, Random random) {
        super(new Identity(), nIn, nOut, new UniformInit(random, nIn), null, false);
    }

    public int dim() {
        return nIn();
    }

    public int vocabSize() {
        return nOut();
    }

    public double[] w(int index) {
        return w[index];
    }

    public void addPretrainedVectors(HashMap<Integer, double[]> embeddingsDictionary) {
        if (embeddingsDictionary == null) return;
        int numOfPretrained = 0;
        for (int i = 0; i < vocabSize(); i++) {
            double[] embeddings = embeddingsDictionary.get(i);
            if (embeddings != null) {
                w[i] = embeddings;
                numOfPretrained++;
            }
        }
        System.out.println("num of pre-trained embedding " + numOfPretrained + " out of " + nOut());
    }
}
