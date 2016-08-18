package edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.Layers;

import edu.columbia.cs.nlp.YaraParser.Learning.Activation.Identity;
import edu.columbia.cs.nlp.YaraParser.Learning.WeightInit.UniformInit;

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
}
