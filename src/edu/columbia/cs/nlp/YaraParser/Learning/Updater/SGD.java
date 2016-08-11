package edu.columbia.cs.nlp.YaraParser.Learning.Updater;

import edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.YaraParser.Structures.Enums.EmbeddingTypes;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 4:14 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class SGD extends Updater {
    double momentum;

    // For Nesterov accelerated gradient.
    boolean accelerated;

    public SGD(MLPNetwork mlpNetwork, double learningRate, double momentum, boolean accelerated) {
        super(mlpNetwork, learningRate);
        this.momentum = momentum;
    }


    @Override
    protected void update(double[][] g, double[][] h, double[][] v, EmbeddingTypes embeddingTypes) throws Exception {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[i].length; j++) {
                if (!accelerated) {
                    h[i][j] = momentum * h[i][j] - g[i][j];
                    mlpNetwork.modify(embeddingTypes, i, j, learningRate * h[i][j]);
                } else {
                    double hPrev = h[i][j];
                    h[i][j] = momentum * h[i][j] - learningRate * g[i][j];
                    mlpNetwork.modify(embeddingTypes, i, j, -momentum * hPrev + (1 + momentum) * h[i][j]);
                }
            }
        }
    }

    @Override
    protected void update(double[] g, double[] h, double[] v, EmbeddingTypes embeddingTypes) throws Exception {
        for (int i = 0; i < g.length; i++) {
            if (accelerated) {
                h[i] = momentum * h[i] - g[i];
                mlpNetwork.modify(embeddingTypes, i, -1, learningRate * h[i]);
            } else {
                double hPrev = h[i];
                h[i] = momentum * h[i] - learningRate * g[i];
                mlpNetwork.modify(embeddingTypes, i, -1, -momentum * hPrev + (1 + momentum) * h[i]);
            }
        }
    }
    /**

     v_prev = v # back this up
     v = mu * v - learning_rate * dx # velocity update stays the same
     x += -mu * v_prev + (1 + mu) * v # position update changes form
     */
}
