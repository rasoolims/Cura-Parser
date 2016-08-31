package edu.columbia.cs.nlp.CuraParser.Learning.Updater;

import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.SGDType;
import edu.columbia.cs.nlp.CuraParser.Structures.Enums.EmbeddingTypes;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 4:14 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class SGD extends Updater {
    double momentum;
    SGDType type;

    public SGD(MLPNetwork mlpNetwork, double learningRate, boolean outputBiasTerm, double momentum, SGDType type) {
        super(mlpNetwork, learningRate, outputBiasTerm);
        this.momentum = momentum;
        this.type = type;
    }

    @Override
    protected void update(double[][] g, double[][] h, double[][] v, EmbeddingTypes embeddingTypes) throws Exception {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[i].length; j++) {
                if (type == SGDType.VANILLA) {
                    mlpNetwork.modify(embeddingTypes, i, j, -learningRate * g[i][j]);
                } else if (type == SGDType.MOMENTUM) {
                    h[i][j] = momentum * h[i][j] - g[i][j];
                    mlpNetwork.modify(embeddingTypes, i, j, learningRate * h[i][j]);
                } else if (type == SGDType.NESTEROV) {
                    double hPrev = h[i][j];
                    h[i][j] = momentum * h[i][j] - learningRate * g[i][j];
                    mlpNetwork.modify(embeddingTypes, i, j, -momentum * hPrev + (1 + momentum) * h[i][j]);
                } else {
                    throw new Exception("SGD type not supported");
                }
            }
        }
    }

    @Override
    protected void update(double[][] g, double[][] h, double[][] v, int layerIndex) throws Exception {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[i].length; j++) {
                if (type == SGDType.VANILLA) {
                    mlpNetwork.modify(layerIndex, i, j, -learningRate * g[i][j]);
                } else if (type == SGDType.MOMENTUM) {
                    h[i][j] = momentum * h[i][j] - g[i][j];
                    mlpNetwork.modify(layerIndex, i, j, learningRate * h[i][j]);
                } else if (type == SGDType.NESTEROV) {
                    double hPrev = h[i][j];
                    h[i][j] = momentum * h[i][j] - learningRate * g[i][j];
                    mlpNetwork.modify(layerIndex, i, j, -momentum * hPrev + (1 + momentum) * h[i][j]);
                } else {
                    throw new Exception("SGD type not supported");
                }
            }
        }
    }

    @Override
    protected void update(double[] g, double[] h, double[] v, int layerIndex) throws Exception {
        if (g == null) return;
        for (int i = 0; i < g.length; i++) {
            if (type == SGDType.VANILLA) {
                mlpNetwork.modify(layerIndex, i, -1, -learningRate * g[i]);
            } else if (type == SGDType.MOMENTUM) {
                h[i] = momentum * h[i] - g[i];
                mlpNetwork.modify(layerIndex, i, -1, learningRate * h[i]);
            } else if (type == SGDType.NESTEROV) {
                double hPrev = h[i];
                h[i] = momentum * h[i] - learningRate * g[i];
                mlpNetwork.modify(layerIndex, i, -1, -momentum * hPrev + (1 + momentum) * h[i]);
            } else {
                throw new Exception("SGD type not supported");
            }
        }
    }
}