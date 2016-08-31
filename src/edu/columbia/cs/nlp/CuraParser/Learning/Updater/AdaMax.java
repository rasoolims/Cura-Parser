package edu.columbia.cs.nlp.CuraParser.Learning.Updater;

import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Structures.Enums.EmbeddingTypes;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/5/16
 * Time: 12:58 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class AdaMax extends Adam {
    public AdaMax(MLPNetwork mlpNetwork, double learningRate, boolean outputBiasTerm, double beta1, double beta2, double eps) {
        super(mlpNetwork, learningRate, outputBiasTerm, beta1, beta2, eps);
    }

    @Override
    protected void update(double[][] g, double[][] m, double[][] v, EmbeddingTypes embeddingTypes) throws Exception {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[i].length; j++) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * g[i][j];
                v[i][j] = Math.max(beta2 * v[i][j], Math.abs(g[i][j]));
                double change = (-learningRate / (1 - beta1_)) * m[i][j] / v[i][j];
                if (Double.isNaN(change) || Double.isInfinite(change))
                    change = 0;
                mlpNetwork.modify(embeddingTypes, i, j, change);
            }
        }
    }

    @Override
    protected void update(double[][] g, double[][] m, double[][] v, int layerIndex) throws Exception {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[i].length; j++) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * g[i][j];
                v[i][j] = Math.max(beta2 * v[i][j], Math.abs(g[i][j]));
                double change = (-learningRate / (1 - beta1_)) * m[i][j] / v[i][j];
                if (Double.isNaN(change) || Double.isInfinite(change))
                    change = 0;
                mlpNetwork.modify(layerIndex, i, j, change);
            }
        }
    }

    @Override
    protected void update(double[] g, double[] m, double[] v, int layerIndex) throws Exception {
        if (g == null) return;
        for (int i = 0; i < g.length; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * g[i];
            v[i] = Math.max(beta2 * v[i], Math.abs(g[i]));
            double change = (-learningRate / (1 - beta1_)) * m[i] / v[i];
            if (Double.isNaN(change) || Double.isInfinite(change))
                change = 0;
            mlpNetwork.modify(layerIndex, i, -1, change);
        }
    }
}
