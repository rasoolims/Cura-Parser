package edu.columbia.cs.nlp.CuraParser.Learning.Updater;

import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.Layer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Structures.Enums.EmbeddingTypes;

import java.util.ArrayList;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 6:25 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */
public class Adam extends Updater {
    double beta1, beta2;
    // beta{1,2} ^ t
    double beta1_, beta2_;

    double eps;

    public Adam(MLPNetwork mlpNetwork, double learningRate, boolean outputBiasTerm, double beta1, double beta2, double eps) {
        super(mlpNetwork, learningRate, outputBiasTerm);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        ArrayList<Layer> netLayers = mlpNetwork.getLayers();
        this.gradientHistoryVariance = new ArrayList<>(netLayers.size());
        for (int i = 0; i < netLayers.size(); i++)
            gradientHistoryVariance.add(netLayers.get(i).copy(true, false));
    }

    @Override
    public void update(ArrayList<Layer> gradients) throws Exception {
        super.update(gradients);
        beta1_ *= beta1;
        beta2_ *= beta2;
    }

    @Override
    protected void update(double[][] g, double[][] m, double[][] v, int layerIndex) throws Exception {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[i].length; j++) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * g[i][j];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * Math.pow(g[i][j], 2);

                double _m = m[i][j] / (1 - beta1_);
                double _v = v[i][j] / (1 - beta2_);

                mlpNetwork.modify(layerIndex, i, j, -learningRate * _m / (Math.sqrt(_v) + eps));
            }
        }
    }

    @Override
    protected void update(double[][] g, double[][] m, double[][] v, EmbeddingTypes embeddingType) throws Exception {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[i].length; j++) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * g[i][j];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * Math.pow(g[i][j], 2);

                double _m = m[i][j] / (1 - beta1_);
                double _v = v[i][j] / (1 - beta2_);

                mlpNetwork.modify(embeddingType, i, j, -learningRate * _m / (Math.sqrt(_v) + eps));
            }
        }
    }

    @Override
    protected void update(double[] g, double[] m, double[] v, int layerIndex) throws Exception {
        if (g == null) return;
        for (int i = 0; i < g.length; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1 - beta2) * Math.pow(g[i], 2);

            double _m = m[i] / (1 - beta1_);
            double _v = v[i] / (1 - beta2_);

            mlpNetwork.modify(layerIndex, i, -1, -learningRate * _m / (Math.sqrt(_v) + eps));
        }
    }
}
