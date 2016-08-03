package YaraParser.Learning.Updater;

import YaraParser.Learning.MLPNetwork;
import YaraParser.Learning.NetworkMatrices;
import YaraParser.Structures.EmbeddingTypes;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 6:25 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Adam extends Updater {
    double beta1;
    double beta2;
    double eps;

    public Adam(MLPNetwork mlpNetwork, double learningRate, double beta1, double beta2, double eps) {
        super(mlpNetwork, learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.gradientHistoryVariance = new NetworkMatrices(mlpNetwork.getNumOfWords(), mlpNetwork.getWordEmbedDim(), mlpNetwork.getNumOfPos(),
                mlpNetwork.getPosEmbeddingDim(), mlpNetwork.getNumOfDepLabels(), mlpNetwork.getLabelEmbedDim(),
                mlpNetwork.getHiddenLayerDim(), mlpNetwork.getHiddenLayerIntDim(), mlpNetwork.getSoftmaxLayerDim());
    }

    @Override
    protected void update(double[][] g, double[][] m, double[][] v, EmbeddingTypes embeddingTypes) throws Exception {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[i].length; j++) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * g[i][j];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * Math.pow(g[i][j], 2);

                double _m = m[i][j] / (1 - beta1);
                double _v = v[i][j] / (1 - beta2);

                mlpNetwork.modify(embeddingTypes, i, j, -learningRate * _m / (Math.sqrt(_v) + eps));
            }
        }
    }

    @Override
    protected void update(double[] g, double[] m, double[] v, EmbeddingTypes embeddingTypes) throws Exception {
        for (int i = 0; i < g.length; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1 - beta2) * Math.pow(g[i], 2);

            double _m = m[i] / (1 - beta1);
            double _v = v[i] / (1 - beta2);

            mlpNetwork.modify(embeddingTypes, i, -1, -learningRate * _m / (Math.sqrt(_v) + eps));
        }
    }
}
