package YaraParser.Learning.Updater;

import YaraParser.Learning.NeuralNetwork.MLPNetwork;
import YaraParser.Structures.Enums.EmbeddingTypes;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 4:53 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Adagrad extends Updater {
    double eps;

    public Adagrad(MLPNetwork mlpNetwork, double learningRate, double eps) {
        super(mlpNetwork, learningRate);
        this.eps = eps;
    }

    @Override
    protected void update(double[][] g, double[][] h, double[][] v, EmbeddingTypes embeddingTypes) throws Exception {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[i].length; j++) {
                h[i][j] += Math.pow(g[i][j], 2);
                mlpNetwork.modify(embeddingTypes, i, j, -learningRate * g[i][j] / (Math.sqrt(h[i][j] + eps)));
            }
        }
    }

    @Override
    protected void update(double[] g, double[] h, double[] v, EmbeddingTypes embeddingTypes) throws Exception {
        for (int i = 0; i < g.length; i++) {
            h[i] += Math.pow(g[i], 2);
            mlpNetwork.modify(embeddingTypes, i, -1, -learningRate * g[i] / (Math.sqrt(h[i] + eps)));
        }
    }
}
