package YaraParser.Learning.Updater;

import YaraParser.Learning.NeuralNetwork.MLPNetwork;
import YaraParser.Structures.Enums.EmbeddingTypes;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 4:14 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class SGD extends Updater {
    double momentum;

    public SGD(MLPNetwork mlpNetwork, double learningRate, double momentum) {
        super(mlpNetwork, learningRate);
        this.momentum = momentum;
    }


    @Override
    protected void update(double[][] g, double[][] h, double[][] v, EmbeddingTypes embeddingTypes) throws Exception {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[i].length; j++) {
                h[i][j] = momentum * h[i][j] - g[i][j];
                mlpNetwork.modify(embeddingTypes, i, j, learningRate * h[i][j]);
            }
        }
    }

    @Override
    protected void update(double[] g, double[] h, double[] v, EmbeddingTypes embeddingTypes) throws Exception {
        for (int i = 0; i < g.length; i++) {
            h[i] = momentum * h[i] - g[i];
            mlpNetwork.modify(embeddingTypes, i, -1, learningRate * h[i]);
        }
    }
}
