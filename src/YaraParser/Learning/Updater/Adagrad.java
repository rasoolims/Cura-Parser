package YaraParser.Learning.Updater;

import YaraParser.Learning.MLPNetwork;
import YaraParser.Learning.NetworkMatrices;
import YaraParser.Structures.EmbeddingTypes;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 4:53 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Adagrad extends Updater {
    NetworkMatrices gradientHistory;
    double eps;

    public Adagrad(MLPNetwork mlpNetwork, double learningRate, double eps) {
        this.mlpNetwork = mlpNetwork;
        this.learningRate = learningRate;
        this.eps = eps;
        gradientHistory = new NetworkMatrices(mlpNetwork.getNumOfWords(), mlpNetwork.getWordEmbeddingSize(), mlpNetwork.getNumOfPos(), mlpNetwork
                .getPosEmbeddingSize(), mlpNetwork.getNumOfDependencyLabels(), mlpNetwork.getLabelEmbeddingSize(), mlpNetwork.getHiddenLayerSize(),
                mlpNetwork.getHiddenLayerIntSize(), mlpNetwork.getSoftmaxLayerSize());
    }

    @Override
    public void update(NetworkMatrices gradients) throws Exception {
        double[][] wordEmbeddingGradient = gradients.getWordEmbedding();
        double[][] wordEmbeddingGradientHistory = gradientHistory.getWordEmbedding();
        for (int i = 0; i < mlpNetwork.getNumOfWords(); i++) {
            for (int j = 0; j < mlpNetwork.getWordEmbeddingSize(); j++) {
                wordEmbeddingGradientHistory[i][j] += Math.pow(wordEmbeddingGradient[i][j], 2);
                mlpNetwork.modify(EmbeddingTypes.WORD, i, j, -learningRate * wordEmbeddingGradient[i][j] / Math.sqrt
                        (wordEmbeddingGradientHistory[i][j] + eps));
            }
        }

        double[][] posEmbeddingGradient = gradients.getPosEmbedding();
        double[][] posEmbeddingGradientHistory = gradientHistory.getPosEmbedding();
        for (int i = 0; i < mlpNetwork.getNumOfPos(); i++) {
            for (int j = 0; j < mlpNetwork.getPosEmbeddingSize(); j++) {
                posEmbeddingGradientHistory[i][j] += Math.pow(posEmbeddingGradient[i][j], 2);
                mlpNetwork.modify(EmbeddingTypes.POS, i, j, -learningRate * posEmbeddingGradient[i][j] / Math.sqrt
                        (posEmbeddingGradientHistory[i][j] + eps));
            }
        }

        double[][] labelEmbeddingGradient = gradients.getLabelEmbedding();
        double[][] labelEmbeddingGradientHistory = gradientHistory.getLabelEmbedding();
        for (int i = 0; i < mlpNetwork.getNumOfDependencyLabels(); i++) {
            for (int j = 0; j < mlpNetwork.getLabelEmbeddingSize(); j++) {
                labelEmbeddingGradientHistory[i][j] += Math.pow(labelEmbeddingGradient[i][j], 2);
                mlpNetwork.modify(EmbeddingTypes.DEPENDENCY, i, j, -learningRate *
                        labelEmbeddingGradient[i][j] / Math.sqrt(labelEmbeddingGradientHistory[i][j] + eps));
            }
        }

        double[][] hiddenLayerGradient = gradients.getHiddenLayer();
        double[][] hiddenLayerGradientHistory = gradientHistory.getHiddenLayer();
        for (int i = 0; i < mlpNetwork.getHiddenLayerSize(); i++) {
            for (int j = 0; j < mlpNetwork.getHiddenLayerIntSize(); j++) {
                hiddenLayerGradientHistory[i][j] += Math.pow(hiddenLayerGradient[i][j], 2);
                mlpNetwork.modify(EmbeddingTypes.HIDDENLAYER, i, j, -learningRate * hiddenLayerGradient[i][j] / Math.sqrt
                        (hiddenLayerGradientHistory[i][j] + eps));
            }
        }

        double[] hiddenLayerBiasGradient = gradients.getHiddenLayerBias();
        double[] hiddenLayerBiasGradientHistory = gradientHistory.getHiddenLayerBias();
        for (int i = 0; i < mlpNetwork.getHiddenLayerSize(); i++) {
            hiddenLayerBiasGradientHistory[i] += Math.pow(hiddenLayerBiasGradient[i], 2);
            mlpNetwork.modify(EmbeddingTypes.HIDDENLAYERBIAS, i, -1, -learningRate * hiddenLayerBiasGradient[i] / Math.sqrt(
                    hiddenLayerBiasGradientHistory[i] + eps));
        }

        double[][] softmaxLayerGradient = gradients.getSoftmaxLayer();
        double[][] softmaxLayerGradientHistory = gradientHistory.getSoftmaxLayer();
        for (int i = 0; i < mlpNetwork.getSoftmaxLayerSize(); i++) {
            for (int j = 0; j < mlpNetwork.getHiddenLayerSize(); j++) {
                softmaxLayerGradientHistory[i][j] += Math.pow(softmaxLayerGradient[i][j], 2);
                mlpNetwork.modify(EmbeddingTypes.SOFTMAX, i, j, -learningRate * softmaxLayerGradient[i][j] / Math.sqrt
                        (softmaxLayerGradientHistory[i][j] + eps));
            }
        }

        double[] softmaxLayerBiasGradient = gradients.getSoftmaxLayerBias();
        double[] softmaxLayerBiasGradientHistory = gradientHistory.getSoftmaxLayerBias();
        for (int i = 0; i < mlpNetwork.getSoftmaxLayerSize(); i++) {
            softmaxLayerBiasGradientHistory[i] += Math.pow(softmaxLayerBiasGradient[i], 2);
            mlpNetwork.modify(EmbeddingTypes.SOFTMAXBIAS, i, -1, -learningRate * softmaxLayerBiasGradient[i] / Math.sqrt(
                    softmaxLayerBiasGradientHistory[i] + eps));
        }
    }
}
