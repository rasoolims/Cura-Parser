package YaraParser.Learning.Updater;

import YaraParser.Learning.MLPNetwork;
import YaraParser.Learning.NetworkMatrices;
import YaraParser.Structures.EmbeddingTypes;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 4:14 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class SgdWithMomentumUpdater extends Updater {
    NetworkMatrices gradientHistory;
    double momentum;
    MLPNetwork mlpNetwork;

    public SgdWithMomentumUpdater(MLPNetwork mlpNetwork, double learningRate, double momentum) {
        this.mlpNetwork = mlpNetwork;
        this.learningRate = learningRate;
        this.momentum = momentum;
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
                wordEmbeddingGradientHistory[i][j] = momentum * wordEmbeddingGradientHistory[i][j] - wordEmbeddingGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.WORD, i, j, learningRate * wordEmbeddingGradientHistory[i][j]);
            }
        }

        double[][] posEmbeddingGradient = gradients.getPosEmbedding();
        double[][] posEmbeddingGradientHistory = gradientHistory.getPosEmbedding();
        for (int i = 0; i < mlpNetwork.getNumOfPos(); i++) {
            for (int j = 0; j < mlpNetwork.getPosEmbeddingSize(); j++) {
                posEmbeddingGradientHistory[i][j] = momentum * posEmbeddingGradientHistory[i][j] - posEmbeddingGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.POS, i, j, learningRate * posEmbeddingGradientHistory[i][j]);
            }
        }

        double[][] labelEmbeddingGradient = gradients.getLabelEmbedding();
        double[][] labelEmbeddingGradientHistory = gradientHistory.getLabelEmbedding();
        for (int i = 0; i < mlpNetwork.getNumOfDependencyLabels(); i++) {
            for (int j = 0; j < mlpNetwork.getLabelEmbeddingSize(); j++) {
                labelEmbeddingGradientHistory[i][j] = momentum * labelEmbeddingGradientHistory[i][j] - labelEmbeddingGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.DEPENDENCY, i, j, learningRate * labelEmbeddingGradientHistory[i][j]);
            }
        }

        double[][] hiddenLayerGradient = gradients.getHiddenLayer();
        double[][] hiddenLayerGradientHistory = gradientHistory.getHiddenLayer();
        for (int i = 0; i < mlpNetwork.getHiddenLayerSize(); i++) {
            for (int j = 0; j < mlpNetwork.getHiddenLayerIntSize(); j++) {
                hiddenLayerGradientHistory[i][j] = momentum * hiddenLayerGradientHistory[i][j] - hiddenLayerGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.HIDDENLAYER, i, j, learningRate * hiddenLayerGradientHistory[i][j]);
            }
        }

        double[] hiddenLayerBiasGradient = gradients.getHiddenLayerBias();
        double[] hiddenLayerBiasGradientHistory = gradientHistory.getHiddenLayerBias();
        for (int i = 0; i < mlpNetwork.getHiddenLayerSize(); i++) {
            hiddenLayerBiasGradientHistory[i] = momentum * hiddenLayerBiasGradientHistory[i] - hiddenLayerBiasGradient[i];
            mlpNetwork.modify(EmbeddingTypes.HIDDENLAYERBIAS, i, -1, learningRate * hiddenLayerBiasGradientHistory[i]);
        }

        double[][] softmaxLayerGradient = gradients.getSoftmaxLayer();
        double[][] softmaxLayerGradientHistory = gradientHistory.getSoftmaxLayer();
        for (int i = 0; i < mlpNetwork.getSoftmaxLayerSize(); i++) {
            for (int j = 0; j < mlpNetwork.getHiddenLayerSize(); j++) {
                softmaxLayerGradientHistory[i][j] = momentum * softmaxLayerGradientHistory[i][j] - softmaxLayerGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.SOFTMAX, i, j, learningRate * softmaxLayerGradientHistory[i][j]);
            }
        }

        double[] softmaxLayerBiasGradient = gradients.getSoftmaxLayerBias();
        double[] softmaxLayerBiasGradientHistory = gradientHistory.getSoftmaxLayerBias();
        for (int i = 0; i < mlpNetwork.getSoftmaxLayerSize(); i++) {
            softmaxLayerBiasGradientHistory[i] = momentum * softmaxLayerBiasGradientHistory[i] - softmaxLayerBiasGradient[i];
            mlpNetwork.modify(EmbeddingTypes.SOFTMAXBIAS, i, -1, learningRate * softmaxLayerBiasGradientHistory[i]);
        }
    }
}
