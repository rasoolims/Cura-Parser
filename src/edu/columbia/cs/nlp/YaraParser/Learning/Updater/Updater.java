package edu.columbia.cs.nlp.YaraParser.Learning.Updater;

import edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.NetworkMatrices;
import edu.columbia.cs.nlp.YaraParser.Structures.Enums.EmbeddingTypes;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 4:14 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public abstract class Updater {
    MLPNetwork mlpNetwork;
    double learningRate;
    NetworkMatrices gradientHistory;
    // only for ADAM/ADAMAX
    NetworkMatrices gradientHistoryVariance;
    // iteration.
    int t;
    boolean outputBiasTerm;

    public Updater(MLPNetwork mlpNetwork, double learningRate, boolean outputBiasTerm) {
        this.mlpNetwork = mlpNetwork;
        this.learningRate = learningRate;
        gradientHistory = new NetworkMatrices(mlpNetwork.getNumWords(), mlpNetwork.getWordEmbedDim(), mlpNetwork.getNumPos(),
                mlpNetwork.getPosEmbedDim(), mlpNetwork.getNumDepLabels(), mlpNetwork.getDepEmbedDim(), mlpNetwork.getHiddenLayerDim(),
                mlpNetwork.getHiddenLayerIntDim(), mlpNetwork.getSecondHiddenLayerDim(), mlpNetwork.getSoftmaxLayerDim());
        gradientHistoryVariance = null;
        t = 1;
        this.outputBiasTerm = outputBiasTerm;
    }

    public void update(NetworkMatrices gradients) throws Exception {
        update(gradients.getWordEmbedding(), gradientHistory.getWordEmbedding(),
                gradientHistoryVariance == null ? null : gradientHistoryVariance.getWordEmbedding(), EmbeddingTypes.WORD);
        update(gradients.getPosEmbedding(), gradientHistory.getPosEmbedding(),
                gradientHistoryVariance == null ? null : gradientHistoryVariance.getPosEmbedding(), EmbeddingTypes.POS);
        update(gradients.getLabelEmbedding(), gradientHistory.getLabelEmbedding(),
                gradientHistoryVariance == null ? null : gradientHistoryVariance.getLabelEmbedding(), EmbeddingTypes.DEPENDENCY);
        update(gradients.getHiddenLayer(), gradientHistory.getHiddenLayer(),
                gradientHistoryVariance == null ? null : gradientHistoryVariance.getHiddenLayer(), EmbeddingTypes.HIDDENLAYER);
        update(gradients.getHiddenLayerBias(), gradientHistory.getHiddenLayerBias(),
                gradientHistoryVariance == null ? null : gradientHistoryVariance.getHiddenLayerBias(), EmbeddingTypes.HIDDENLAYERBIAS);
        update(gradients.getSoftmaxLayer(), gradientHistory.getSoftmaxLayer(),
                gradientHistoryVariance == null ? null : gradientHistoryVariance.getSoftmaxLayer(), EmbeddingTypes.SOFTMAX);
        if (outputBiasTerm)
            update(gradients.getSoftmaxLayerBias(), gradientHistory.getSoftmaxLayerBias(),
                    gradientHistoryVariance == null ? null : gradientHistoryVariance.getSoftmaxLayerBias(), EmbeddingTypes.SOFTMAXBIAS);
        t++;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    protected abstract void update(double[][] g, double[][] h, double[][] v, EmbeddingTypes embeddingTypes) throws Exception;

    protected abstract void update(double[] g, double[] h, double[] v, EmbeddingTypes embeddingTypes) throws Exception;

    public final NetworkMatrices getGradientHistory() {
        return gradientHistory;
    }
}
