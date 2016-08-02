package YaraParser.Learning.Updater;

import YaraParser.Learning.MLPNetwork;
import YaraParser.Learning.NetworkMatrices;

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

    public Updater(MLPNetwork mlpNetwork, double learningRate){
        this.mlpNetwork = mlpNetwork;
        this.learningRate = learningRate;
        gradientHistory = new NetworkMatrices(mlpNetwork.getNumOfWords(), mlpNetwork.getWordEmbeddingSize(), mlpNetwork.getNumOfPos(), mlpNetwork
                .getPosEmbeddingSize(), mlpNetwork.getNumOfDependencyLabels(), mlpNetwork.getLabelEmbeddingSize(), mlpNetwork.getHiddenLayerSize(),
                mlpNetwork.getHiddenLayerIntSize(), mlpNetwork.getSoftmaxLayerSize());
    }

    public abstract void update(NetworkMatrices gradients) throws Exception;

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

}
