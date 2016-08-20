package edu.columbia.cs.nlp.CuraParser.Learning.Updater;

import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.FirstHiddenLayer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.Layer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Structures.Enums.EmbeddingTypes;

import java.util.ArrayList;

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
    ArrayList<Layer> gradientHistory;
    // only for ADAM/ADAMAX
    ArrayList<Layer> gradientHistoryVariance;
    // iteration.
    int t;
    boolean outputBiasTerm;

    public Updater(MLPNetwork mlpNetwork, double learningRate, boolean outputBiasTerm) {
        this.mlpNetwork = mlpNetwork;
        this.learningRate = learningRate;
        ArrayList<Layer> netLayers = mlpNetwork.getLayers();
        gradientHistory = new ArrayList<>(netLayers.size());
        for (int i = 0; i < netLayers.size(); i++)
            gradientHistory.add(netLayers.get(i).copy(true, false));
        gradientHistoryVariance = null;
        t = 1;
        this.outputBiasTerm = outputBiasTerm;
    }

    public void update(ArrayList<Layer> gradients) throws Exception {
        for (int i = 0; i < gradients.size(); i++) {
            update(gradients.get(i).getW(), gradientHistory.get(i).getW(), gradientHistoryVariance == null ? null :
                    gradientHistoryVariance.get(i).getW(), i);
            update(gradients.get(i).getB(), gradientHistory.get(i).getB(), gradientHistoryVariance == null ? null :
                    gradientHistoryVariance.get(i).getB(), i);

            if (i == 0) {
                FirstHiddenLayer layer = (FirstHiddenLayer) gradients.get(i);
                FirstHiddenLayer layerH = (FirstHiddenLayer) gradientHistory.get(i);
                update(layer.getWordEmbeddings().getW(), layerH.getWordEmbeddings().getW(), gradientHistoryVariance == null ? null :
                        ((FirstHiddenLayer) gradientHistoryVariance.get(i)).getWordEmbeddings().getW(), EmbeddingTypes.WORD);
                update(layer.getPosEmbeddings().getW(), layerH.getPosEmbeddings().getW(), gradientHistoryVariance == null ? null :
                        ((FirstHiddenLayer) gradientHistoryVariance.get(i)).getPosEmbeddings().getW(), EmbeddingTypes.POS);
                update(layer.getDepEmbeddings().getW(), layerH.getDepEmbeddings().getW(), gradientHistoryVariance == null ? null :
                        ((FirstHiddenLayer) gradientHistoryVariance.get(i)).getDepEmbeddings().getW(), EmbeddingTypes.DEPENDENCY);
            }
        }
        t++;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    protected abstract void update(double[][] g, double[][] h, double[][] v, int i) throws Exception;

    protected abstract void update(double[][] g, double[][] h, double[][] v, EmbeddingTypes types) throws Exception;

    protected abstract void update(double[] g, double[] h, double[] v, int i) throws Exception;

    public final ArrayList<Layer> getGradientHistory() {
        return gradientHistory;
    }
}
