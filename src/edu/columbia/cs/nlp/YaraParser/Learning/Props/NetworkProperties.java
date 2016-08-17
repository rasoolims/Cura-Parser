package edu.columbia.cs.nlp.YaraParser.Learning.Props;

import edu.columbia.cs.nlp.YaraParser.Learning.Activation.Enums.ActivationType;

import java.io.Serializable;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/17/16
 * Time: 11:31 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class NetworkProperties implements Serializable {
    public int hiddenLayer1Size;
    public int hiddenLayer2Size;
    public int posDim;
    public int depDim;
    public int wDim;
    public int batchSize;
    public double dropoutProbForHiddenLayer;
    public ActivationType activationType;
    public double regularization;
    public boolean outputBiasTerm;
    public boolean regualarizeAllLayers;

    public NetworkProperties() {
        regualarizeAllLayers = true;
        hiddenLayer1Size = 200;
        hiddenLayer2Size = 200;
        outputBiasTerm = false;
        regularization = 1e-8;
        regualarizeAllLayers = true;
        batchSize = 1000;
        dropoutProbForHiddenLayer = 0;
        activationType = ActivationType.RELU;
        posDim = 32;
        depDim = 32;
        wDim = 64;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("h1-size: " + hiddenLayer1Size + "\n");
        builder.append("h2-size: " + hiddenLayer2Size + "\n");
        builder.append("activation: " + activationType + "\n");
        builder.append("regularization: " + regularization + "\n");
        builder.append("regularize all layers: " + regualarizeAllLayers + "\n");
        builder.append("batch size: " + batchSize + "\n");
        builder.append("dropout probability: " + dropoutProbForHiddenLayer + "\n");
        builder.append("word dim: " + wDim + "\n");
        builder.append("pos dim: " + posDim + "\n");
        builder.append("dep dim: " + depDim + "\n");
        return builder.toString();
    }
}
