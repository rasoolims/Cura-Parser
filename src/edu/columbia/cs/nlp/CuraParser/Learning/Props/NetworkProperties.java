package edu.columbia.cs.nlp.CuraParser.Learning.Props;

import edu.columbia.cs.nlp.CuraParser.Learning.Activation.Enums.ActivationType;

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
    public int beamBatchSize;
    public double dropoutProbForHiddenLayer;
    public ActivationType activationType;
    public double regularization;
    public boolean outputBiasTerm;
    public boolean regualarizeAllLayers;

    public NetworkProperties() {
        hiddenLayer1Size = 256;
        hiddenLayer2Size = 256;
        outputBiasTerm = true;
        regularization = 1e-4;
        regualarizeAllLayers = false;
        batchSize = 1000;
        beamBatchSize = 8;
        dropoutProbForHiddenLayer = 0;
        activationType = ActivationType.RELU;
        posDim = 32;
        depDim = 32;
        wDim = 64;
    }

    private NetworkProperties(int hiddenLayer1Size, int hiddenLayer2Size, int posDim, int depDim, int wDim, int batchSize, int beamBatchSize, double
            dropoutProbForHiddenLayer, ActivationType activationType, double regularization, boolean outputBiasTerm, boolean regualarizeAllLayers) {
        this.hiddenLayer1Size = hiddenLayer1Size;
        this.hiddenLayer2Size = hiddenLayer2Size;
        this.posDim = posDim;
        this.depDim = depDim;
        this.wDim = wDim;
        this.batchSize = batchSize;
        this.beamBatchSize = beamBatchSize;
        this.dropoutProbForHiddenLayer = dropoutProbForHiddenLayer;
        this.activationType = activationType;
        this.regularization = regularization;
        this.outputBiasTerm = outputBiasTerm;
        this.regualarizeAllLayers = regualarizeAllLayers;
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
        builder.append("beam batch size: " + beamBatchSize + "\n");
        builder.append("dropout probability: " + dropoutProbForHiddenLayer + "\n");
        builder.append("word dim: " + wDim + "\n");
        builder.append("pos dim: " + posDim + "\n");
        builder.append("dep dim: " + depDim + "\n");
        return builder.toString();
    }

    @Override
    public NetworkProperties clone() {
        return new NetworkProperties(hiddenLayer1Size, hiddenLayer2Size, posDim, depDim, wDim, batchSize, beamBatchSize, dropoutProbForHiddenLayer,
                activationType, regularization, outputBiasTerm, regualarizeAllLayers);
    }
}
