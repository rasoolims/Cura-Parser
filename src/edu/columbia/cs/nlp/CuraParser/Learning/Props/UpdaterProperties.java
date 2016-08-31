package edu.columbia.cs.nlp.CuraParser.Learning.Props;

import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.SGDType;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.UpdaterType;

import java.io.Serializable;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/17/16
 * Time: 11:41 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class UpdaterProperties implements Serializable {
    public double momentum;
    public SGDType sgdType;
    public double learningRate;
    public UpdaterType updaterType;

    public UpdaterProperties() {
        updaterType = UpdaterType.ADAM;
        sgdType = SGDType.NESTEROV;
        // good for ADAM.
        learningRate = 0.0005;
        momentum = 0.9;
    }

    private UpdaterProperties(double momentum, SGDType sgdType, double learningRate, UpdaterType updaterType) {
        this.momentum = momentum;
        this.sgdType = sgdType;
        this.learningRate = learningRate;
        this.updaterType = updaterType;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("updater: ").append(updaterType).append("\n");
        builder.append("learning rate: ").append(learningRate).append("\n");

        if (updaterType == UpdaterType.SGD) {
            builder.append("momentum: ").append(momentum).append("\n");
            builder.append("sgd type: ").append(sgdType).append("\n");
        }
        return builder.toString();
    }

    @Override
    public UpdaterProperties clone() {
        return new UpdaterProperties(momentum, sgdType, learningRate, updaterType);
    }
}
