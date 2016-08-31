package edu.columbia.cs.nlp.CuraParser.Learning.Activation;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/31/16
 * Time: 2:33 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class LeakyRelu extends Relu {
    double alpha;

    public LeakyRelu(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public double activate(double value, boolean test) {
        if (value >= 0)
            return value;
        return value / alpha;
    }

    @Override
    public double gradient(double value, double gradient, double activation, boolean test) {
        return (value <= 0 ? gradient / alpha : gradient);
    }
}
