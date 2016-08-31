package edu.columbia.cs.nlp.CuraParser.Learning.Activation;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/7/16
 * Time: 1:53 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Relu extends Activation {
    @Override
    public double activate(double value, boolean test) {
        return Math.max(0, value);
    }

    @Override
    public double gradient(double value, double gradient, double activation, boolean test) {
        return (value <= 0 ? 0 : gradient);
    }
}
