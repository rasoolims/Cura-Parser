package edu.columbia.cs.nlp.CuraParser.Learning.Activation;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/7/16
 * Time: 1:57 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Cubic extends Activation {
    @Override
    public double activate(double value, boolean test) {
        return Math.pow(value, 3);
    }

    @Override
    public double gradient(double value, double gradient, double activation, boolean test) {
        return 3 * gradient * Math.pow(value, 2);
    }
}
