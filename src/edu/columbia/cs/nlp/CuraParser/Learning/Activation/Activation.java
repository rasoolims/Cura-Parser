package edu.columbia.cs.nlp.CuraParser.Learning.Activation;

import java.io.Serializable;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/7/16
 * Time: 1:50 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public abstract class Activation implements Serializable {
    public Activation() {
    }

    public abstract double activate(double value, boolean test);

    public double[] activate(double[] values, boolean test) {
        double[] a = new double[values.length];
        for (int i = 0; i < a.length; i++) a[i] = activate(values[i], test);
        return a;
    }

    /**
     * Should be usually used for output layers.
     *
     * @param values
     * @param labels  <0 not allowed, 1 gold, >=0 allowed.
     * @param takeLog if to take log of the output.
     * @return
     */
    public double[] activate(double[] values, double[] labels, boolean takeLog, boolean test) {
        return activate(values, test);
    }

    public abstract double gradient(double value, double gradient, double activation, boolean test);

    public double[] gradient(double[] values, double[] gradients, double[] activations, boolean test) {
        double[] g = new double[values.length];
        for (int i = 0; i < g.length; i++) g[i] = gradient(values[i], gradients[i], activations[i], test);
        return g;
    }
}
