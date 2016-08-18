package edu.columbia.cs.nlp.YaraParser.Learning.Activation;

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

    public abstract double activate(double value);

    public double[] activate(double[] values) {
        double[] a = new double[values.length];
        for(int i=0; i<a.length;i++) a[i] = activate(values[i]);
        return a;
    }

    public abstract double gradient(double value, double gradient);

    public double[] gradient(double[] values, double[] gradients) {
        double[] g = new double[values.length];
        for(int i=0; i<g.length;i++) g[i] = gradient(values[i], gradients[i]);
        return g;
    }
}
