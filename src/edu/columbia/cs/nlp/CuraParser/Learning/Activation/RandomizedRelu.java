package edu.columbia.cs.nlp.CuraParser.Learning.Activation;

import java.util.Random;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/31/16
 * Time: 3:05 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class RandomizedRelu extends Relu {
    double l, u;
    Random random;
    private double diff;
    private double mid;

    public RandomizedRelu(double l, double u, Random random) {
        assert u > l;
        this.l = l;
        this.u = u;
        this.random = random;
        diff = u - l;
        mid = (l + u) / 2;
    }

    @Override
    public double activate(double value, boolean test) {
        if (value >= 0)
            return value;
        if (test)
            return value / mid;
        else {
            double alpha = (random.nextDouble() * diff) + l;
            return value / alpha;
        }
    }

    @Override
    public double gradient(double value, double gradient, double activation, boolean test) {
        if (value >= 0)
            return gradient;
        if (test)
            return gradient / mid;
        else {
            double alpha = value / activation;
            return gradient / alpha;
        }
    }
}
