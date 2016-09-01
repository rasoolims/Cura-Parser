package edu.columbia.cs.nlp.CuraParser.Learning.WeightInit;

import java.util.Random;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 3:10 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public abstract class Initializer {
    Random random;
    int nIn;
    int nOut;
    double stdDev;

    public Initializer(Random random, int nIn, int nOut) {
        this.random = random;
        this.nIn = nIn;
        this.nOut = nOut;
    }

    public Initializer(Random random, double stdDev) {
        this.random = random;
        this.stdDev = stdDev;
    }

    public abstract double next();

    public void init(double[][] m) {
        for (int i = 0; i < m.length; i++)
            init(m[i]);
    }

    public void init(double[] v) {
        for (int i = 0; i < v.length; i++)
            v[i] = next();
    }
}
