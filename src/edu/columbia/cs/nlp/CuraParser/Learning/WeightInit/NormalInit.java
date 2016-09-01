package edu.columbia.cs.nlp.CuraParser.Learning.WeightInit;

import java.util.Random;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 3:16 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class NormalInit extends Initializer {
    public NormalInit(Random random, int nIn) {
        super(random, nIn, 0);
        this.stdDev = Math.sqrt(1.0 / nIn);
    }

    public NormalInit(Random random, double stdDev) {
        super(random, stdDev);
    }

    @Override
    public double next() {
        return random.nextGaussian() * stdDev;
    }
}
