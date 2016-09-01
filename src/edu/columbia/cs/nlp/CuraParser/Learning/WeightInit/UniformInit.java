package edu.columbia.cs.nlp.CuraParser.Learning.WeightInit;

import java.util.Random;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 3:17 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class UniformInit extends Initializer {
    public UniformInit(Random random, int nIn) {
        super(random, 0, nIn);
        this.stdDev = Math.sqrt(1.0 / nIn);
    }

    public UniformInit(Random random, double stdDev) {
        super(random, stdDev);
    }

    @Override
    public double next() {
        return random.nextDouble() * 2 * stdDev - stdDev;
    }
}
