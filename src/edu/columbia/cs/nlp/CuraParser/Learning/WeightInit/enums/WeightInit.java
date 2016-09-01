package edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.enums;

import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.*;

import java.util.Random;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/19/16
 * Time: 10:55 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public enum WeightInit {
    FIX,
    NORMAL,
    RELU,
    UNIFORM,
    XAVIER;

    public static Initializer initializer(WeightInit weightInit, Random random, int nIn, int nOut, double initValue) {
        switch (weightInit) {
            case FIX:
                return new FixInit(initValue);
            case NORMAL:
                return new NormalInit(random, nIn);
            case RELU:
                return new ReluInit(random, nIn, nOut);
            case UNIFORM:
                return new UniformInit(random, nOut);
            case XAVIER:
                return new XavierInit(random, nIn, nOut);
            default:
                return null;
        }
    }

    public static Initializer initializer(WeightInit weightInit, Random random, double stdDev) {
        switch (weightInit) {
            case NORMAL:
                return new NormalInit(random, stdDev);
            case UNIFORM:
                return new UniformInit(random, stdDev);
            default:
                return null;
        }
    }
}
