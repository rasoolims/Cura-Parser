package edu.columbia.cs.nlp.CuraParser.Learning.WeightInit;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 3:50 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class FixInit extends Initializer {
    double initVal;

    public FixInit(double initVal) {
        super(null, 0, 0);
        this.initVal = initVal;
    }

    @Override
    public double next() {
        return initVal;
    }
}
