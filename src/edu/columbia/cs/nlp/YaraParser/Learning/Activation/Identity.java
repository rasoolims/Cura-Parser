package edu.columbia.cs.nlp.YaraParser.Learning.Activation;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 7:34 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Identity extends Activation{
    @Override
    public double activate(double value) {
        return value;
    }

    @Override
    public double gradient(double value, double gradient) {
        return gradient;
    }
}
