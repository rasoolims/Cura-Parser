package edu.columbia.cs.nlp.CuraParser.Learning.Activation;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/19/16
 * Time: 11:29 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class LogisticSoftMax extends Activation {
    @Override
    public double activate(double value, boolean test) {
        return value;
    }

    @Override
    public double gradient(double value, double gradient, double activation, boolean test) {
        return gradient;
    }

    @Override
    public double[] activate(double[] values, double[] labels, boolean takeLog, boolean test) {
        int argmax = 0;
        for (int i = 0; i < values.length; i++) {
            if (labels[i] >= 0)
                if (values[i] >= values[argmax])
                    argmax = i;
        }
        double[] a = new double[values.length];
        double sum = 0;
        for (int i = 0; i < values.length; i++) {
            if (labels[i] >= 0) {
                a[i] = Math.exp(values[i] - values[argmax]);
                sum += a[i];
            }
        }

        for (int i = 0; i < a.length; i++) {
            if (labels[i] >= 0) {
                a[i] /= sum;
                if (takeLog) {
                    a[i] = a[i] != 0. ? Math.log(a[i]) : Math.log(1e-120);
                }
            }
        }
        return a;
    }


}
