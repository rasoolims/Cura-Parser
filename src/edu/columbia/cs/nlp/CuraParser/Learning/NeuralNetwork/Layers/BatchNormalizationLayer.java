package edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers;

import edu.columbia.cs.nlp.CuraParser.Learning.Activation.Identity;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.FixInit;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.HashSet;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/30/16
 * Time: 10:39 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class BatchNormalizationLayer extends Layer {
    double eps;

    public BatchNormalizationLayer(int nIn, double eps) {
        super(new Identity(), 1, nIn, new FixInit(1));
        this.eps = eps;
    }

    @Override
    public double[][] forward(double[][] x) {
        // calculating mean
        double[] mu = new double[x[0].length];
        for (double[] i : x) {
            for (int j = 0; j < i.length; j++)
                mu[j] += i[j];
        }
        for (int j = 0; j < mu.length; j++)
            mu[j] /= x.length;

        // calculating variance
        double[] variance = new double[x[0].length];
        for (double[] i : x) {
            for (int j = 0; j < i.length; j++)
                variance[j] += Math.pow(mu[j] - i[j], 2);
        }
        for (int j = 0; j < variance.length; j++)
            variance[j] /= x.length;

        // calculating inverse std-dev
        double[] invStdDev = new double[x[0].length];
        for (int j = 0; j < invStdDev.length; j++)
            invStdDev[j] = 1.0 / Math.sqrt(variance[j] + eps);

        // Calculating xHat
        double[][] xHat = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                xHat[i][j] = (x[i][j] - mu[j]) * invStdDev[j];
            }
        }

        // Calculating output
        double[][] y = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                y[i][j] = w(j, 0) * xHat[i][j] + b(j);
            }
        }

        return y;
    }

    @Override
    public double[] forward(double[] i, double[] labels, boolean takeLog) {
        throw new NotImplementedException();
    }

    @Override
    public double[] forward(double[] i, double[] labels) {
        throw new NotImplementedException();
    }

    @Override
    public double[][] forward(double[][] i, HashSet<Integer>[] wIndexToUse, HashSet<Integer>[] inputToUse) {
        throw new NotImplementedException();
    }

    @Override
    public double[][] forward(double[][] i, HashSet<Integer>[] wIndexToUse) {
        throw new NotImplementedException();
    }

    @Override
    public double[][] forward(double[][] i, double[][] labels, boolean takeLog) {
        throw new NotImplementedException();
    }
}

