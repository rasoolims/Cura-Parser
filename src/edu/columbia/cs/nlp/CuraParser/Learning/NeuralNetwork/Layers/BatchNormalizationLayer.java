package edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers;

import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Learning.Activation.Identity;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
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

    // Current cache
    double[] mu;
    double[] variance;
    double[] invStdDev;

    // for Testing
    double[] avgMean;
    double[] avgVariance;
    int iter = 0;

    public BatchNormalizationLayer(int nOut, double eps) {
        super(new Identity(), 1, nOut, new FixInit(1));
        this.eps = eps;
        avgMean = new double[nOut];
        avgVariance = new double[nOut];
    }

    /**
     * This is used for decoding not training.
     * @param x input
     * @return
     */
    @Override
    public double[] forward(double[] x) {
        // Calculating xHat
        double[] xHat = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            xHat[i] = (x[i] - avgMean[i] / iter) / Math.sqrt((avgVariance[i] / iter) + eps);
        }

        // Calculating output
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = gamma(i) * xHat[i] + b(i);
        }
        return y;
    }

    @Override
    public double[][] forward(double[][] x) {
        // calculating mean
        mu = new double[x[0].length];
        for (double[] i : x) {
            for (int j = 0; j < i.length; j++)
                mu[j] += i[j];
        }
        for (int j = 0; j < mu.length; j++)
            mu[j] /= x.length;

        // calculating variance
        variance = new double[x[0].length];
        for (double[] i : x) {
            for (int j = 0; j < i.length; j++)
                variance[j] += Math.pow(mu[j] - i[j], 2);
        }
        for (int j = 0; j < variance.length; j++)
            variance[j] /= x.length;

        // calculating inverse std-dev
        invStdDev = new double[x[0].length];
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
                y[i][j] = gamma(j) * xHat[i][j] + b(j);
            }
        }

        Utils.sumi(avgMean, mu);
        Utils.sumi(avgVariance, variance);
        iter++;
        return y;
    }

    @Override
    public double[] forward(double[] x, double[] labels, boolean takeLog) {
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

    @Override
    public double[] backward(double[] delta, int layerIndex, double[] hInput, double[] prevH, HashSet<Integer>[] seenFeatures, double[][][]
            savedGradients, MLPNetwork network) {
        throw new NotImplementedException();
    }

    @Override
    public double[][] backward(double[][] delta, int layerIndex, double[][] xHat, double[][] x, HashSet<Integer>[] seenFeatures,
                               double[][][] savedGradients, MLPNetwork network) {

        BatchNormalizationLayer layer = (BatchNormalizationLayer) network.layer(layerIndex);
        final double[] variance = layer.variance;
        final double[] invStdDev = layer.invStdDev;
        final double[] mu = layer.mu;

        // dL/db
        for (int i = 0; i < delta.length; i++)
            Utils.sumi(b, delta[i]);

        // dL/dW
        for (int i = 0; i < delta.length; i++)
            Utils.sumi(w[0], Utils.prod(delta[i], xHat[i]));

        // dL/dxHat
        double[][] dxHat = new double[delta.length][delta[0].length];
        for (int i = 0; i < delta.length; i++)
            dxHat[i] = Utils.prod(delta[i], layer.gamma(), 0);

        // dL/dVar
        double[] dvar = new double[variance.length];
        for (int i = 0; i < delta.length; i++) {
            for (int j = 0; j < delta[i].length; j++) {
                dvar[j] += dxHat[i][j] * (x[i][j] - mu[j]) * (-0.5) * Math.pow(variance[j] + eps, -1.5);
            }
        }

        // dL/dmu
        double[] dmu = new double[mu.length];
        for (int i = 0; i < delta.length; i++) {
            for (int j = 0; j < delta[i].length; j++) {
                dmu[j] += dxHat[i][j] * (-1 * invStdDev[j]);
            }
        }

        // dL/dx
        double[][] newDelta = new double[delta.length][delta[0].length];
        for (int i = 0; i < delta.length; i++) {
            for (int j = 0; j < delta[i].length; j++) {
                newDelta[i][j] += dxHat[i][j] * invStdDev[j] + dvar[j] * (2 * (x[i][j] - mu[j])) / delta.length + dmu[j] / delta.length;
            }
        }

        return newDelta;
    }

    public double[][] gamma() {
        return w;
    }

    public double gamma(int j) {
        return w[j][0];
    }
}

