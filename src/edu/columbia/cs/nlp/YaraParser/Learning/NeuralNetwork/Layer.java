package edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork;

import edu.columbia.cs.nlp.YaraParser.Learning.Activation.Activation;
import edu.columbia.cs.nlp.YaraParser.Learning.WeightInit.FixInit;
import edu.columbia.cs.nlp.YaraParser.Learning.WeightInit.Initializer;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 3:43 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Layer {
    Activation activation;

    double[][] w;
    double[] b;
    boolean useBias;

    public Layer(Activation activation, int nIn, int nOut, Initializer initializer) {
        this(activation, nIn, nOut, initializer, new FixInit(0));
    }

    public Layer(Activation activation, int nIn, int nOut, Initializer initializer, Initializer biasInitializer) {
        this(activation, nIn, nOut, initializer, biasInitializer, true);
    }

    public Layer(Activation activation, int nIn, int nOut, Initializer initializer, Initializer biasInitializer, boolean useBias) {
        this.activation = activation;
        this.useBias = useBias;
        b = new double[nOut];
        w = new double[nOut][nIn];

        initializer.init(w);
        if (useBias)
            biasInitializer.init(b);
    }

    public double[] forward(double[] i) {
        assert i.length == w[0].length;

        double[] o = new double[b.length];

        for (int j = 0; j < w.length; j++) {
            for (int k = 0; k < w[j].length; k++)
                o[j] += w[j][k] * i[k];
            o[j] = activation.activate(o[j] + b[j]);
        }
        return o;
    }

    public void modifyW(int i, int j, double change) {
        w[i][j] += change;
    }

    public void modifyb(int i, double change) {
        b[i] += change;
    }

    public final double[][] getW() {
        return w;
    }

    public final double[] getB() {
        return b;
    }
}
