package edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.Layers;

import edu.columbia.cs.nlp.YaraParser.Accessories.Utils;
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
    protected Activation activation;

    protected double[][] w;
    protected double[] b;
    protected boolean useBias;

    public Layer(Activation activation, int nIn, int nOut, Initializer initializer) {
        this(activation, nIn, nOut, initializer, new FixInit(0));
    }

    public Layer(Activation activation, int nIn, int nOut, Initializer initializer, Initializer biasInitializer) {
        this(activation, nIn, nOut, initializer, biasInitializer, true);
    }

    public Layer(Activation activation, int nIn, int nOut, Initializer initializer, Initializer biasInitializer, boolean useBias) {
        this.activation = activation;
        this.useBias = useBias;
        w = new double[nOut][nIn];
        initializer.init(w);

        if (useBias) {
            b = new double[nOut];
            biasInitializer.init(b);
        }
    }

    public double[] forward(double[] i) {
        assert i.length == w[0].length;
        if (useBias)
            return activation.activate(Utils.sum(Utils.dot(w, i), b));
        else
            return activation.activate(Utils.dot(w, i));
    }

    public double[] backward(double[] delta, double[][] nextW, double[] hInput, double[][] wG, double[] bG, double[] prevH) {
        assert delta.length == nextW.length;
        assert nextW[0].length == w.length;
        assert hInput.length == w.length;
        assert prevH.length == w[0].length;

        double[] dL_dH = Utils.dot(nextW, delta);
        double[] newDelta = activation.gradient(hInput, dL_dH);
        if (useBias) Utils.sumi(bG, newDelta);
        Utils.sumi(wG, Utils.dotTranspose(newDelta, prevH));

        return newDelta;
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

    public int nOut() {
        return w.length;
    }

    public int nIn() {
        return w[0].length;
    }

    public double w(int i, int j) {
        return w[i][j];
    }

    public double b(int i) {
        return b[i];
    }
}
