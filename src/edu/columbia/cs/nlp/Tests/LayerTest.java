package edu.columbia.cs.nlp.Tests;

import edu.columbia.cs.nlp.CuraParser.Learning.Activation.Relu;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.Layer;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.FixInit;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.Initializer;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.NormalInit;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.ReluInit;
import org.junit.Test;

import java.util.Random;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 4:05 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class LayerTest {
    @Test
    public void testIO() {
        int[] inSizes = new int[]{5, 10, 15};
        int[] outSizes = new int[]{5, 10, 15};
        Random random = new Random();

        for (int nIn : inSizes) {
            for (int nOut : outSizes) {
                // make small bias to make sure everything is fine
                Layer layer = new Layer(new Relu(), nIn, nOut, new ReluInit(random, nIn, nOut), new FixInit(0.01));

                Initializer normalInit = new NormalInit(random, nIn);
                double[] input = new double[nIn];
                normalInit.init(input);

                double[] output = new double[nOut];
                for (int i = 0; i < nOut; i++) {
                    for (int j = 0; j < nIn; j++) {
                        output[i] += layer.getW()[i][j] * input[j];
                    }
                    output[i] = Math.max(0, output[i] + layer.getB()[i]);
                }

                double[] o = layer.activate(layer.forward(input), false);
                for (int i = 0; i < output.length; i++) {
                    System.out.println(output[i] + "\t" + o[i]);
                    assert output[i] - o[i] <= 1e-16;
                }

            }
        }
    }

    @Test
    public void testMath() {
        int nIn = 2;
        int nOut = 3;

        // make small bias to make sure everything is fine
        Layer layer = new Layer(new Relu(), nIn, nOut, new FixInit(0), new FixInit(0));

        double[][] w = layer.getW();
        w[0][0] = 1;
        w[0][1] = -1;
        w[1][0] = 2;
        w[1][1] = 3;
        w[2][0] = -5;
        w[2][1] = 2;
        double[] b = layer.getB();
        b[0] = -1;
        b[1] = .2;
        b[2] = 0;

        double[] input = new double[]{2, 4};

        double[] o = layer.activate(layer.forward(input), false);
        double[] expectedOutput = new double[]{0, 16.2, 0};
        for (int i = 0; i < expectedOutput.length; i++) {
            System.out.println(expectedOutput[i] + "\t" + o[i]);
            assert expectedOutput[i] - o[i] <= 1e-16;
        }
    }
}
