package edu.columbia.cs.nlp.Tests;

import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.BatchNormalizationLayer;
import org.junit.Test;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/30/16
 * Time: 10:58 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class BathNormalizationTest {
    @Test
    public void testForward() {
        BatchNormalizationLayer layer = new BatchNormalizationLayer(10, 1e-6);

        double[][] input = new double[20][10];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[i].length; j++) {
                input[i][j] = 10;
            }
        }

        double[][] f = layer.forward(input);
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[i].length; j++) {
                assert f[i][j] == 0;
            }
        }
    }
}
