package edu.columbia.cs.nlp.Tests;

import edu.columbia.cs.nlp.YaraParser.Accessories.Utils;
import org.junit.Test;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 6:30 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class UtilsTest {
    @Test
    public void testDot() {
        double[][] x = new double[][]{{1, 2}, {3, 7}, {5, 6}};
        double[] y = new double[]{1, -1};

        double[] dot = Utils.dot(x, y);
        assert dot.length == x.length;
        assert dot[0] == -1;
        assert dot[1] == -4;
        assert dot[2] == -1;
    }

    @Test
    public void TestDotTranspose() {
        double[][] x = new double[][]{{1, 3, 5}, {2, 7, 6}};
        double[] y = new double[]{1, -1};

        double[] dot = Utils.dotTranspose(x, y);
        assert dot.length == x[0].length;
        assert dot[0] == -1;
        assert dot[1] == -4;
        assert dot[2] == -1;
    }

    @Test
    public void TestProd() {
        double[] x = new double[]{1, 2, 3};
        double[] y = new double[]{-1, -2, -3};

        double[] prod = Utils.prod(x, y);

        assert prod.length == x.length;
        assert prod[0] == -1;
        assert prod[1] == -4;
        assert prod[2] == -9;
    }

    @Test
    public void TestSum() {
        double[] x = new double[]{1, 2, 3};
        double[] y = new double[]{-1, -2, -3};

        double[] prod = Utils.sum(x, y);

        assert prod.length == x.length;
        assert prod[0] == 0;
        assert prod[1] == 0;
        assert prod[2] == 0;
    }
}
