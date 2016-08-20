package edu.columbia.cs.nlp.Tests;

import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import org.junit.Test;

import java.util.HashSet;

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
    public void testDotDropout() {
        double[][] x = new double[][]{{1, 2}, {3, 7}, {5, 6}};
        double[] y = new double[]{1, -1};
        HashSet<Integer> xToUse = new HashSet<>();
        xToUse.add(0);
        xToUse.add(2);

        double[] dot = Utils.dot(x, y, xToUse);
        assert dot.length == x.length;
        assert dot[0] == -1;
        assert dot[1] == 0;
        assert dot[2] == -1;

        HashSet<Integer> yToUse = new HashSet<>();
        yToUse.add(0);

        dot = Utils.dot(x, y, xToUse, yToUse);
        assert dot.length == x.length;
        assert dot[0] == 1;
        assert dot[1] == 0;
        assert dot[2] == 5;
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
    public void testDotTransposeVectors() {
        double[] x = new double[]{2, 4, 7};
        double[] y = new double[]{1, -3};

        double[][] dot = Utils.dotTranspose(x, y);
        assert dot.length == x.length;
        assert dot[0].length == y.length;
        assert dot[0][0] == 2;
        assert dot[0][1] == -6;
        assert dot[1][0] == 4;
        assert dot[1][1] == -12;
        assert dot[2][0] == 7;
        assert dot[2][1] == -21;
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

    @Test
    public void TestSumi() {
        double[] x = new double[]{1, 2, 3};
        double[] y = new double[]{-1, -2, -3};
        Utils.sumi(x, y);
        assert x[0] == 0;
        assert x[1] == 0;
        assert x[2] == 0;
    }
}
