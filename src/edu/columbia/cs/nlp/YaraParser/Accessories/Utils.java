package edu.columbia.cs.nlp.YaraParser.Accessories;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 12:24 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Utils {
    public static final double SmallDouble = Double.MAX_VALUE * 10;

    public static double[][] clone(double[][] object) {
        if (object == null) return null;
        double[][] cloned = new double[object.length][];
        for (int i = 0; i < object.length; i++) {
            cloned[i] = clone(object[i]);
        }
        return cloned;
    }

    public static double[] clone(double[] object) {
        if (object == null) return null;
        double[] cloned = new double[object.length];
        for (int i = 0; i < object.length; i++) {
            cloned[i] = object[i];
        }
        return cloned;
    }

    public static void addInPlace(double[][] m1, double[][] m2) {
        if (m1 == null) return;
        for (int i = 0; i < m1.length; i++)
            addInPlace(m1[i], m2[i]);
    }

    public static void addInPlace(double[] m1, double[] m2) {
        if (m1 == null) return;
        for (int i = 0; i < m1.length; i++)
            m1[i] += m2[i];
    }

    public static boolean equals(double[][] o1, double[][] o2) {
        if (o1.length != o2.length)
            return false;
        for (int i = 0; i < o1.length; i++) {
            if (!equals(o1[i], o2[i]))
                return false;
        }
        return true;
    }

    public static boolean equals(double[] o1, double[] o2) {
        if (o1.length != o2.length)
            return false;
        for (int i = 0; i < o1.length; i++) {
            if (o1[i] != o2[i])
                return false;
        }
        return true;
    }

    public static boolean equals(int[][] o1, int[][] o2) {
        if (o1.length != o2.length)
            return false;
        for (int i = 0; i < o1.length; i++) {
            if (!equals(o1[i], o2[i]))
                return false;
        }
        return true;
    }

    public static boolean equals(int[] o1, int[] o2) {
        if (o1.length != o2.length)
            return false;
        for (int i = 0; i < o1.length; i++) {
            if (o1[i] != o2[i])
                return false;
        }
        return true;
    }

    public static <T> HashSet<T>[] createHashSetArray(int size) {
        HashSet<T>[] a = new HashSet[size];
        for (int i = 0; i < a.length; i++)
            a[i] = new HashSet<>();
        return a;
    }

    public static List getRandomSubset(List lst, Random random, int size) {
        ArrayList ar = new ArrayList(size);
        for (int i = 0; i < size; i++)
            ar.add(lst.get(random.nextInt(lst.size())));
        return ar;
    }

    public static double[] dot(double[][] x, double[] y) {
        assert x[0].length == y.length;

        double[] o = new double[x.length];
        for (int i = 0; i < o.length; i++) {
            for (int j = 0; j < y.length; j++) {
                o[i] += x[i][j] * y[j];
            }
        }
        return o;
    }

    public static double[] prod(double[] x, double[] y) {
        assert x.length == y.length;

        double[] o = new double[x.length];
        for (int i = 0; i < o.length; i++) {
            o[i] = x[i] * y[i];
        }
        return o;
    }

    public static double[] sum(double[] x, double[] y) {
        assert x.length == y.length;

        double[] o = new double[x.length];
        for (int i = 0; i < o.length; i++) {
            o[i] = x[i] + y[i];
        }
        return o;
    }


    public static double[] dotTranspose(double[][] x, double[] y) {
        assert x.length == y.length;

        double[] o = new double[x[0].length];
        for (int j = 0; j < y.length; j++) {
            for (int i = 0; i < o.length; i++) {
                o[i] += x[j][i] * y[j];
            }
        }
        return o;
    }
}
