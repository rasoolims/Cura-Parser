package edu.columbia.cs.nlp.CuraParser.Accessories;

import java.text.SimpleDateFormat;
import java.util.*;

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
        System.arraycopy(object, 0, cloned, 0, object.length);
        return cloned;
    }

    /**
     * Sum in place
     *
     * @param m1 this value will be affected.
     * @param m2
     */
    public static void sumi(double[][] m1, double[][] m2) {
        if (m1 == null || m2 == null) return;
        for (int i = 0; i < m1.length; i++) sumi(m1[i], m2[i]);
    }

    /**
     * Sum in place
     *
     * @param v1 this value will be affected.
     * @param v2
     */
    public static void sumi(double[] v1, double[] v2) {
        if (v1 == null || v2 == null) return;
        for (int i = 0; i < v1.length; i++) v1[i] += v2[i];
    }

    public static void sumi(double[] v1, double[] v2, HashSet<Integer> v1ToUSe) {
        if (v1 == null || v2 == null) return;
        for (int i : v1ToUSe) v1[i] += v2[i];
    }

    public static int argmax(double[] o) {
        int argmax = 0;
        for (int i = 1; i < o.length; i++) {
            if (o[i] > o[argmax])
                argmax = i;
        }
        return argmax;
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

    public static void avgMatrices(double[][] m, double[][] toAvgM, double r1, double r2) {
        if (m == null) return;
        for (int i = 0; i < m.length; i++) {
            avgVectors(m[i], toAvgM[i], r1, r2);
        }
    }

    public static void avgVectors(double[] v1, double[] toAvgV, double r1, double r2) {
        if (v1 == null) return;
        for (int j = 0; j < v1.length; j++) {
            toAvgV[j] = r1 * v1[j] + r2 * toAvgV[j];
        }
    }

    public static double[] dot4Output(double[][] x, double[] y, double[] labels) {
        assert x[0].length == y.length;

        double[] o = new double[x.length];
        for (int i = 0; i < o.length; i++) {
            for (int j = 0; j < y.length; j++) {
                if (labels[i] >= 0)
                    o[i] += x[i][j] * y[j];
            }
        }
        return o;
    }

    public static double[] dot(double[][] x, double[] y, HashSet<Integer> xToUse) {
        assert x[0].length == y.length;

        double[] o = new double[x.length];
        for (int i : xToUse) {
            for (int j = 0; j < y.length; j++) {
                o[i] += x[i][j] * y[j];
            }
        }
        return o;
    }

    public static double[] dot(double[][] x, double[] y, HashSet<Integer> xToUse, HashSet<Integer> yToUse) {
        assert x[0].length == y.length;

        double[] o = new double[x.length];
        for (int i : xToUse) {
            for (int j : yToUse) {
                o[i] += x[i][j] * y[j];
            }
        }
        return o;
    }


    public static double[][] dotTranspose(double[] x, double[] y) {
        double[][] o = new double[x.length][y.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < y.length; j++) {
                o[i][j] = x[i] * y[j];
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
        if (y == null) return Utils.clone(x);
        assert x.length == y.length;

        double[] o = new double[x.length];
        for (int i = 0; i < o.length; i++) {
            o[i] = x[i] + y[i];
        }
        return o;
    }

    public static double[] sum4Output(double[] x, double[] y, double[] labels) {
        if (y == null) return x;
        assert x.length == y.length;

        double[] o = new double[x.length];
        for (int i = 0; i < o.length; i++) {
            if (labels[i] >= 0)
                o[i] = x[i] + y[i];
        }
        return o;
    }

    public static boolean allZero(double[] x) {
        if (x == null) return true;
        for (double v : x)
            if (v != 0.0) return false;
        return true;
    }

    public static boolean allZero(double[][] x) {
        if (x == null) return true;
        for (double[] v : x)
            if (!allZero(v)) return false;
        return true;
    }

    public static double[] sum(double[] x, double[] y, HashSet<Integer> xToUse) {
        if (y == null) return Utils.clone(x);
        assert x.length == y.length;

        double[] o = new double[x.length];
        for (int i : xToUse) {
            o[i] = x[i] + y[i];
        }
        return o;
    }

    public static double[] dotTranspose(double[][] x, double[] y) {
        assert x.length == y.length;

        double[] o = new double[x[0].length];
        for (int j = 0; j < y.length; j++) {
            if (y[j] != 0) {
                for (int i = 0; i < o.length; i++) {
                    o[i] += x[j][i] * y[j];
                }
            }
        }
        return o;
    }

    public static void normalize(double[] e) {
        double n = 0;
        for (double v : e) n += v * v;
        n = Math.sqrt(n);
        if (n > 0)
            for (int i = 0; i < e.length; i++) e[i] /= n;
    }

    public static String timeStamp() {
        return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date());
    }

    public static boolean isNumeric(String str) {
        try {
            double d = Double.parseDouble(str);
        } catch (NumberFormatException nfe) {
            return false;
        }
        return true;
    }
}
