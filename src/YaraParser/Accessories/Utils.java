package YaraParser.Accessories;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 12:24 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Utils {
    public static double[][] clone(double[][] object) {
        double[][] cloned = new double[object.length][];
        for (int i = 0; i < object.length; i++) {
            cloned[i] = clone(object[i]);
        }
        return cloned;
    }

    public static double[] clone(double[] object) {
        double[] cloned = new double[object.length];
        for (int i = 0; i < object.length; i++) {
            cloned[i] = object[i];
        }
        return cloned;
    }

    public static void addInPlace(double[][] m1, double[][] m2) {
        for (int i = 0; i < m1.length; i++)
            for (int j = 0; j < m1[0].length; j++)
                m1[i][j] += m2[i][j];
    }

    public static void addInPlace(double[] m1, double[] m2) {
        for (int i = 0; i < m1.length; i++)
            m1[i] += m2[i];
    }
}
