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
}
