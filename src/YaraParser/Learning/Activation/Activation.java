package YaraParser.Learning.Activation;

import java.io.Serializable;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/7/16
 * Time: 1:50 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public abstract class Activation implements Serializable {
    public Activation() {
    }

    public abstract double activate(double value);

    public abstract double gradient(double value, double gradient);
}
