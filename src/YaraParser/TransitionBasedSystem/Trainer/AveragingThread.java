package YaraParser.TransitionBasedSystem.Trainer;

import org.deeplearning4j.nn.graph.ComputationGraph;

import java.util.concurrent.Callable;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/8/16
 * Time: 10:47 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class AveragingThread implements Callable<Boolean> {
    int i;
    ComputationGraph net;
    ComputationGraph avgNet;
    double decay;

    public AveragingThread(int i, ComputationGraph net, ComputationGraph avgNet, double decay) {
        this.i = i;
        this.net = net;
        this.avgNet = avgNet;
        this.decay = decay;
    }

    /**
     * Computes a result, or throws an exception if unable to do so.
     *
     * @return computed result
     * @throws Exception if unable to compute a result
     */
    @Override
    public Boolean call() throws Exception {
        avgNet.getLayer(i).getParam("W").muli(decay).addi(net.getLayer(i).getParam("W").mul(1 - decay));
        avgNet.getLayer(i).getParam("b").muli(decay).addi(net.getLayer(i).getParam("b").mul(1 - decay));
        return true;
    }
}
