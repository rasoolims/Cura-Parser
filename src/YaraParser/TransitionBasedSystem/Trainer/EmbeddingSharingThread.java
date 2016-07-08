package YaraParser.TransitionBasedSystem.Trainer;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.concurrent.Callable;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/8/16
 * Time: 10:21 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class EmbeddingSharingThread implements Callable<Boolean>{
    INDArray wArrBefore;
    INDArray bArrBefore;
    ComputationGraph net;
    int s,e;

    public EmbeddingSharingThread(INDArray wArrBefore, INDArray bArrBefore, ComputationGraph net, int s, int e) {
        this.wArrBefore = wArrBefore;
        this.bArrBefore = bArrBefore;
        this.net = net;
        this.s = s;
        this.e = e;
    }

    /**
     * Computes a result, or throws an exception if unable to do so.
     *
     * @return computed result
     * @throws Exception if unable to compute a result
     */
    @Override
    public Boolean call() throws Exception {
        INDArray wArr = net.getLayer(s).getParam("W");
        INDArray bArr = net.getLayer(s).getParam("b");
        for (int i = s + 1; i < e; i++) {
            wArr.addi(net.getLayer(i).getParam("W"));
            bArr.addi(net.getLayer(i).getParam("b"));
        }
        wArr = wArr.subi(wArrBefore);
        bArr = bArr.subi(bArrBefore);
        for (int i = s; i < e; i++) {
            net.getLayer(i).setParam("W", wArr);
            net.getLayer(i).setParam("b", bArr);
        }
        return true;
    }
}
