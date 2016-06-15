package YaraParser.Structures;

import YaraParser.Accessories.Options;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.Index;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 6/15/16
 * Time: 11:19 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class NNInfStruct implements Serializable {
    public ComputationGraph net;
    public int dependencySize;
    public IndexMaps maps;
    public ArrayList<Integer> dependencyLabels;
    public Options options;

    public NNInfStruct(ComputationGraph net, int dependencySize, IndexMaps maps, ArrayList<Integer> dependencyLabels, Options options) {
        this.net = net;
        this.dependencySize = dependencySize;
        this.maps = maps;
        this.dependencyLabels = dependencyLabels;
        this.options = options;
    }
}
