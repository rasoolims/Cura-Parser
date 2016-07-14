package YaraParser.Structures;

import YaraParser.Accessories.Options;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;

import java.io.IOException;
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
    public final String netPath;
    public ComputationGraph net;
    public final int dependencySize;
    public final IndexMaps maps;
    public final ArrayList<Integer> dependencyLabels;
    public final Options options;

    public NNInfStruct(String netPath, int dependencySize, IndexMaps maps, ArrayList<Integer> dependencyLabels,
                       Options options) throws IOException {
        this.netPath = netPath;
        this.dependencySize = dependencySize;
        this.maps = maps;
        this.dependencyLabels = dependencyLabels;
        this.options = options;
        this.net = null;
    }

    public void loadModel() throws  IOException{
        this.net = ModelSerializer.restoreComputationGraph(netPath);
    }
}
