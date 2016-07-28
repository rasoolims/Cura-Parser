package YaraParser.Learning;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/21/16
 * Time: 3:39 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

import YaraParser.Accessories.Options;
import YaraParser.Accessories.Pair;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.NNInfStruct;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

/**
 * Manual MLP model
 */
public class MLPNetwork implements Serializable {
    final public IndexMaps maps;
    final public Options options;
    final public ArrayList<Integer> dependencyLabels;
    //todo make them private.
    double[][] wordEmbeddings;
    double[][] posEmbeddings;
    double[][] labelEmbeddings;
    double[][] hiddenLayer;
    double[] hiddenLayerBias;
    double[][] softmaxLayer;
    double[] softmaxLayerBias;

    final int numberOfWordEmbeddingLayers = 19;
    final int numberOfPosEmbeddingLayers = 19;
    final int numberOfLabelEmbeddingLayers = 11;

    double[][][] saved;

    public MLPNetwork(IndexMaps maps, Options options, ArrayList<Integer> dependencyLabels, int wDim) {
        this.maps = maps;
        this.options = options;
        this.dependencyLabels = dependencyLabels;
        wordEmbeddings = new double[maps.vocabSize() + 2][wDim];
        posEmbeddings = new double[maps.posSize() + 2][32];
        labelEmbeddings = new double[maps.relSize() + 2][32];
        hiddenLayer = new double[options.hiddenLayer1Size][];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenLayer[i] = new double[wordEmbeddings.length * wordEmbeddings[0].length + posEmbeddings.length *
                    posEmbeddings[0].length + labelEmbeddings.length * labelEmbeddings[0].length];
        }
        hiddenLayerBias = new double[options.hiddenLayer1Size];
        softmaxLayer = new double[2 * (dependencyLabels.size() + 1)][options.hiddenLayer1Size];
        softmaxLayerBias = new double[2 * (dependencyLabels.size() + 1)];

        Random random = new Random();
        for (int i = 0; i < wordEmbeddings.length; i++) {
            for (int j = 0; j < wordEmbeddings[i].length; j++) {
                wordEmbeddings[i][j] = random.nextDouble() * 0.02 - 0.01;
            }
        }

        for (int i = 0; i < posEmbeddings.length; i++) {
            for (int j = 0; j < posEmbeddings[i].length; j++) {
                posEmbeddings[i][j] = random.nextDouble() * 0.02 - 0.01;
            }
        }

        for (int i = 0; i < labelEmbeddings.length; i++) {
            for (int j = 0; j < labelEmbeddings[i].length; j++) {
                labelEmbeddings[i][j] = random.nextDouble() * 0.02 - 0.01;
            }
        }

        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenLayerBias[i] = 0.2;
            for (int j = 0; j < hiddenLayer[i].length; j++) {
                hiddenLayer[i][j] = random.nextDouble() * 0.02 - 0.01;
            }
        }

        for (int i = 0; i < softmaxLayer.length; i++) {
            softmaxLayerBias[i] = 0;
            for (int j = 0; j < softmaxLayer[i].length; j++) {
                softmaxLayer[i][j] = random.nextDouble() * 0.02 - 0.01;
            }
        }

        for (int i = 0; i < maps.vocabSize() + 2; i++) {
            double[] embeddings = maps.embeddings(i);
            if (embeddings != null) {
                wordEmbeddings[i] = embeddings;
            }
        }
    }

    public MLPNetwork(final NNInfStruct nnInfStruct) {

        ComputationGraph net = nnInfStruct.net;
        dependencyLabels = nnInfStruct.dependencyLabels;
        options = nnInfStruct.options;
        maps = nnInfStruct.maps;

        INDArray wEArr = net.getLayer(0).getParam("W");
        wordEmbeddings = new double[wEArr.rows()][wEArr.columns()];
        for (int i = 0; i < wordEmbeddings.length; i++) {
            for (int j = 0; j < wordEmbeddings[i].length; j++)
                wordEmbeddings[i][j] = wEArr.getRow(i).getColumn(j).getDouble(0);
        }

        INDArray pEArr = net.getLayer(numberOfWordEmbeddingLayers).getParam("W");
        posEmbeddings = new double[pEArr.rows()][pEArr.columns()];
        for (int i = 0; i < posEmbeddings.length; i++) {
            for (int j = 0; j < posEmbeddings[i].length; j++)
                posEmbeddings[i][j] = pEArr.getRow(i).getColumn(j).getDouble(0);
        }

        INDArray lEArr = net.getLayer(numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers).getParam("W");
        labelEmbeddings = new double[lEArr.rows()][lEArr.columns()];
        for (int i = 0; i < labelEmbeddings.length; i++) {
            for (int j = 0; j < labelEmbeddings[i].length; j++)
                labelEmbeddings[i][j] = lEArr.getRow(i).getColumn(j).getDouble(0);
        }

        INDArray hW = net.getLayer(numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers + numberOfLabelEmbeddingLayers).getParam("W");
        hiddenLayer = new double[hW.rows()][hW.columns()];
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < hiddenLayer[i].length; j++)
                hiddenLayer[i][j] = hW.getRow(i).getColumn(j).getDouble(0);
        }

        INDArray hB = net.getLayer(numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers + numberOfLabelEmbeddingLayers).getParam("b");
        hiddenLayerBias = new double[hB.columns()];
        for (int i = 0; i < hiddenLayerBias.length; i++) {
            hiddenLayerBias[i] = hB.getColumn(i).getDouble(0);
        }

        INDArray sW = net.getLayer(numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers +
                numberOfLabelEmbeddingLayers + 1).getParam("W");
        softmaxLayer = new double[sW.rows()][sW.columns()];
        for (int i = 0; i < softmaxLayer.length; i++) {
            for (int j = 0; j < softmaxLayer[i].length; j++)
                softmaxLayer[i][j] = sW.getRow(i).getColumn(j).getDouble(0);
        }

        INDArray sB = net.getLayer(numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers +
                numberOfLabelEmbeddingLayers + 1).getParam("b");
        softmaxLayerBias = new double[sB.columns()];
        for (int i = 0; i < softmaxLayerBias.length; i++) {
            softmaxLayerBias[i] = sB.getColumn(i).getDouble(0);
        }
        preCompute();
    }

    public void preCompute() {
        saved = new double[numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers + numberOfLabelEmbeddingLayers][][];
        for (int pos = 0; pos < numberOfWordEmbeddingLayers; pos++) {
            saved[pos] = new double[maps.preComputeMap.size()][hiddenLayer[pos].length];
        }
        for (int pos = numberOfWordEmbeddingLayers; pos < numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers; pos++) {
            saved[pos] = new double[posEmbeddings.length][hiddenLayer[pos].length];
        }
        for (int pos = numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers; pos < numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers +
                numberOfLabelEmbeddingLayers; pos++) {
            saved[pos] = new double[labelEmbeddings.length][hiddenLayer[pos].length];
        }

        int wordEmbeddingSize = wordEmbeddings[0].length;
        int posEmbeddingSize = posEmbeddings[0].length;
        int labelEmbeddingSize = labelEmbeddings[0].length;
        for (int id : maps.preComputeMap.keySet()) {
            int v = maps.preComputeMap.get(id);
            int offset = 0;
            for (int pos = 0; pos < numberOfWordEmbeddingLayers; pos++) {
                for (int j = 0; j < hiddenLayer.length; j++) {
                    for (int k = 0; k < wordEmbeddingSize; k++) {
                        //todo does not work with dl4j anymore
                        saved[pos][v][j] += hiddenLayer[j][offset + k] * wordEmbeddings[id][k];
                    }
                }
                offset += wordEmbeddingSize;
            }
        }

        for (int id = 0; id < posEmbeddings.length; id++) {
            int offset = numberOfWordEmbeddingLayers * wordEmbeddingSize;
            for (int pos = 0; pos < numberOfPosEmbeddingLayers; pos++) {
                for (int j = 0; j < hiddenLayer.length; j++) {
                    for (int k = 0; k < posEmbeddingSize; k++) {
                        saved[pos + numberOfWordEmbeddingLayers][id][j] += hiddenLayer[j][offset + k] * posEmbeddings[id][k];
                    }
                }
                offset += posEmbeddingSize;
            }
        }

        for (int id = 0; id < labelEmbeddings.length; id++) {
            int offset = numberOfWordEmbeddingLayers * wordEmbeddingSize + numberOfPosEmbeddingLayers *posEmbeddingSize;
            for (int pos = 0; pos < numberOfLabelEmbeddingLayers; pos++) {
                for (int j = 0; j < hiddenLayer.length; j++) {
                    for (int k = 0; k < labelEmbeddingSize; k++) {
                        saved[pos + numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers][id][j] += hiddenLayer[j][offset + k] *
                                labelEmbeddings[id][k];
                    }
                }
                offset += labelEmbeddingSize;
            }
        }
    }

    public double[] output(final int[] feats) {
        double[] hidden = new double[hiddenLayer.length];

        int offset = 0;
        for (int j = 0; j < feats.length; j++) {
            int tok = feats[j];
            double[][] embedding;
            if (j < numberOfWordEmbeddingLayers)
                embedding = wordEmbeddings;
            else if (j < numberOfWordEmbeddingLayers + numberOfWordEmbeddingLayers)
                embedding = posEmbeddings;
            else embedding = labelEmbeddings;

            if (saved != null && (j >= numberOfWordEmbeddingLayers || maps.preComputeMap.containsKey(tok))) {
                int id = tok;
                if (j < numberOfWordEmbeddingLayers)
                    id = maps.preComputeMap.get(tok);
                for (int i = 0; i < hidden.length; ++i) {
                    hidden[i] += saved[j][id][i];
                }
            } else {
                for (int i = 0; i < hidden.length; i++) {
                    for (int k = 0; k < embedding[0].length; k++) {
                        hidden[i] += hiddenLayer[i][offset + k] * embedding[tok][k];
                    }
                }
            }
            offset += embedding[0].length;
        }

        for (int i = 0; i < hidden.length; i++) {
            hidden[i] += hiddenLayerBias[i];
            //relu
            hidden[i] = Math.max(hidden[i],0);
        }

        double[] probs = new double[softmaxLayerBias.length];
        double sum = 0;
        for (int i = 0; i < probs.length; i++) {
            for (int j = 0; j < hidden.length; j++) {
                probs[i] += softmaxLayer[i][j] * hidden[j];
            }
            probs[i] += softmaxLayerBias[i];
            probs[i] = Math.exp(probs[i]);
            sum += probs[i];
        }

        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
            probs[i] = Math.log(probs[i]);
        }
        return probs;
    }


}
