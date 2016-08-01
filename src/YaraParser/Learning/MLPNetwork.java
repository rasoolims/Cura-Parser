package YaraParser.Learning;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/21/16
 * Time: 3:39 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

import YaraParser.Accessories.Options;
import YaraParser.Structures.EmbeddingTypes;
import YaraParser.Structures.IndexMaps;

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
    final int numberOfWordEmbeddingLayers = 19;
    final int numberOfPosEmbeddingLayers = 19;
    final int numberOfLabelEmbeddingLayers = 11;
    final int numOfDependencyLabels;
    final int labelEmbeddingSize;
    final int wordEmbeddingSize;
    final int hiddenLayerSize;
    final int hiddenLayerIntSize;
    final int numOfWords;
    final int numOfPos;
    final int posEmbeddingSize;
    final int softmaxLayerSize;
    //todo make them private.
    NetworkMatrices matrices;
    double[][][] saved;

    public MLPNetwork(IndexMaps maps, Options options, ArrayList<Integer> dependencyLabels, int wDim) throws Exception {
        this.maps = maps;
        this.options = options;
        this.dependencyLabels = dependencyLabels;
        softmaxLayerSize = 2 * (dependencyLabels.size() + 1);
        numOfWords = maps.vocabSize() + 2;
        numOfDependencyLabels = maps.relSize() + 2;
        labelEmbeddingSize = 32;
        wordEmbeddingSize = wDim;
        hiddenLayerSize = options.hiddenLayer1Size;
        numOfPos = maps.posSize() + 2;
        posEmbeddingSize = 32;

        hiddenLayerIntSize =
                numberOfPosEmbeddingLayers * wDim + numberOfPosEmbeddingLayers * posEmbeddingSize + numberOfLabelEmbeddingLayers * labelEmbeddingSize;
        matrices = new NetworkMatrices(numOfWords, wDim, numOfPos, posEmbeddingSize, numOfDependencyLabels, labelEmbeddingSize, hiddenLayerSize,
                hiddenLayerIntSize, softmaxLayerSize);

        initializeLayers();
        addPretrainedWordEmbeddings(maps);
        preCompute();
    }

    public static void averageNetworks(MLPNetwork toAverageFrom, MLPNetwork averaged, double r1, double r2) {
        ArrayList<double[][]> matrices1 = toAverageFrom.matrices.getAllMatrices();
        ArrayList<double[][]> matrices2 = averaged.matrices.getAllMatrices();
        for (int m = 0; m < matrices1.size(); m++) {
            for (int i = 0; i < matrices1.get(m).length; i++) {
                for (int j = 0; j < matrices1.get(m)[i].length; j++) {
                    matrices2.get(m)[i][j] = r1 * matrices1.get(m)[i][j] + r2 * matrices2.get(m)[i][j];
                }
            }
        }

        ArrayList<double[]> vectors1 = toAverageFrom.matrices.getAllVectors();
        ArrayList<double[]> vectors2 = averaged.matrices.getAllVectors();
        for (int m = 0; m < vectors1.size(); m++) {
            for (int i = 0; i < vectors1.get(m).length; i++) {
                vectors2.get(m)[i] = r1 * vectors1.get(m)[i] + r2 * vectors2.get(m)[i];
            }
        }
    }

    private void addPretrainedWordEmbeddings(IndexMaps maps) {
        int numOfPretrained = 0;
        for (int i = 0; i < numOfWords; i++) {
            double[] embeddings = maps.embeddings(i);
            if (embeddings != null) {
                matrices.resetToPretrainedWordEmbeddings(i, embeddings);
                numOfPretrained++;
            }
        }
        System.out.println("num of pre-trained embedding " + numOfPretrained + " out of " + maps.vocabSize());
    }

    private void initializeLayers() throws Exception {
        Random random = new Random();
        for (int i = 0; i < numOfWords; i++) {
            for (int j = 0; j < wordEmbeddingSize; j++) {
                matrices.modify(EmbeddingTypes.WORD, i, j, random.nextGaussian() * 0.01);
            }
        }

        for (int i = 0; i < numOfPos; i++) {
            for (int j = 0; j < posEmbeddingSize; j++) {
                matrices.modify(EmbeddingTypes.POS, i, j, random.nextGaussian() * 0.01);
            }
        }

        for (int i = 0; i < numOfDependencyLabels; i++) {
            for (int j = 0; j < labelEmbeddingSize; j++) {
                matrices.modify(EmbeddingTypes.DEPENDENCY, i, j, random.nextGaussian() * 0.01);
            }
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            matrices.modify(EmbeddingTypes.HIDDENLAYERBIAS, i, -1, 0.2);
            for (int j = 0; j < hiddenLayerIntSize; j++) {
                matrices.modify(EmbeddingTypes.HIDDENLAYER, i, j, random.nextGaussian() * 0.01);
            }
        }

        for (int i = 0; i < softmaxLayerSize; i++) {
            for (int j = 0; j < hiddenLayerSize; j++) {
                matrices.modify(EmbeddingTypes.SOFTMAX, i, j, random.nextGaussian() * 0.01);
            }
        }
    }

    public void modify(EmbeddingTypes t, int i, int j, double change) throws Exception {
        matrices.modify(t, i, j, change);
    }

    public void preCompute() {
        final double[][] hiddenLayer = matrices.getHiddenLayer();
        final double[][] wordEmbeddings = matrices.getWordEmbedding();
        final double[][] posEmbeddings = matrices.getPosEmbedding();
        final double[][] labelEmbeddings = matrices.getLabelEmbedding();

        saved = new double[numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers + numberOfLabelEmbeddingLayers][][];
        for (int i = 0; i < numberOfWordEmbeddingLayers; i++)
            saved[i] = new double[maps.preComputeMap.size()][hiddenLayerSize];
        for (int i = numberOfWordEmbeddingLayers; i < numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers; i++)
            saved[i] = new double[numOfPos][hiddenLayerSize];
        for (int i = numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers;
             i < numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers + numberOfLabelEmbeddingLayers; i++)
            saved[i] = new double[numOfDependencyLabels][hiddenLayerSize];


        for (int id : maps.preComputeMap.keySet()) {
            int v = maps.preComputeMap.get(id);
            int offset = 0;
            for (int pos = 0; pos < numberOfWordEmbeddingLayers; pos++) {
                for (int h = 0; h < hiddenLayerSize; h++) {
                    for (int k = 0; k < wordEmbeddingSize; k++) {
                        saved[pos][v][h] += hiddenLayer[h][offset + k] * wordEmbeddings[id][k];
                    }
                }
                offset += wordEmbeddingSize;
            }
        }

        for (int id = 0; id < numOfPos; id++) {
            int offset = numberOfWordEmbeddingLayers * wordEmbeddingSize;
            for (int pos = 0; pos < numberOfPosEmbeddingLayers; pos++) {
                for (int h = 0; h < hiddenLayerSize; h++) {
                    for (int k = 0; k < posEmbeddingSize; k++) {
                        saved[pos + numberOfWordEmbeddingLayers][id][h] += hiddenLayer[h][offset + k] * posEmbeddings[id][k];
                    }
                }
                offset += posEmbeddingSize;
            }
        }

        for (int id = 0; id < numOfDependencyLabels; id++) {
            int offset = numberOfWordEmbeddingLayers * wordEmbeddingSize + numberOfPosEmbeddingLayers * posEmbeddingSize;
            for (int pos = 0; pos < numberOfLabelEmbeddingLayers; pos++) {
                for (int h = 0; h < hiddenLayerSize; h++) {
                    for (int k = 0; k < labelEmbeddingSize; k++) {
                        saved[pos + numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers][id][h] += hiddenLayer[h][offset + k] *
                                labelEmbeddings[id][k];
                    }
                }
                offset += labelEmbeddingSize;
            }
        }
    }

    public double[] output(final int[] feats) {
        final double[][] softmaxLayer = matrices.getSoftmaxLayer();
        final double[] softmaxLayerBias = matrices.getSoftmaxLayerBias();
        final double[][] hiddenLayer = matrices.getHiddenLayer();
        final double[] hiddenLayerBias = matrices.getHiddenLayerBias();
        final double[][] wordEmbeddings = matrices.getWordEmbedding();
        final double[][] posEmbeddings = matrices.getPosEmbedding();
        final double[][] labelEmbeddings = matrices.getLabelEmbedding();
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
            hidden[i] = Math.max(hidden[i], 0);
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
