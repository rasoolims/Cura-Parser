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
    final public ArrayList<Integer> depLabels;
    final int numOfWordLayers = 19;
    final int numOfPosLayers = 19;
    final int numOfDepLayers = 11;
    final int numOfDepLabels;
    final int labelEmbedDim;
    final int wordEmbedDim;
    final int hiddenLayerDim;
    final int hiddenLayerIntDim;
    final int numOfWords;
    final int numOfPos;
    final int posEmbeddingDim;
    final int softmaxLayerDim;
    //todo make them private.
    NetworkMatrices matrices;
    double[][][] saved;

    public MLPNetwork(IndexMaps maps, Options options, ArrayList<Integer> depLabels, int wDim) throws Exception {
        this.maps = maps;
        this.options = options;
        this.depLabels = depLabels;
        softmaxLayerDim = 2 * (depLabels.size() + 1);
        numOfWords = maps.vocabSize() + 2;
        numOfDepLabels = maps.relSize() + 2;
        labelEmbedDim = 32;
        wordEmbedDim = wDim;
        hiddenLayerDim = options.hiddenLayer1Size;
        numOfPos = maps.posSize() + 2;
        posEmbeddingDim = 32;

        hiddenLayerIntDim = numOfPosLayers * wDim + numOfPosLayers * posEmbeddingDim + numOfDepLayers * labelEmbedDim;
        matrices = new NetworkMatrices(numOfWords, wDim, numOfPos, posEmbeddingDim, numOfDepLabels, labelEmbedDim, hiddenLayerDim,
                hiddenLayerIntDim, softmaxLayerDim);
        initializeLayers();
        addPretrainedWordEmbeddings(maps);
        preCompute();
    }


    /**
     * This is only for testing.
     *
     * @param wDim
     * @throws Exception
     */
    public MLPNetwork(int hiddenLayer1Size, int wDim, int numOfWords, int numOfPos, int numOfDepLabels, int pDim, int lDim) throws Exception {
        this.maps = null;
        this.options = null;
        this.depLabels = new ArrayList<>(numOfDepLabels);
        for (int i = 0; i < numOfDepLabels; i++)
            this.depLabels.add(i);
        softmaxLayerDim = 2 * (depLabels.size() + 1);
        this.numOfWords = numOfWords;
        this.numOfDepLabels = numOfDepLabels;
        labelEmbedDim = lDim;
        wordEmbedDim = wDim;
        this.hiddenLayerDim = hiddenLayer1Size;
        this.numOfPos = numOfPos;
        posEmbeddingDim = pDim;

        hiddenLayerIntDim =
                numOfPosLayers * wDim + numOfPosLayers * posEmbeddingDim + numOfDepLayers * labelEmbedDim;
        matrices = new NetworkMatrices(numOfWords, wDim, numOfPos, posEmbeddingDim, numOfDepLabels, labelEmbedDim, hiddenLayerDim,
                hiddenLayerIntDim, softmaxLayerDim);

        initializeLayers();
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
        double wEmbedStdDev = Math.pow(1.0 / wordEmbedDim, 0.5);
        for (int i = 0; i < numOfWords; i++) {
            for (int j = 0; j < wordEmbedDim; j++) {
                matrices.modify(EmbeddingTypes.WORD, i, j, random.nextGaussian() * wEmbedStdDev);
            }
        }

        for (int i = 0; i < numOfPos; i++) {
            for (int j = 0; j < posEmbeddingDim; j++) {
                matrices.modify(EmbeddingTypes.POS, i, j, random.nextGaussian() * 0.01);
            }
        }

        for (int i = 0; i < numOfDepLabels; i++) {
            for (int j = 0; j < labelEmbedDim; j++) {
                matrices.modify(EmbeddingTypes.DEPENDENCY, i, j, random.nextGaussian() * 0.01);
            }
        }

        for (int i = 0; i < hiddenLayerDim; i++) {
            matrices.modify(EmbeddingTypes.HIDDENLAYERBIAS, i, -1, 0.2);
            for (int j = 0; j < hiddenLayerIntDim; j++) {
                matrices.modify(EmbeddingTypes.HIDDENLAYER, i, j, random.nextGaussian() * 0.01);
            }
        }

        for (int i = 0; i < softmaxLayerDim; i++) {
            for (int j = 0; j < hiddenLayerDim; j++) {
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

        saved = new double[numOfWordLayers + numOfPosLayers + numOfDepLayers][][];
        for (int i = 0; i < numOfWordLayers; i++)
            saved[i] = new double[maps.preComputeMap.size()][hiddenLayerDim];
        for (int i = numOfWordLayers; i < numOfWordLayers + numOfPosLayers; i++)
            saved[i] = new double[numOfPos][hiddenLayerDim];
        for (int i = numOfWordLayers + numOfPosLayers; i < numOfWordLayers + numOfPosLayers + numOfDepLayers; i++)
            saved[i] = new double[numOfDepLabels][hiddenLayerDim];

        for (int tok : maps.preComputeMap.keySet()) {
            int id = maps.preComputeMap.get(tok);
            int offset = 0;
            for (int pos = 0; pos < numOfWordLayers; pos++) {
                for (int h = 0; h < hiddenLayerDim; h++) {
                    for (int k = 0; k < wordEmbedDim; k++) {
                        saved[pos][id][h] += hiddenLayer[h][offset + k] * wordEmbeddings[tok][k];
                    }
                }
                offset += wordEmbedDim;
            }
        }

        for (int tok = 0; tok < numOfPos; tok++) {
            int offset = numOfWordLayers * wordEmbedDim;
            int indOffset = numOfWordLayers;
            for (int pos = 0; pos < numOfPosLayers; pos++) {
                for (int h = 0; h < hiddenLayerDim; h++) {
                    for (int k = 0; k < posEmbeddingDim; k++) {
                        saved[pos + indOffset][tok][h] += hiddenLayer[h][offset + k] * posEmbeddings[tok][k];
                    }
                }
                offset += posEmbeddingDim;
            }
        }

        for (int tok = 0; tok < numOfDepLabels; tok++) {
            int offset = numOfWordLayers * wordEmbedDim + numOfPosLayers * posEmbeddingDim;
            int indOffset = numOfWordLayers + numOfPosLayers;
            for (int pos = 0; pos < numOfDepLayers; pos++) {
                for (int h = 0; h < hiddenLayerDim; h++) {
                    for (int k = 0; k < labelEmbedDim; k++) {
                        saved[pos + indOffset][tok][h] += hiddenLayer[h][offset + k] * labelEmbeddings[tok][k];
                    }
                }
                offset += labelEmbedDim;
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
            if (j < numOfWordLayers)
                embedding = wordEmbeddings;
            else if (j < numOfWordLayers + numOfPosLayers)
                embedding = posEmbeddings;
            else embedding = labelEmbeddings;

            if (saved != null && (j >= numOfWordLayers || maps.preComputeMap.containsKey(tok))) {
                int id = tok;
                if (j < numOfWordLayers)
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


    public IndexMaps getMaps() {
        return maps;
    }

    public Options getOptions() {
        return options;
    }

    public ArrayList<Integer> getDepLabels() {
        return depLabels;
    }

    public int getNumOfWordLayers() {
        return numOfWordLayers;
    }

    public int getNumOfPosLayers() {
        return numOfPosLayers;
    }

    public int getNumOfDepLayers() {
        return numOfDepLayers;
    }

    public int getNumOfDepLabels() {
        return numOfDepLabels;
    }

    public int getLabelEmbedDim() {
        return labelEmbedDim;
    }

    public int getWordEmbedDim() {
        return wordEmbedDim;
    }

    public int getHiddenLayerDim() {
        return hiddenLayerDim;
    }

    public int getHiddenLayerIntDim() {
        return hiddenLayerIntDim;
    }

    public int getNumOfWords() {
        return numOfWords;
    }

    public int getNumOfPos() {
        return numOfPos;
    }

    public int getPosEmbeddingDim() {
        return posEmbeddingDim;
    }

    public int getSoftmaxLayerDim() {
        return softmaxLayerDim;
    }

    public NetworkMatrices getMatrices() {
        return matrices;
    }
}
