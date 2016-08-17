package edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/21/16
 * Time: 3:39 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

import edu.columbia.cs.nlp.YaraParser.Accessories.Options;
import edu.columbia.cs.nlp.YaraParser.Learning.Activation.Activation;
import edu.columbia.cs.nlp.YaraParser.Learning.Activation.Cubic;
import edu.columbia.cs.nlp.YaraParser.Learning.Activation.Enums.ActivationType;
import edu.columbia.cs.nlp.YaraParser.Learning.Activation.Relu;
import edu.columbia.cs.nlp.YaraParser.Structures.Enums.EmbeddingTypes;
import edu.columbia.cs.nlp.YaraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Enums.ParserType;

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
    public final Activation activation;
    public final ActivationType activationType;
    final int numDepLabels;
    final int depEmbedDim;
    final int wordEmbedDim;
    final int hiddenLayerDim;
    final int hiddenLayerIntDim;
    final int secondHiddenLayerDim;
    final int numWords;
    final int numPos;
    final int posEmbedDim;
    final int softmaxLayerDim;
    //todo make them private.
    NetworkMatrices matrices;
    double[][][] saved;
    private int numWordLayers;
    private int numPosLayers;
    private int numDepLayers;

    public MLPNetwork(IndexMaps maps, Options options, ArrayList<Integer> depLabels, int wDim, int pDim, int lDim, ParserType parserType)
            throws Exception {
        if (parserType == ParserType.ArcEager) {
            numWordLayers = 22;
            numPosLayers = 22;
            numDepLayers = 11;
        } else {
            numWordLayers = 20;
            numPosLayers = 20;
            numDepLayers = 12;
        }
        this.maps = maps;
        this.options = options;
        this.depLabels = depLabels;
        softmaxLayerDim = 2 * (depLabels.size() + 1);
        numWords = maps.vocabSize() + 2;
        numDepLabels = maps.relSize() + 2;
        depEmbedDim = lDim;
        wordEmbedDim = wDim;
        hiddenLayerDim = options.networkProperties.hiddenLayer1Size;
        secondHiddenLayerDim = options.networkProperties.hiddenLayer2Size;
        numPos = maps.posSize() + 2;
        posEmbedDim = pDim;

        this.activationType = options.networkProperties.activationType;
        activation = activationType == ActivationType.RELU ? new Relu() : new Cubic();

        hiddenLayerIntDim = numPosLayers * wDim + numPosLayers * posEmbedDim + numDepLayers * depEmbedDim;
        matrices = new NetworkMatrices(numWords, wDim, numPos, posEmbedDim, numDepLabels, depEmbedDim, hiddenLayerDim,
                hiddenLayerIntDim, secondHiddenLayerDim, softmaxLayerDim);
        initializeLayers(activationType);
        addPretrainedWordEmbeddings(maps);
        preCompute();
    }

    public MLPNetwork(IndexMaps maps, Options options, ArrayList<Integer> depLabels, int numDepLabels, int depEmbedDim, int wordEmbedDim, int
            hiddenLayerDim, int hiddenLayerIntDim, int secondHiddenLayerDim, int numWords, int numPos, int posEmbedDim, int softmaxLayerDim,
                      NetworkMatrices matrices, double[][][] saved, ActivationType activationType, int numWordLayers, int numPosLayers, int
                              numDepLayers) {
        this.maps = maps;
        this.options = options;
        this.depLabels = depLabels;
        this.numDepLabels = numDepLabels;
        this.depEmbedDim = depEmbedDim;
        this.wordEmbedDim = wordEmbedDim;
        this.hiddenLayerDim = hiddenLayerDim;
        this.hiddenLayerIntDim = hiddenLayerIntDim;
        this.secondHiddenLayerDim = secondHiddenLayerDim;
        this.numWords = numWords;
        this.numPos = numPos;
        this.posEmbedDim = posEmbedDim;
        this.softmaxLayerDim = softmaxLayerDim;
        this.matrices = matrices;
        this.saved = saved;
        this.activationType = activationType;
        activation = activationType == ActivationType.RELU ? new Relu() : new Cubic();
        this.numWordLayers = numWordLayers;
        this.numPosLayers = numPosLayers;
        this.numDepLayers = numDepLayers;
    }

    public static void averageNetworks(final MLPNetwork toAverageFrom, MLPNetwork averaged, final double r1, final double r2) {
        ArrayList<double[][]> matrices1 = toAverageFrom.matrices.getAllMatrices();
        ArrayList<double[][]> matrices2 = averaged.matrices.getAllMatrices();
        for (int m = 0; m < matrices1.size(); m++) {
            if(matrices1.get(m)==null) continue;
            for (int i = 0; i < matrices1.get(m).length; i++) {
                for (int j = 0; j < matrices1.get(m)[i].length; j++) {
                    matrices2.get(m)[i][j] = r1 * matrices1.get(m)[i][j] + r2 * matrices2.get(m)[i][j];
                }
            }
        }

        ArrayList<double[]> vectors1 = toAverageFrom.matrices.getAllVectors();
        ArrayList<double[]> vectors2 = averaged.matrices.getAllVectors();
        for (int m = 0; m < vectors1.size(); m++) {
            if(vectors1.get(m)==null) continue;
            for (int i = 0; i < vectors1.get(m).length; i++) {
                vectors2.get(m)[i] = r1 * vectors1.get(m)[i] + r2 * vectors2.get(m)[i];
            }
        }
    }

    private void addPretrainedWordEmbeddings(IndexMaps maps) {
        int numOfPretrained = 0;
        for (int i = 0; i < numWords; i++) {
            double[] embeddings = maps.embeddings(i);
            if (embeddings != null) {
                matrices.resetToPretrainedWordEmbeddings(i, embeddings);
                numOfPretrained++;
            }
        }
        System.out.println("num of pre-trained embedding " + numOfPretrained + " out of " + maps.vocabSize());
    }

    private void initializeLayers(ActivationType activationType) throws Exception {
        Random random = new Random();
        double stdDev = 1e-2;
        double reluBiasInit = 0.2;

        for (int i = 0; i < numWords; i++) {
            for (int j = 0; j < wordEmbedDim; j++) {
                matrices.modify(EmbeddingTypes.WORD, i, j, random.nextGaussian() * 1.0 / Math.sqrt(wordEmbedDim));
            }
        }

        for (int i = 0; i < numPos; i++) {
            if (i != IndexMaps.UnknownIndex) {
                for (int j = 0; j < posEmbedDim; j++) {
                    matrices.modify(EmbeddingTypes.POS, i, j, random.nextGaussian() * 1.0 / Math.sqrt(posEmbedDim));
                }
            }
        }

        for (int i = 0; i < numDepLabels; i++) {
            if (i != maps.labelUnkIndex) {
                for (int j = 0; j < depEmbedDim; j++) {
                    matrices.modify(EmbeddingTypes.DEPENDENCY, i, j, random.nextGaussian() * 1.0 / Math.sqrt(depEmbedDim));
                }
            }
        }

        for (int i = 0; i < hiddenLayerDim; i++) {
            if (activationType == ActivationType.RELU)
                matrices.modify(EmbeddingTypes.HIDDENLAYERBIAS, i, -1, reluBiasInit);
            else
                matrices.modify(EmbeddingTypes.HIDDENLAYERBIAS, i, -1, random.nextGaussian() * stdDev);

            for (int j = 0; j < hiddenLayerIntDim; j++) {
                matrices.modify(EmbeddingTypes.HIDDENLAYER, i, j, random.nextGaussian() * stdDev);
            }
        }

        for (int i = 0; i < softmaxLayerDim; i++) {
            for (int j = 0; j < hiddenLayerDim; j++) {
                matrices.modify(EmbeddingTypes.SOFTMAX, i, j, random.nextGaussian() * stdDev);
            }
        }

        if(secondHiddenLayerDim>0){
            for (int i = 0; i < secondHiddenLayerDim; i++) {
                if (activationType == ActivationType.RELU)
                    matrices.modify(EmbeddingTypes.SECONDHIDDENLAYERBIAS, i, -1, reluBiasInit);
                else
                    matrices.modify(EmbeddingTypes.SECONDHIDDENLAYERBIAS, i, -1, random.nextGaussian() * stdDev);

                for (int j = 0; j < hiddenLayerDim; j++) {
                    matrices.modify(EmbeddingTypes.SECONDHIDDENLAYER, i, j, random.nextGaussian() * stdDev);
                }
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

        saved = new double[numWordLayers + numPosLayers + numDepLayers][][];
        for (int i = 0; i < numWordLayers; i++)
            saved[i] = new double[maps.preComputeMap[i].size()][hiddenLayerDim];
        for (int i = numWordLayers; i < numWordLayers + numPosLayers; i++)
            saved[i] = new double[numPos][hiddenLayerDim];
        for (int i = numWordLayers + numPosLayers; i < numWordLayers + numPosLayers + numDepLayers; i++)
            saved[i] = new double[numDepLabels][hiddenLayerDim];

        int offset = 0;
        for (int pos = 0; pos < numWordLayers; pos++) {
            for (int tok : maps.preComputeMap[pos].keySet()) {
                int id = maps.preComputeMap[pos].get(tok);
                for (int h = 0; h < hiddenLayerDim; h++) {
                    for (int k = 0; k < wordEmbedDim; k++) {
                        saved[pos][id][h] += hiddenLayer[h][offset + k] * wordEmbeddings[tok][k];
                    }
                }
            }
            offset += wordEmbedDim;
        }

        for (int pos = 0; pos < numPosLayers; pos++) {
            for (int tok = 0; tok < numPos; tok++) {
                int indOffset = numWordLayers;
                for (int h = 0; h < hiddenLayerDim; h++) {
                    for (int k = 0; k < posEmbedDim; k++) {
                        saved[pos + indOffset][tok][h] += hiddenLayer[h][offset + k] * posEmbeddings[tok][k];
                    }
                }
            }
            offset += posEmbedDim;
        }

        for (int pos = 0; pos < numDepLayers; pos++) {
            for (int tok = 0; tok < numDepLabels; tok++) {
                int indOffset = numWordLayers + numPosLayers;
                for (int h = 0; h < hiddenLayerDim; h++) {
                    for (int k = 0; k < depEmbedDim; k++) {
                        saved[pos + indOffset][tok][h] += hiddenLayer[h][offset + k] * labelEmbeddings[tok][k];
                    }
                }
            }
            offset += depEmbedDim;
        }
    }

    public double[] output(final int[] feats, final int[] labels) {
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
            if (j < numWordLayers)
                embedding = wordEmbeddings;
            else if (j < numWordLayers + numPosLayers)
                embedding = posEmbeddings;
            else embedding = labelEmbeddings;

            if (saved != null && (j >= numWordLayers || maps.preComputeMap[j].containsKey(tok))) {
                int id = tok;
                if (j < numWordLayers)
                    id = maps.preComputeMap[j].get(tok);
                double[] s = saved[j][id];
                for (int i = 0; i < hidden.length; i++) {
                    hidden[i] += s[i];
                }
            } else {
                for (int i = 0; i < hidden.length; i++) {
                    for (int k = 0; k < embedding[tok].length; k++) {
                        hidden[i] += hiddenLayer[i][offset + k] * embedding[tok][k];
                    }
                }
            }
            offset += embedding[tok].length;
        }

        for (int i = 0; i < hidden.length; i++) {
            hidden[i] += hiddenLayerBias[i];
            hidden[i] = activation.activate(hidden[i]);
        }

        double[] probs = new double[softmaxLayerBias.length];
        double sum = 0;
        int argmax = 0;
        int numActiveLabels = 0;
        for (int i = 0; i < probs.length; i++) {
            if (labels[i] >= 0) {
                for (int j = 0; j < hidden.length; j++) {
                    probs[i] += softmaxLayer[i][j] * hidden[j];
                }
                probs[i] += softmaxLayerBias[i];
                if (probs[i] > probs[argmax])
                    argmax = i;
            }
        }

        double max = probs[argmax];
        for (int i = 0; i < probs.length; i++) {
            if (labels[i] >= 0) {
                numActiveLabels++;
                probs[i] = probs[i] - max;
                sum += Math.exp(probs[i]);
            }
        }

        for (int i = 0; i < probs.length; i++) {
            if (sum != 0)
                probs[i] -= Math.log(sum);
            else
                probs[i] = 1.0 / numActiveLabels;
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

    public int getNumWordLayers() {
        return numWordLayers;
    }

    public int getNumPosLayers() {
        return numPosLayers;
    }

    public int getNumDepLayers() {
        return numDepLayers;
    }

    public int getNumDepLabels() {
        return numDepLabels;
    }

    public int getDepEmbedDim() {
        return depEmbedDim;
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

    public int getSecondHiddenLayerDim() {
        return secondHiddenLayerDim;
    }

    public int getNumWords() {
        return numWords;
    }

    public int getNumPos() {
        return numPos;
    }

    public int getPosEmbedDim() {
        return posEmbedDim;
    }

    public int getSoftmaxLayerDim() {
        return softmaxLayerDim;
    }

    public NetworkMatrices getMatrices() {
        return matrices;
    }

    /**
     * This is used just for testing.
     *
     * @return
     */
    public MLPNetwork clone() {
        MLPNetwork network = new MLPNetwork(maps, options, depLabels, numDepLabels, depEmbedDim, wordEmbedDim, hiddenLayerDim,
                hiddenLayerIntDim, secondHiddenLayerDim, numWords, numPos, posEmbedDim, softmaxLayerDim, null, null, activationType,
                numWordLayers, numPosLayers, numDepLayers);
        network.matrices = matrices.clone();
        return network;
    }


}
