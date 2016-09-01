package edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/21/16
 * Time: 3:39 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Learning.Activation.*;
import edu.columbia.cs.nlp.CuraParser.Learning.Activation.Enums.ActivationType;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.FirstHiddenLayer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.Layer;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.FixInit;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.Initializer;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.XavierInit;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.enums.WeightInit;
import edu.columbia.cs.nlp.CuraParser.Structures.Enums.EmbeddingTypes;
import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.ParserType;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

/**
 * Manual MLP model
 */
public class MLPNetwork implements Serializable {
    final public Options options;
    //todo try to get rid of this.
    public IndexMaps maps;
    int numWordLayers;
    int numPosLayers;
    int numDepLayers;
    int numOutputs;
    int numDepLabels;
    int wDim;
    int pDim;
    int depDim;
    ArrayList<Integer> depLabels;
    private ArrayList<Layer> layers;

    public MLPNetwork(IndexMaps maps, Options options, ArrayList<Integer> depLabels, int wDim, int pDim, int lDim, ParserType parserType) {
        Random random = new Random();
        this.maps = maps;

        if (parserType == ParserType.ArcEager) {
            numWordLayers = 22;
            numPosLayers = 22;
            numDepLayers = 11;
        } else {
            numWordLayers = 20;
            numPosLayers = 20;
            numDepLayers = 12;
        }
        this.wDim = wDim;
        this.pDim = pDim;
        this.depDim = lDim;
        int numWords = maps.vocabSize() + 2;
        int numDepLabels = maps.relSize() + 2;
        int numPos = maps.posSize() + 2;
        this.layers = new ArrayList<>();
        int nIn = numWordLayers * wDim + numPosLayers * pDim + numDepLayers * lDim;

        WeightInit hiddenInit = isRelu(options) ? WeightInit.RELU : WeightInit.UNIFORM;
        WeightInit hiddenBiasInit = isSimpleRelu(options) ? WeightInit.FIX : WeightInit.UNIFORM;
        Activation activation = getActivation(options, random);
        Initializer hiddenInitializer = WeightInit.initializer(hiddenInit, random, nIn, options.networkProperties.hiddenLayer1Size, 0);
        Initializer hiddenBiasInitializer = WeightInit.initializer(hiddenBiasInit, random, nIn, options.networkProperties.hiddenLayer1Size, 0.2);

        Layer inputLayer = new FirstHiddenLayer(activation, nIn, options.networkProperties.hiddenLayer1Size, hiddenInitializer, hiddenBiasInitializer,
                numWordLayers, numPosLayers, numDepLayers, random, maps.preComputeMap, numWords, numPos, numDepLabels, wDim, pDim, lDim,
                maps.getEmbeddingsDictionary());

        layers.add(inputLayer);

        if (options.networkProperties.hiddenLayer2Size > 0) {
            layers.add(new Layer(activation, options.networkProperties.hiddenLayer1Size, options.networkProperties.hiddenLayer2Size,
                    hiddenInitializer, hiddenBiasInitializer));
        }

        int outputnIn = layers.get(layers.size() - 1).nOut();
        numOutputs = 2 * (depLabels.size() + 1);
        layers.add(new Layer(new LogisticSoftMax(), outputnIn, numOutputs, new XavierInit(random, outputnIn, numOutputs), new FixInit(0),
                options.networkProperties.outputBiasTerm));

        this.depLabels = depLabels;
        this.numDepLabels = depLabels.size();
        this.options = options;
    }

    public MLPNetwork(Options options, ArrayList<Layer> layers, int numWordLayers, int numPosLayers, int numDepLayers, int numOutputs,
                      ArrayList<Integer> depLabels) {
        this.options = options;
        this.layers = layers;
        this.numWordLayers = numWordLayers;
        this.numPosLayers = numPosLayers;
        this.numDepLayers = numDepLayers;
        this.numOutputs = numOutputs;
        this.numDepLabels = depLabels.size();
        this.depLabels = depLabels;
        this.maps = null;
    }

    private Activation getActivation(Options options, Random random) {
        if (options.networkProperties.activationType == ActivationType.RELU) return new Relu();
        if (options.networkProperties.activationType == ActivationType.LeakyRELU) return new LeakyRelu(options.networkProperties.reluLeakAlpha);
        if (options.networkProperties.activationType == ActivationType.RandomRelu)
            return new RandomizedRelu(options.networkProperties.rReluL, options.networkProperties.rReluU, random);
        if (options.networkProperties.activationType == ActivationType.CUBIC) return new Cubic();
        if (options.networkProperties.activationType == ActivationType.LogisticSoftMax) return new LogisticSoftMax();
        return new Identity();
    }

    private boolean isRelu(Options options) {
        return options.networkProperties.activationType == ActivationType.RELU || options.networkProperties.activationType == ActivationType
                .LeakyRELU || options.networkProperties.activationType == ActivationType.RandomRelu;
    }

    private boolean isSimpleRelu(Options options) {
        return options.networkProperties.activationType == ActivationType.RELU;
    }

    public void resetPreComputeMap() {
        ((FirstHiddenLayer) layer(0)).getWordEmbeddings().setPrecomputationMap(maps.preComputeMap);
    }

    public void averageNetworks(MLPNetwork averaged, final double r1, final double r2) {
        ArrayList<Layer> avgLayers = averaged.getLayers();
        assert avgLayers.size() == layers.size();
        for (int i = 0; i < layers.size(); i++) {
            Utils.avgMatrices(layers.get(i).getW(), avgLayers.get(i).getW(), r1, r2);
            Utils.avgVectors(layers.get(i).getB(), avgLayers.get(i).getB(), r1, r2);

            if (i == 0) {
                Utils.avgMatrices(((FirstHiddenLayer) layer(i)).getWordEmbeddings().getW(), ((FirstHiddenLayer) avgLayers.get(i))
                        .getWordEmbeddings().getW(), r1, r2);
                Utils.avgMatrices(((FirstHiddenLayer) layer(i)).getPosEmbeddings().getW(), ((FirstHiddenLayer) avgLayers.get(i))
                        .getPosEmbeddings().getW(), r1, r2);
                Utils.avgMatrices(((FirstHiddenLayer) layer(i)).getDepEmbeddings().getW(), ((FirstHiddenLayer) avgLayers.get(i))
                        .getDepEmbeddings().getW(), r1, r2);
            }
        }
    }

    public double[] output(final double[] feats, final double[] labels) {
        double[] o = feats;
        for (int l = 0; l < layers.size() - 1; l++) {
            o = layers.get(l).activate(layers.get(l).forward(o), true);
        }
        o = layers.get(layers.size() - 1).forward(o, labels);
        return o;
    }

    public double[] output(final double[] feats, final double[] labels, boolean logOut) {
        double[] o = feats;
        for (int l = 0; l < layers.size() - 1; l++) {
            o = layers.get(l).activate(layers.get(l).forward(o), true);
        }
        o = layers.get(layers.size() - 1).forward(o, labels, logOut, true);
        return o;
    }

    public Options getOptions() {
        return options;
    }

    public ArrayList<Layer> getLayers() {
        return layers;
    }

    public Layer layer(int index) {
        return layers.get(index);
    }

    public int numLayers() {
        return layers.size();
    }

    /**
     * This is used just for testing.
     *
     * @return
     */
    public MLPNetwork clone() {
        return clone(false, true);
    }

    public void emptyPrecomputedMap() {
        ((FirstHiddenLayer) layer(0)).emptyPrecomputedMap();
    }

    public void modify(int layerIndex, int i, int j, double change) {
        if (j == -1)
            layers.get(layerIndex).modifyB(i, change);
        else
            layers.get(layerIndex).modifyW(i, j, change);
    }

    public void modify(EmbeddingTypes embeddingTypes, int i, int j, double change) {
        ((FirstHiddenLayer) layers.get(0)).modify(embeddingTypes, i, j, change);
    }

    public MLPNetwork clone(boolean zeroOut, boolean deepCopy) {
        ArrayList<Layer> layers = new ArrayList<>(this.layers.size());
        for (int l = 0; l < this.layers.size(); l++) {
            layers.add(this.layers.get(l).copy(zeroOut, deepCopy));
        }
        MLPNetwork network = new MLPNetwork(options, layers, numWordLayers, numPosLayers, numDepLayers, numOutputs, depLabels);
        ((FirstHiddenLayer) network.layer(0)).setPrecomputationMap(((FirstHiddenLayer) layer(0)).getPrecomputationMap());
        return network;
    }

    public int getNumOutputs() {
        return numOutputs;
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

    public ArrayList<Integer> getDepLabels() {
        return depLabels;
    }

    public void preCompute() {
        ((FirstHiddenLayer) layer(0)).preCompute();
    }

    public double[][][] instantiateSavedGradients() {
        int numDepLabels = ((FirstHiddenLayer) layer(0)).getDepEmbeddings().vocabSize();
        int numPos = ((FirstHiddenLayer) layer(0)).getPosEmbeddings().vocabSize();
        double[][][] savedGradients = new double[getNumWordLayers() + getNumPosLayers() + getNumDepLayers()][][];
        for (int i = 0; i < getNumWordLayers(); i++)
            savedGradients[i] = new double[((FirstHiddenLayer) layer(0)).getWordEmbeddings().numOfPrecomputedItems(i)][layer(0).nOut()];
        for (int i = getNumWordLayers(); i < getNumWordLayers() + getNumPosLayers(); i++)
            savedGradients[i] = new double[numPos][layer(0).nOut()];
        for (int i = getNumWordLayers() + getNumPosLayers();
             i < getNumWordLayers() + getNumPosLayers() + getNumDepLayers(); i++)
            savedGradients[i] = new double[numDepLabels][layer(0).nOut()];
        return savedGradients;
    }

    public int getwDim() {
        return wDim;
    }

    public int getpDim() {
        return pDim;
    }

    public int getDepDim() {
        return depDim;
    }

    public final double[][] getWordEmbedding() {
        return ((FirstHiddenLayer) layer(0)).getWordEmbeddings().getW();
    }

    public final double[][] getPosEmbedding() {
        return ((FirstHiddenLayer) layer(0)).getPosEmbeddings().getW();
    }

    public final double[][] getDepEmbedding() {
        return ((FirstHiddenLayer) layer(0)).getDepEmbeddings().getW();
    }
}
