package edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork;

import edu.columbia.cs.nlp.YaraParser.Accessories.Utils;
import edu.columbia.cs.nlp.YaraParser.Structures.Enums.EmbeddingTypes;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/29/16
 * Time: 2:34 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */
public class NetworkMatrices implements Serializable {
    private double[][] wordEmbedding;
    private double[][] posEmbedding;
    private double[][] labelEmbedding;
    private double[][] hiddenLayer;
    private double[] hiddenLayerBias;
    private double[][] softmaxLayer;
    private double[] softmaxLayerBias;
    private double[][] secondHiddenLayer;
    private double[] secondHiddenLayerBias;

    public NetworkMatrices(double[][] wordEmbedding, double[][] posEmbedding, double[][] labelEmbedding, double[][] hiddenLayer, double[]
            hiddenLayerBias, double[][] secondHiddenLayer, double[] secondHiddenLayerBias, double[][] softmaxLayer, double[] softmaxLayerBias) {
        this.wordEmbedding = wordEmbedding;
        this.posEmbedding = posEmbedding;
        this.labelEmbedding = labelEmbedding;
        this.hiddenLayer = hiddenLayer;
        this.hiddenLayerBias = hiddenLayerBias;
        this.softmaxLayer = softmaxLayer;
        this.softmaxLayerBias = softmaxLayerBias;
        this.secondHiddenLayer = secondHiddenLayer;
        this.secondHiddenLayerBias = secondHiddenLayerBias;
    }

    public NetworkMatrices(int wSize, int wDim, int pSize, int pDim, int lSize, int lDim, int hDim, int hIntDim, int h2Dim, int softDim) {
        wordEmbedding = new double[wSize][wDim];
        posEmbedding = new double[pSize][pDim];
        labelEmbedding = new double[lSize][lDim];
        hiddenLayer = new double[hDim][hIntDim];

        hiddenLayerBias = new double[hDim];
        int s2Dim = h2Dim > 0 ? h2Dim : hDim;
        softmaxLayer = new double[softDim][s2Dim];
        softmaxLayerBias = new double[softDim];
        if (h2Dim > 0) {
            secondHiddenLayer = new double[h2Dim][hDim];
            secondHiddenLayerBias = new double[h2Dim];
        }
    }

    public void resetToPretrainedWordEmbeddings(int i, double[] embeddings) {
        this.wordEmbedding[i] = embeddings;
    }

    public void modify(EmbeddingTypes t, int i, int j, double change) throws Exception {
        double newValue;
        if (t.equals(EmbeddingTypes.WORD)) {
            newValue = wordEmbedding[i][j] + change;
            wordEmbedding[i][j] = newValue;
        } else if (t.equals(EmbeddingTypes.POS)) {
            newValue = posEmbedding[i][j] + change;
            posEmbedding[i][j] = newValue;
        } else if (t.equals(EmbeddingTypes.DEPENDENCY)) {
            newValue = labelEmbedding[i][j] + change;
            labelEmbedding[i][j] = newValue;
        } else if (t.equals(EmbeddingTypes.HIDDENLAYER)) {
            newValue = hiddenLayer[i][j] + change;
            hiddenLayer[i][j] = newValue;
        } else if (t.equals(EmbeddingTypes.HIDDENLAYERBIAS)) {
            assert j == -1;
            newValue = hiddenLayerBias[i] + change;
            hiddenLayerBias[i] = newValue;
        } else if (t.equals(EmbeddingTypes.SECONDHIDDENLAYER)) {
            newValue = secondHiddenLayer[i][j] + change;
            secondHiddenLayer[i][j] = newValue;
        } else if (t.equals(EmbeddingTypes.SECONDHIDDENLAYERBIAS)) {
            assert j == -1;
            newValue = secondHiddenLayerBias[i] + change;
            secondHiddenLayerBias[i] = newValue;
        } else if (t.equals(EmbeddingTypes.SOFTMAX)) {
            newValue = softmaxLayer[i][j] + change;
            softmaxLayer[i][j] = newValue;
        } else if (t.equals(EmbeddingTypes.SOFTMAXBIAS)) {
            assert j == -1;
            newValue = softmaxLayerBias[i] + change;
            softmaxLayerBias[i] = newValue;
        } else
            throw new Exception("Embedding type not supported");

        if (Double.isNaN(newValue))
            throw new Exception("Modify matrix value to NAN for " + t);
    }

    public double[][] getWordEmbedding() {
        return wordEmbedding;
    }

    public double[][] getPosEmbedding() {
        return posEmbedding;
    }

    public double[][] getLabelEmbedding() {
        return labelEmbedding;
    }

    public double[][] getHiddenLayer() {
        return hiddenLayer;
    }

    public double[] getHiddenLayerBias() {
        return hiddenLayerBias;
    }

    public double[][] getSecondHiddenLayer() {
        return secondHiddenLayer;
    }

    public double[] getSecondHiddenLayerBias() {
        return secondHiddenLayerBias;
    }

    public double[][] getSoftmaxLayer() {
        return softmaxLayer;
    }

    public double[] getSoftmaxLayerBias() {
        return softmaxLayerBias;
    }

    public ArrayList<double[][]> getAllMatrices() {
        ArrayList<double[][]> matrices = new ArrayList<>();
        matrices.add(wordEmbedding);
        matrices.add(posEmbedding);
        matrices.add(labelEmbedding);
        matrices.add(hiddenLayer);
        matrices.add(secondHiddenLayer);
        matrices.add(softmaxLayer);
        return matrices;
    }

    public ArrayList<double[]> getAllVectors() {
        ArrayList<double[]> vectors = new ArrayList<>();
        vectors.add(hiddenLayerBias);
        vectors.add(secondHiddenLayerBias);
        vectors.add(softmaxLayerBias);
        return vectors;
    }

    /**
     * Merges the values by summation and puts everything to the first layer
     *
     * @param matrices
     */
    public void mergeMatricesInPlaceForNonSaved(NetworkMatrices matrices) {
        Utils.addInPlace(wordEmbedding, matrices.getWordEmbedding());
        Utils.addInPlace(posEmbedding, matrices.getPosEmbedding());
        Utils.addInPlace(labelEmbedding, matrices.getLabelEmbedding());
        Utils.addInPlace(hiddenLayer, matrices.getHiddenLayer());
        Utils.addInPlace(hiddenLayerBias, matrices.getHiddenLayerBias());
        Utils.addInPlace(secondHiddenLayer, matrices.getSecondHiddenLayer());
        Utils.addInPlace(secondHiddenLayerBias, matrices.getSecondHiddenLayerBias());
        Utils.addInPlace(softmaxLayer, matrices.getSoftmaxLayer());
        Utils.addInPlace(softmaxLayerBias, matrices.getSoftmaxLayerBias());
    }

    public NetworkMatrices clone() {
        return new NetworkMatrices(Utils.clone(wordEmbedding), Utils.clone(posEmbedding), Utils.clone(labelEmbedding), Utils.clone(hiddenLayer),
                Utils.clone(hiddenLayerBias), Utils.clone(secondHiddenLayer), Utils.clone(secondHiddenLayerBias), Utils.clone(softmaxLayer),
                Utils.clone(softmaxLayerBias));
    }
}
