package YaraParser.Learning;

import YaraParser.Accessories.Utils;
import YaraParser.Structures.EmbeddingTypes;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;

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

    private NetworkMatrices(double[][] wordEmbedding, double[][] posEmbedding, double[][] labelEmbedding, double[][] hiddenLayer, double[]
            hiddenLayerBias, double[][] softmaxLayer, double[] softmaxLayerBias) {
        this.wordEmbedding = wordEmbedding;
        this.posEmbedding = posEmbedding;
        this.labelEmbedding = labelEmbedding;
        this.hiddenLayer = hiddenLayer;
        this.hiddenLayerBias = hiddenLayerBias;
        this.softmaxLayer = softmaxLayer;
        this.softmaxLayerBias = softmaxLayerBias;
    }

    public NetworkMatrices(int wSize, int wDim, int pSize, int pDim, int lSize, int lDim, int hDim, int hIntDim, int softDim) {
        wordEmbedding = new double[wSize][wDim];
        posEmbedding = new double[pSize][pDim];
        labelEmbedding = new double[lSize][lDim];
        hiddenLayer = new double[hDim][hIntDim];

        hiddenLayerBias = new double[hDim];
        softmaxLayer = new double[softDim][hDim];
        softmaxLayerBias = new double[softDim];
    }

    public void resetToPretrainedWordEmbeddings(int i, double[] embeddings) {
        this.wordEmbedding[i] = embeddings;
    }

    public void modify(EmbeddingTypes t, int i, int j, double change) throws Exception {
        if (t.equals(EmbeddingTypes.WORD))
            wordEmbedding[i][j] += change;
        else if (t.equals(EmbeddingTypes.POS))
            posEmbedding[i][j] += change;
        else if (t.equals(EmbeddingTypes.DEPENDENCY))
            labelEmbedding[i][j] += change;
        else if (t.equals(EmbeddingTypes.HIDDENLAYER))
            hiddenLayer[i][j] += change;
        else if (t.equals(EmbeddingTypes.HIDDENLAYERBIAS)) {
            assert j == -1;
            hiddenLayerBias[i] += change;
        } else if (t.equals(EmbeddingTypes.SOFTMAX))
            softmaxLayer[i][j] += change;
        else if (t.equals(EmbeddingTypes.SOFTMAXBIAS)) {
            assert j == -1;
            softmaxLayerBias[i] += change;
        } else
            throw new Exception("Embedding type not supported");
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
        matrices.add(softmaxLayer);
        return matrices;
    }

    public ArrayList<double[]> getAllVectors() {
        ArrayList<double[]> vectors = new ArrayList<>();
        vectors.add(hiddenLayerBias);
        vectors.add(softmaxLayerBias);
        return vectors;
    }

    /**
     * Merges the values by summation and puts everything to the first layer
     *
     * @param matrices
     */
    public void mergeMatricesInPlaceForNonSaved(NetworkMatrices matrices, HashSet<Integer> seenWords) {
        double[][] otherWordEmbedding = matrices.getWordEmbedding();
        for (int i : seenWords)
            for (int j = 0; j < wordEmbedding[i].length; j++)
                wordEmbedding[i][j] += otherWordEmbedding[i][j];

        Utils.addInPlace(posEmbedding, matrices.getPosEmbedding());
        Utils.addInPlace(labelEmbedding, matrices.getLabelEmbedding());
        Utils.addInPlace(hiddenLayer, matrices.getHiddenLayer());
        Utils.addInPlace(hiddenLayerBias, matrices.getHiddenLayerBias());
        Utils.addInPlace(softmaxLayer, matrices.getSoftmaxLayer());
        Utils.addInPlace(softmaxLayerBias, matrices.getSoftmaxLayerBias());
    }

    public NetworkMatrices clone() {
        return new NetworkMatrices(Utils.clone(wordEmbedding), Utils.clone(posEmbedding), Utils.clone(labelEmbedding), Utils.clone(hiddenLayer),
                Utils.clone(hiddenLayerBias), Utils.clone(softmaxLayer), Utils.clone(softmaxLayerBias));
    }
}
