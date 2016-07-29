package YaraParser.Learning;

import YaraParser.Structures.EmbeddingTypes;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/29/16
 * Time: 2:34 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Gradiants {
    private double[][] wordEmbeddingGradient;
    private double[][] posEmbeddingGradient;
    private double[][] labelEmbeddingGradient;
    private double[][] hiddenLayerGradient;
    private double[] hiddenLayerBiasGradient;
    private double[][] softmaxLayerGradient;
    private double[] softmaxLayerBiasGradient;

    public Gradiants(int wSize, int wDim, int pSize, int pDim, int lSize, int lDim, int hDim, int hIntDim, int softDim) {
        wordEmbeddingGradient = new double[wSize][wDim];
        posEmbeddingGradient = new double[pSize][pDim];
        labelEmbeddingGradient = new double[lSize][lDim];
        hiddenLayerGradient = new double[hDim][hIntDim];

        hiddenLayerBiasGradient = new double[hDim];
        softmaxLayerGradient = new double[softDim][hDim];
        softmaxLayerBiasGradient = new double[softDim];
    }

    public void modify(EmbeddingTypes t, int i, int j, double change) throws Exception {
        if (t.equals(EmbeddingTypes.WORD))
            wordEmbeddingGradient[i][j] += change;
       else if (t.equals(EmbeddingTypes.POS))
            posEmbeddingGradient[i][j] += change;
        else if (t.equals(EmbeddingTypes.DEPENDENCY))
            labelEmbeddingGradient[i][j] += change;
        else  if (t.equals(EmbeddingTypes.HIDDENLAYER))
            hiddenLayerGradient[i][j] += change;
        else if (t.equals(EmbeddingTypes.HIDDENLAYERBIAS)) {
            assert j == -1;
            hiddenLayerBiasGradient[i] += change;
        }
        else  if (t.equals(EmbeddingTypes.SOFTMAX))
            softmaxLayerGradient[i][j] += change;
        else if (t.equals(EmbeddingTypes.SOFTMAXBIAS)) {
            assert j == -1;
            softmaxLayerBiasGradient[i] += change;
        }  else
            throw new Exception("Embedding type not supported");
    }


    public double[][] getWordEmbeddingGradient() {
        return wordEmbeddingGradient;
    }

    public double[][] getPosEmbeddingGradient() {
        return posEmbeddingGradient;
    }

    public double[][] getLabelEmbeddingGradient() {
        return labelEmbeddingGradient;
    }

    public double[][] getHiddenLayerGradient() {
        return hiddenLayerGradient;
    }

    public double[] getHiddenLayerBiasGradient() {
        return hiddenLayerBiasGradient;
    }

    public double[][] getSoftmaxLayerGradient() {
        return softmaxLayerGradient;
    }

    public double[] getSoftmaxLayerBiasGradient() {
        return softmaxLayerBiasGradient;
    }
}
