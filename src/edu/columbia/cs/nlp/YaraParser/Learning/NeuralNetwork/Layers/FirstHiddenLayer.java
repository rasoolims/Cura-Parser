package edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.Layers;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 7:50 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

import edu.columbia.cs.nlp.YaraParser.Accessories.Utils;
import edu.columbia.cs.nlp.YaraParser.Learning.Activation.Activation;
import edu.columbia.cs.nlp.YaraParser.Learning.WeightInit.Initializer;

import java.util.HashMap;
import java.util.Random;

/**
 * This class shows the first hidden layer layer which its input is a concatenation of different embedding layer.
 */
public class FirstHiddenLayer extends Layer {
    WordEmbeddingLayer wordEmbeddings;
    EmbeddingLayer posEmbeddings;
    EmbeddingLayer depEmbeddings;
    int numWordLayers;
    int numPosLayers;
    int numDepLayers;

    // pre-computed items.
    private double[][][] saved;

    public FirstHiddenLayer(Activation activation, int nIn, int nOut, Initializer initializer,
                            int numWordLayers, int numPosLayers, int numDepLayers,
                            Random random, HashMap<Integer, Integer>[] precomputationMap,
                            int numW, int numPos, int numDep,
                            int wDim, int posDim, int depDim,
                            HashMap<Integer, double[]> embeddingsDictionary) {
        super(activation, nIn, nOut, initializer);
        assert wordEmbeddings.numOfEmbeddingSlot() == numWordLayers;
        this.wordEmbeddings = new WordEmbeddingLayer(numW, wDim, random, precomputationMap);
        this.posEmbeddings = new EmbeddingLayer(numPos, posDim, random);
        this.depEmbeddings = new EmbeddingLayer(numDep, depDim, random);
        this.numWordLayers = numWordLayers;
        this.numPosLayers = numPosLayers;
        this.numDepLayers = numDepLayers;
        this.wordEmbeddings.addPretrainedVectors(embeddingsDictionary);
        preCompute();
    }

    public void preCompute() {
        saved = new double[numWordLayers + numPosLayers + numDepLayers][][];
        for (int i = 0; i < numWordLayers; i++)
            saved[i] = new double[wordEmbeddings.numOfPrecomputedItems(i)][nOut()];
        for (int i = numWordLayers; i < numWordLayers + numPosLayers; i++)
            saved[i] = new double[posEmbeddings.nOut()][nOut()];
        for (int i = numWordLayers + numPosLayers; i < numWordLayers + numPosLayers + numDepLayers; i++)
            saved[i] = new double[depEmbeddings.nOut()][nOut()];

        int offset = 0;
        for (int pos = 0; pos < numWordLayers; pos++) {
            for (int tok : wordEmbeddings.preComputedIds(pos)) {
                int id = wordEmbeddings.preComputeId(pos, tok);
                for (int h = 0; h < nOut(); h++) {
                    for (int k = 0; k < wordEmbeddings.dim(); k++) {
                        saved[pos][id][h] += w[h][offset + k] * wordEmbeddings.w(tok, k);
                    }
                }
            }
            offset += wordEmbeddings.dim();
        }

        for (int pos = 0; pos < numPosLayers; pos++) {
            for (int tok = 0; tok < posEmbeddings.nOut(); tok++) {
                int indOffset = numWordLayers;
                for (int h = 0; h < nOut(); h++) {
                    for (int k = 0; k < posEmbeddings.dim(); k++) {
                        saved[pos + indOffset][tok][h] += w[h][offset + k] * posEmbeddings.w(tok, k);
                    }
                }
            }
            offset += posEmbeddings.dim();
        }

        for (int pos = 0; pos < numDepLayers; pos++) {
            for (int tok = 0; tok < depEmbeddings.nOut(); tok++) {
                int indOffset = numWordLayers + numPosLayers;
                for (int h = 0; h < nOut(); h++) {
                    for (int k = 0; k < depEmbeddings.dim(); k++) {
                        saved[pos + indOffset][tok][h] += w[h][offset + k] * depEmbeddings.w(tok, k);
                    }
                }
            }
            offset += depEmbeddings.dim();
        }
    }


    /**
     * Uses pre-computed maps in order to speed things up.
     *
     * @param input
     * @return
     */
    @Override
    public double[] forward(double[] input) {
        int offset = 0;
        double[] hidden = new double[nOut()];
        for (int j = 0; j < input.length; j++) {
            int tok = (int) input[j];
            EmbeddingLayer embedding;
            if (j < numWordLayers)
                embedding = wordEmbeddings;
            else if (j < numWordLayers + numPosLayers)
                embedding = posEmbeddings;
            else embedding = depEmbeddings;

            if (saved != null && (j >= numWordLayers || wordEmbeddings.isFrequent(j, tok))) {
                int id = tok;
                if (j < numWordLayers)
                    id = wordEmbeddings.preComputeId(j, tok);
                double[] s = saved[j][id];
                for (int i = 0; i < hidden.length; i++) {
                    hidden[i] += s[i];
                }
            } else {
                for (int i = 0; i < hidden.length; i++) {
                    for (int k = 0; k < embedding.dim(); k++) {
                        hidden[i] += w[i][offset + k] * embedding.w(tok, k);
                    }
                }
            }
            offset += embedding.dim();
        }

        if (useBias)
            Utils.sumi(hidden, b);

        return activation.activate(hidden);
    }
}
