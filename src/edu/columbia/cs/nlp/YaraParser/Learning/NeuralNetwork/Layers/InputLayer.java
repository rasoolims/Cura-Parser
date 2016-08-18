package edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.Layers;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/18/16
 * Time: 7:50 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

/**
 * This class shows the input layer which is a concatenation of different embedding layer.
 */
public class InputLayer {
    WordEmbeddingLayer wordEmbeddings;
    EmbeddingLayer posEmbeddings;
    EmbeddingLayer depEmbeddings;
    int numWordLayers;
    int numPosLayers;
    int numDepLayers;

    public InputLayer(WordEmbeddingLayer wordEmbeddings, EmbeddingLayer posEmbeddings, EmbeddingLayer depEmbeddings, int numWordLayers,
                      int numPosLayers, int numDepLayers) {
        assert wordEmbeddings.numOfEmbeddingSlot() == numWordLayers;
        this.wordEmbeddings = wordEmbeddings;
        this.posEmbeddings = posEmbeddings;
        this.depEmbeddings = depEmbeddings;
        this.numWordLayers = numWordLayers;
        this.numPosLayers = numPosLayers;
        this.numDepLayers = numDepLayers;
    }
}
