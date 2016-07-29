package YaraParser.Learning;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/21/16
 * Time: 3:39 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

import YaraParser.Accessories.Options;
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
    //todo make them private.
    double[][] wordEmbeddings;
    double[][] posEmbeddings;
    double[][] labelEmbeddings;
    double[][] hiddenLayer;
    double[] hiddenLayerBias;
    double[][] softmaxLayer;
    double[] softmaxLayerBias;
    double[][][] saved;

    public MLPNetwork(IndexMaps maps, Options options, ArrayList<Integer> dependencyLabels, int wDim) {
        this.maps = maps;
        this.options = options;
        this.dependencyLabels = dependencyLabels;
        wordEmbeddings = new double[maps.vocabSize() + 2][wDim];
        posEmbeddings = new double[maps.posSize() + 2][32];
        labelEmbeddings = new double[maps.relSize() + 2][32];
        hiddenLayer = new double[options.hiddenLayer1Size][numberOfPosEmbeddingLayers * wordEmbeddings[0].length +
                numberOfPosEmbeddingLayers * posEmbeddings[0].length + numberOfLabelEmbeddingLayers * labelEmbeddings[0].length];

        hiddenLayerBias = new double[options.hiddenLayer1Size];
        softmaxLayer = new double[2 * (dependencyLabels.size() + 1)][options.hiddenLayer1Size];
        softmaxLayerBias = new double[2 * (dependencyLabels.size() + 1)];

        Random random = new Random();
        for (int i = 0; i < wordEmbeddings.length; i++) {
            for (int j = 0; j < wordEmbeddings[i].length; j++) {
                wordEmbeddings[i][j] = random.nextGaussian() * 0.01;
            }
        }

        for (int i = 0; i < posEmbeddings.length; i++) {
            for (int j = 0; j < posEmbeddings[i].length; j++) {
                posEmbeddings[i][j] = random.nextGaussian() * 0.01;
            }
        }

        for (int i = 0; i < labelEmbeddings.length; i++) {
            for (int j = 0; j < labelEmbeddings[i].length; j++) {
                labelEmbeddings[i][j] = random.nextGaussian() * 0.01;
            }
        }

        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenLayerBias[i] = 0.2;
            for (int j = 0; j < hiddenLayer[i].length; j++) {
                hiddenLayer[i][j] = random.nextGaussian() * 0.01;
            }
        }

        for (int i = 0; i < softmaxLayer.length; i++) {
            softmaxLayerBias[i] = 0;
            for (int j = 0; j < softmaxLayer[i].length; j++) {
                softmaxLayer[i][j] = random.nextGaussian() * 0.01;
            }
        }

        int numOfPretrained = 0;
        for (int i = 0; i < maps.vocabSize() + 2; i++) {
            double[] embeddings = maps.embeddings(i);
            if (embeddings != null) {
                wordEmbeddings[i] = embeddings;
                numOfPretrained++;
            }
        }
        System.out.println("num of pre-trained embedding " + numOfPretrained + " out of " + maps.vocabSize());
    }

    public static void averageNetworks(MLPNetwork toAverageFrom, MLPNetwork averaged, double r1, double r2) {
        ArrayList<double[][]> matrices1 = toAverageFrom.getAllMatrices();
        ArrayList<double[][]> matrices2 = averaged.getAllMatrices();
        for (int m = 0; m < matrices1.size(); m++) {
            for (int i = 0; i < matrices1.get(m).length; i++) {
                for (int j = 0; j < matrices1.get(m)[i].length; j++) {
                    matrices2.get(m)[i][j] = r1 * matrices1.get(m)[i][j] + r2 * matrices2.get(m)[i][j];
                }
            }
        }

        ArrayList<double[]> vectors1 = toAverageFrom.getAllVectors();
        ArrayList<double[]> vectors2 = averaged.getAllVectors();
        for (int m = 0; m < vectors1.size(); m++) {
            for (int i = 0; i < vectors1.get(m).length; i++) {
                vectors2.get(m)[i] = r1 * vectors1.get(m)[i] + r2 * vectors2.get(m)[i];
            }
        }
    }

    public void preCompute() {
        saved = new double[numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers + numberOfLabelEmbeddingLayers][][];
        for (int i = 0; i < numberOfWordEmbeddingLayers; i++)
            saved[i] = new double[maps.preComputeMap.size()][hiddenLayer.length];
        for (int i = numberOfWordEmbeddingLayers; i < numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers; i++)
            saved[i] = new double[posEmbeddings.length][hiddenLayer.length];
        for (int i = numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers;
             i < numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers + numberOfLabelEmbeddingLayers; i++)
            saved[i] = new double[labelEmbeddings.length][hiddenLayer.length];


        int wordEmbeddingSize = wordEmbeddings[0].length;
        int posEmbeddingSize = posEmbeddings[0].length;
        int labelEmbeddingSize = labelEmbeddings[0].length;
        for (int id : maps.preComputeMap.keySet()) {
            int v = maps.preComputeMap.get(id);
            int offset = 0;
            for (int pos = 0; pos < numberOfWordEmbeddingLayers; pos++) {
                for (int j = 0; j < hiddenLayer.length; j++) {
                    for (int k = 0; k < wordEmbeddingSize; k++) {
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
            int offset = numberOfWordEmbeddingLayers * wordEmbeddingSize + numberOfPosEmbeddingLayers * posEmbeddingSize;
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

    public ArrayList<double[][]> getAllMatrices() {
        ArrayList<double[][]> matrices = new ArrayList<>();
        matrices.add(wordEmbeddings);
        matrices.add(posEmbeddings);
        matrices.add(labelEmbeddings);
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

}
