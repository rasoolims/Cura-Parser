package YaraParser.Learning;

import YaraParser.Accessories.Pair;
import YaraParser.Structures.EmbeddingTypes;
import YaraParser.Structures.NeuralTrainingInstance;

import java.util.List;
import java.util.concurrent.Callable;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/1/16
 * Time: 12:12 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class CostThread implements Callable<Pair<Pair<Double, Double>, NetworkMatrices>> {

    final MLPNetwork mlpNetwork;
    List<NeuralTrainingInstance> instances;
    int batchSize;

    public CostThread(final MLPNetwork mlpNetwork, List<NeuralTrainingInstance> instances, int batchSize) {
        this.mlpNetwork = mlpNetwork;
        this.instances = instances;
        this.batchSize = batchSize;
    }

    @Override
    public Pair<Pair<Double, Double>, NetworkMatrices> call() throws Exception {
        NetworkMatrices gradients = new NetworkMatrices(mlpNetwork.numOfWords, mlpNetwork.wordEmbeddingSize, mlpNetwork.numOfPos,
                mlpNetwork.posEmbeddingSize, mlpNetwork.numOfDependencyLabels, mlpNetwork.labelEmbeddingSize, mlpNetwork.hiddenLayerSize,
                mlpNetwork.hiddenLayerIntSize, mlpNetwork.softmaxLayerSize);
        double cost = 0;
        double correct = 0;

        for (NeuralTrainingInstance instance : instances) {
            int[] features = instance.getFeatures();
            int[] label = instance.getLabel();

            double[] hidden = new double[mlpNetwork.hiddenLayerSize];

            final double[][] softmaxLayer = mlpNetwork.matrices.getSoftmaxLayer();
            final double[] softmaxLayerBias = mlpNetwork.matrices.getSoftmaxLayerBias();
            final double[][] hiddenLayer = mlpNetwork.matrices.getHiddenLayer();
            final double[] hiddenLayerBias = mlpNetwork.matrices.getHiddenLayerBias();
            final double[][] wordEmbeddings = mlpNetwork.matrices.getWordEmbedding();
            final double[][] posEmbeddings = mlpNetwork.matrices.getPosEmbedding();
            final double[][] labelEmbeddings = mlpNetwork.matrices.getLabelEmbedding();

            int offset = 0;
            for (int j = 0; j < features.length; j++) {
                int tok = features[j];
                final double[] embedding;
                if (j < mlpNetwork.numberOfWordEmbeddingLayers)
                    embedding = wordEmbeddings[tok];
                else if (j < mlpNetwork.numberOfWordEmbeddingLayers + mlpNetwork.numberOfPosEmbeddingLayers)
                    embedding = posEmbeddings[tok];
                else
                    embedding = labelEmbeddings[tok];

                if (mlpNetwork.saved != null && (j >= mlpNetwork.numberOfWordEmbeddingLayers || mlpNetwork.maps.preComputeMap.containsKey(tok))) {
                    int id = tok;
                    if (j < mlpNetwork.numberOfWordEmbeddingLayers)
                        id = mlpNetwork.maps.preComputeMap.get(tok);
                    for (int h = 0; h < hidden.length; h++) {
                        hidden[h] += mlpNetwork.saved[j][id][h];
                    }
                } else {
                    for (int h = 0; h < hidden.length; h++) {
                        for (int k = 0; k < embedding.length; k++) {
                            hidden[h] += hiddenLayer[h][offset + k] * embedding[k];
                        }
                    }
                }
                offset += embedding.length;
            }

            double[] reluHidden = new double[hidden.length];
            for (int h = 0; h < hidden.length; h++) {
                hidden[h] += hiddenLayerBias[h];
                //relu
                reluHidden[h] = Math.max(0, hidden[h]);
            }

            int argmax = -1;
            int gold = -1;
            double sum = 0;
            double[] probs = new double[softmaxLayerBias.length];
            for (int i = 0; i < probs.length; i++) {
                if (label[i] >= 0) {
                    if (label[i] == 1)
                        gold = i;
                    for (int j = 0; j < reluHidden.length; j++) {
                        probs[i] += softmaxLayer[i][j] * reluHidden[j];
                    }

                    probs[i] += softmaxLayerBias[i];
                    probs[i] = Math.exp(probs[i]);
                    sum += probs[i];

                    if (argmax < 0 || probs[i] > probs[argmax])
                        argmax = i;
                }
            }

            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sum;
            }

            cost -= Math.log(probs[gold]);
            if (argmax == gold)
                correct += 1.0;

            double[] reluGradW = new double[reluHidden.length];
            for (int i = 0; i < probs.length; i++) {
                if (label[i] >= 0) {
                    double delta = (-label[i] + probs[i]) / batchSize;
                    gradients.modify(EmbeddingTypes.SOFTMAXBIAS, i, -1, delta);
                    for (int h = 0; h < reluHidden.length; h++) {
                        gradients.modify(EmbeddingTypes.SOFTMAX, i, h, delta * reluHidden[h]);
                        reluGradW[h] += delta * softmaxLayer[i][h];
                    }
                }
            }

            double[] hiddenGrad = new double[hidden.length];
            for (int h = 0; h < reluHidden.length; h++) {
                hiddenGrad[h] = (reluHidden[h] == 0. ? 0 : reluGradW[h]);
                gradients.modify(EmbeddingTypes.HIDDENLAYERBIAS, h, -1, hiddenGrad[h]);
            }

            offset = 0;
            for (int index = 0; index < mlpNetwork.numberOfWordEmbeddingLayers; index++) {
                double[] embeddings = wordEmbeddings[features[index]];
                for (int h = 0; h < reluHidden.length; h++) {
                    for (int k = 0; k < embeddings.length; k++) {
                        gradients.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, hiddenGrad[h] * embeddings[k]);
                        gradients.modify(EmbeddingTypes.WORD, features[index], k, hiddenGrad[h] * hiddenLayer[h][offset + k]);
                    }
                }
                offset += embeddings.length;
            }

            for (int index = mlpNetwork.numberOfWordEmbeddingLayers; index < mlpNetwork
                    .numberOfWordEmbeddingLayers + mlpNetwork.numberOfPosEmbeddingLayers; index++) {
                double[] embeddings = posEmbeddings[features[index]];
                for (int h = 0; h < reluHidden.length; h++) {
                    for (int k = 0; k < embeddings.length; k++) {
                        gradients.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, hiddenGrad[h] * embeddings[k]);
                        gradients.modify(EmbeddingTypes.POS, features[index], k, hiddenGrad[h] * hiddenLayer[h][offset + k]);
                    }
                }
                offset += embeddings.length;
            }
            for (int index = mlpNetwork.numberOfWordEmbeddingLayers + mlpNetwork
                    .numberOfPosEmbeddingLayers; index < mlpNetwork.numberOfWordEmbeddingLayers +
                    mlpNetwork.numberOfPosEmbeddingLayers + mlpNetwork.numberOfLabelEmbeddingLayers; index++) {
                double[] embeddings = labelEmbeddings[features[index]];
                for (int h = 0; h < reluHidden.length; h++) {
                    for (int k = 0; k < embeddings.length; k++) {
                        gradients.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, hiddenGrad[h] * embeddings[k]);
                        gradients.modify(EmbeddingTypes.DEPENDENCY, features[index], k, hiddenGrad[h] * hiddenLayer[h][offset + k]);
                    }
                }
                offset += embeddings.length;
            }
        }

        return new Pair<>(new Pair<>(cost, correct), gradients);
    }
}
