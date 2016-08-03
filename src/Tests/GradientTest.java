package Tests;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 11:07 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

import YaraParser.Learning.MLPNetwork;
import YaraParser.Learning.NetworkMatrices;
import YaraParser.Structures.EmbeddingTypes;
import YaraParser.Structures.NeuralTrainingInstance;
import org.junit.Test;

import java.util.Random;


public class GradientTest {

    @Test
    public void TestWordEmbeddingGradients() throws Exception {
        MLPNetwork network = new MLPNetwork(100, 8, 100, 10, 10, 4, 4);
        NeuralTrainingInstance randomInstance = getRandomInstance(100, 10, 10);
        int goldLabel = randomInstance.gold();
        NetworkMatrices gradients = cost(network, randomInstance);

        double eps = 0.000001;
        for (int i = 0; i < 19; i++) {
            for (int slot = 0; slot < network.getWordEmbedDim(); slot++) {
                int tokNum = randomInstance.getFeatures()[i];
                double gradForTok = gradients.getWordEmbedding()[tokNum][slot];

                NetworkMatrices plusEPS = purturb(network.getMatrices(), EmbeddingTypes.WORD, tokNum, slot, eps);
                NetworkMatrices negEPS = purturb(network.getMatrices(), EmbeddingTypes.WORD, tokNum, slot, -eps);
                double[] plusPurturb = output(plusEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                        network.getNumOfPosLayers());
                double[] negPurturb = output(negEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                        network.getNumOfPosLayers());

                double diff = -(plusPurturb[goldLabel] - negPurturb[goldLabel]) / (2 * eps);
                System.out.println(gradForTok + "\t" + diff);
                assert Math.abs(gradForTok - diff) <= 0.00001;
            }
        }
    }

    @Test
    public void TestPOSEmbeddingGradients() throws Exception {
        MLPNetwork network = new MLPNetwork(100, 8, 100, 10, 10, 4, 4);
        NeuralTrainingInstance randomInstance = getRandomInstance(100, 10, 10);
        int goldLabel = randomInstance.gold();
        NetworkMatrices gradients = cost(network, randomInstance);

        double eps = 0.000001;
        for (int i = 19; i < 38; i++) {
            for (int slot = 0; slot < network.getPosEmbeddingDim(); slot++) {
                int tokNum = randomInstance.getFeatures()[i];
                double gradForTok = gradients.getPosEmbedding()[tokNum][slot];

                NetworkMatrices plusEPS = purturb(network.getMatrices(), EmbeddingTypes.POS, tokNum, slot, eps);
                NetworkMatrices negEPS = purturb(network.getMatrices(), EmbeddingTypes.POS, tokNum, slot, -eps);
                double[] plusPurturb = output(plusEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                        network.getNumOfPosLayers());
                double[] negPurturb = output(negEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                        network.getNumOfPosLayers());

                double diff = -(plusPurturb[goldLabel] - negPurturb[goldLabel]) / (2 * eps);
                System.out.println(gradForTok + "\t" + diff);
                assert Math.abs(gradForTok - diff) <= 0.00001;
            }
        }
    }

    @Test
    public void TestLabelEmbeddingGradients() throws Exception {
        MLPNetwork network = new MLPNetwork(100, 8, 100, 10, 10, 4, 4);
        NeuralTrainingInstance randomInstance = getRandomInstance(100, 10, 10);
        int goldLabel = randomInstance.gold();
        NetworkMatrices gradients = cost(network, randomInstance);

        double eps = 0.000001;
        for (int i = 38; i < 49; i++) {
            for (int slot = 0; slot < network.getLabelEmbeddingSize(); slot++) {
                int tokNum = randomInstance.getFeatures()[42];
                double gradForTok = gradients.getLabelEmbedding()[tokNum][slot];

                NetworkMatrices plusEPS = purturb(network.getMatrices(), EmbeddingTypes.DEPENDENCY, tokNum, slot, eps);
                NetworkMatrices negEPS = purturb(network.getMatrices(), EmbeddingTypes.DEPENDENCY, tokNum, slot, -eps);
                double[] plusPurturb = output(plusEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                        network.getNumOfPosLayers());
                double[] negPurturb = output(negEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                        network.getNumOfPosLayers());

                double diff = -(plusPurturb[goldLabel] - negPurturb[goldLabel]) / (2 * eps);
                System.out.println(gradForTok + "\t" + diff);
                assert Math.abs(gradForTok - diff) <= 0.00001;
            }
        }
    }

    @Test
    public void TestHiddenGradients() throws Exception {
        MLPNetwork network = new MLPNetwork(100, 8, 100, 10, 10, 4, 4);
        NeuralTrainingInstance randomInstance = getRandomInstance(100, 10, 10);
        int goldLabel = randomInstance.gold();
        NetworkMatrices gradients = cost(network, randomInstance);

        double eps = 0.000001;
        for (int i = 0; i < network.getHiddenLayerDim(); i++) {
            for (int slot = 0; slot < network.getHiddenLayerIntDim(); slot++) {
                int tokNum = i;
                double gradForTok = gradients.getHiddenLayer()[tokNum][slot];

                NetworkMatrices plusEPS = purturb(network.getMatrices(), EmbeddingTypes.HIDDENLAYER, tokNum, slot, eps);
                NetworkMatrices negEPS = purturb(network.getMatrices(), EmbeddingTypes.HIDDENLAYER, tokNum, slot, -eps);
                double[] plusPurturb = output(plusEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                        network.getNumOfPosLayers());
                double[] negPurturb = output(negEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                        network.getNumOfPosLayers());

                double diff = -(plusPurturb[goldLabel] - negPurturb[goldLabel]) / (2 * eps);
                System.out.println(gradForTok + "\t" + diff);
                assert Math.abs(gradForTok - diff) <= 0.00001;
            }
        }
    }

    @Test
    public void TestHiddenBiasGradients() throws Exception {
        MLPNetwork network = new MLPNetwork(100, 8, 100, 10, 10, 4, 4);
        NeuralTrainingInstance randomInstance = getRandomInstance(100, 10, 10);
        int goldLabel = randomInstance.gold();
        NetworkMatrices gradients = cost(network, randomInstance);

        double eps = 0.000001;
        for (int i = 0; i < network.getHiddenLayerDim(); i++) {
            int tokNum = i;
            double gradForTok = gradients.getHiddenLayerBias()[tokNum];

            NetworkMatrices plusEPS = purturb(network.getMatrices(), EmbeddingTypes.HIDDENLAYERBIAS, tokNum, -1, eps);
            NetworkMatrices negEPS = purturb(network.getMatrices(), EmbeddingTypes.HIDDENLAYERBIAS, tokNum, -1, -eps);
            double[] plusPurturb = output(plusEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                    network.getNumOfPosLayers());
            double[] negPurturb = output(negEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                    network.getNumOfPosLayers());

            double diff = -(plusPurturb[goldLabel] - negPurturb[goldLabel]) / (2 * eps);
            System.out.println(gradForTok + "\t" + diff);
            assert Math.abs(gradForTok - diff) <= 0.00001;
        }
    }

    @Test
    public void TestSoftmaxGradients() throws Exception {
        MLPNetwork network = new MLPNetwork(100, 8, 100, 10, 10, 4, 4);
        NeuralTrainingInstance randomInstance = getRandomInstance(100, 10, 10);
        int goldLabel = randomInstance.gold();
        NetworkMatrices gradients = cost(network, randomInstance);

        double eps = 0.000001;
        for (int i = 0; i < network.getSoftmaxLayerDim(); i++) {
            for (int slot = 0; slot < network.getHiddenLayerDim(); slot++) {
                int tokNum = i;
                double gradForTok = gradients.getSoftmaxLayer()[tokNum][slot];

                NetworkMatrices plusEPS = purturb(network.getMatrices(), EmbeddingTypes.SOFTMAX, tokNum, slot, eps);
                NetworkMatrices negEPS = purturb(network.getMatrices(), EmbeddingTypes.SOFTMAX, tokNum, slot, -eps);
                double[] plusPurturb = output(plusEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                        network.getNumOfPosLayers());
                double[] negPurturb = output(negEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                        network.getNumOfPosLayers());

                double diff = -(plusPurturb[goldLabel] - negPurturb[goldLabel]) / (2 * eps);
                System.out.println(gradForTok + "\t" + diff);
                assert Math.abs(gradForTok - diff) <= 0.00001;
            }
        }
    }

    @Test
    public void TestSoftMaxBiasGradients() throws Exception {
        MLPNetwork network = new MLPNetwork(100, 8, 100, 10, 10, 4, 4);
        NeuralTrainingInstance randomInstance = getRandomInstance(100, 10, 10);
        int goldLabel = randomInstance.gold();
        NetworkMatrices gradients = cost(network, randomInstance);

        double eps = 0.000001;
        for (int i = 0; i < network.getSoftmaxLayerDim(); i++) {
            int tokNum = i;
            double gradForTok = gradients.getSoftmaxLayerBias()[tokNum];

            NetworkMatrices plusEPS = purturb(network.getMatrices(), EmbeddingTypes.SOFTMAXBIAS, tokNum, -1, eps);
            NetworkMatrices negEPS = purturb(network.getMatrices(), EmbeddingTypes.SOFTMAXBIAS, tokNum, -1, -eps);
            double[] plusPurturb = output(plusEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                    network.getNumOfPosLayers());
            double[] negPurturb = output(negEPS, randomInstance.getFeatures(), network.getNumOfWordLayers(),
                    network.getNumOfPosLayers());

            double diff = -(plusPurturb[goldLabel] - negPurturb[goldLabel]) / (2 * eps);
            System.out.println(gradForTok + "\t" + diff);
            assert Math.abs(gradForTok - diff) <= 0.00001;
        }
    }

    private NeuralTrainingInstance getRandomInstance(int nW, int nP, int nL) {
        Random random = new Random(0);
        int[] features = new int[49];
        for (int i = 0; i < 19; i++)
            features[i] = random.nextInt(nW);
        for (int i = 19; i < 38; i++)
            features[i] = random.nextInt(nP);
        for (int i = 38; i < 49; i++)
            features[i] = random.nextInt(nL);
        int[] label = new int[2 * (nL + 1)];
        label[random.nextInt(2 * (nL + 1))] = 1;

        NeuralTrainingInstance instance = new NeuralTrainingInstance(features, label);
        return instance;
    }

    /**
     * Same as cost function copied here
     */
    private NetworkMatrices cost(MLPNetwork mlpNetwork, NeuralTrainingInstance instance) throws Exception {
        int[] features = instance.getFeatures();
        int[] label = instance.getLabel();
        NetworkMatrices gradients = new NetworkMatrices(
                mlpNetwork.getNumOfWords(), mlpNetwork.getWordEmbedDim(), mlpNetwork.getNumOfPos(), mlpNetwork.getPosEmbeddingDim(),
                mlpNetwork.getNumOfDepLabels(), mlpNetwork.getLabelEmbeddingSize(), mlpNetwork.getHiddenLayerDim(),
                mlpNetwork.getHiddenLayerIntDim(), mlpNetwork.getSoftmaxLayerDim());

        double[] hidden = new double[mlpNetwork.getHiddenLayerDim()];

        final double[][] softmaxLayer = mlpNetwork.getMatrices().getSoftmaxLayer();
        final double[] softmaxLayerBias = mlpNetwork.getMatrices().getSoftmaxLayerBias();
        final double[][] hiddenLayer = mlpNetwork.getMatrices().getHiddenLayer();
        final double[] hiddenLayerBias = mlpNetwork.getMatrices().getHiddenLayerBias();
        final double[][] wordEmbeddings = mlpNetwork.getMatrices().getWordEmbedding();
        final double[][] posEmbeddings = mlpNetwork.getMatrices().getPosEmbedding();
        final double[][] labelEmbeddings = mlpNetwork.getMatrices().getLabelEmbedding();

        int offset = 0;
        for (int j = 0; j < features.length; j++) {
            int tok = features[j];
            final double[] embedding;
            if (j < mlpNetwork.getNumOfWordLayers())
                embedding = wordEmbeddings[tok];
            else if (j < mlpNetwork.getNumOfWordLayers() + mlpNetwork.getNumOfPosLayers())
                embedding = posEmbeddings[tok];
            else
                embedding = labelEmbeddings[tok];

            for (int h = 0; h < hidden.length; h++) {
                for (int k = 0; k < embedding.length; k++) {
                    hidden[h] += hiddenLayer[h][offset + k] * embedding[k];
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
                for (int h = 0; h < reluHidden.length; h++) {
                    probs[i] += softmaxLayer[i][h] * reluHidden[h];
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

        double[] reluGradW = new double[reluHidden.length];
        for (int i = 0; i < probs.length; i++) {
            if (label[i] >= 0) {
                double delta = (-label[i] + probs[i]);
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
        for (int index = 0; index < mlpNetwork.getNumOfWordLayers(); index++) {
            double[] embeddings = wordEmbeddings[features[index]];
            for (int h = 0; h < reluHidden.length; h++) {
                for (int k = 0; k < embeddings.length; k++) {
                    gradients.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, hiddenGrad[h] * embeddings[k]);
                    gradients.modify(EmbeddingTypes.WORD, features[index], k, hiddenGrad[h] * hiddenLayer[h][offset + k]);
                }
            }
            offset += embeddings.length;
        }

        for (int index = mlpNetwork.getNumOfWordLayers(); index < mlpNetwork
                .getNumOfWordLayers() + mlpNetwork.getNumOfPosLayers(); index++) {
            double[] embeddings = posEmbeddings[features[index]];
            for (int h = 0; h < reluHidden.length; h++) {
                for (int k = 0; k < embeddings.length; k++) {
                    gradients.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, hiddenGrad[h] * embeddings[k]);
                    gradients.modify(EmbeddingTypes.POS, features[index], k, hiddenGrad[h] * hiddenLayer[h][offset + k]);
                }
            }
            offset += embeddings.length;
        }
        for (int index = mlpNetwork.getNumOfWordLayers() + mlpNetwork
                .getNumOfPosLayers(); index < mlpNetwork.getNumOfWordLayers() +
                mlpNetwork.getNumOfPosLayers() + mlpNetwork.getNumOfDepLayers(); index++) {
            double[] embeddings = labelEmbeddings[features[index]];
            for (int h = 0; h < reluHidden.length; h++) {
                for (int k = 0; k < embeddings.length; k++) {
                    gradients.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, hiddenGrad[h] * embeddings[k]);
                    gradients.modify(EmbeddingTypes.DEPENDENCY, features[index], k, hiddenGrad[h] * hiddenLayer[h][offset + k]);
                }
            }
            offset += embeddings.length;
        }
        return gradients;
    }

    private double[] output(NetworkMatrices matrices, int[] feats, int numberOfWordEmbeddingLayers, int numberOfPosEmbeddingLayers) {
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
            double pt = 0;
            if (j < numberOfWordEmbeddingLayers) {
                embedding = wordEmbeddings;
            } else if (j < numberOfWordEmbeddingLayers + numberOfPosEmbeddingLayers) {
                embedding = posEmbeddings;
            } else {
                embedding = labelEmbeddings;
            }
            for (int i = 0; i < hidden.length; i++) {
                for (int k = 0; k < embedding[0].length; k++) {
                    hidden[i] += hiddenLayer[i][offset + k] * embedding[tok][k];
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

    private NetworkMatrices purturb(NetworkMatrices matrices, EmbeddingTypes type, int tokNum, int slotNum, double eps) {
        NetworkMatrices cloned = matrices.clone();
        if (type == EmbeddingTypes.WORD) {
            cloned.getWordEmbedding()[tokNum][slotNum] += eps;
        } else if (type == EmbeddingTypes.POS) {
            cloned.getPosEmbedding()[tokNum][slotNum] += eps;
        } else if (type == EmbeddingTypes.DEPENDENCY) {
            cloned.getLabelEmbedding()[tokNum][slotNum] += eps;
        } else if (type == EmbeddingTypes.HIDDENLAYER) {
            cloned.getHiddenLayer()[tokNum][slotNum] += eps;
        } else if (type == EmbeddingTypes.SOFTMAX) {
            cloned.getSoftmaxLayer()[tokNum][slotNum] += eps;
        } else if (type == EmbeddingTypes.HIDDENLAYERBIAS) {
            cloned.getHiddenLayerBias()[tokNum] += eps;
        } else if (type == EmbeddingTypes.SOFTMAXBIAS) {
            cloned.getSoftmaxLayerBias()[tokNum] += eps;
        }
        return cloned;
    }
}
