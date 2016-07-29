package YaraParser.Learning;

import YaraParser.Structures.EmbeddingTypes;
import YaraParser.Structures.NeuralTrainingInstance;

import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/27/16
 * Time: 10:40 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class MLPClassifier {
    public int[][] confusionMatrix;
    MLPNetwork mlpNetwork;

    /**
     * Keep track of loss function
     */
    double cost = 0.0;
    double correct = 0.0;
    int samples = 0;

    /**
     * Gradients
     */
    private NetworkMatrices gradiants;

    /**
     * Gradient histories for momentum update
     */
    private double[][] wordEmbeddingGradientHistory;
    private double[][] posEmbeddingGradientHistory;
    private double[][] labelEmbeddingGradientHistory;
    private double[][] hiddenLayerGradientHistory;
    private double[] hiddenLayerBiasGradientHistory;
    private double[][] softmaxLayerGradientHistory;
    private double[] softmaxLayerBiasGradientHistory;

    private double momentum;
    private double learningRate;
    private double regularizerCoefficient;

    public MLPClassifier(MLPNetwork mlpNetwork, double momentum, double learningRate, double regularizerCoefficient) {
        this.mlpNetwork = mlpNetwork;
        this.momentum = momentum;
        this.learningRate = learningRate;
        this.regularizerCoefficient = regularizerCoefficient;

        wordEmbeddingGradientHistory = new double[mlpNetwork.numOfWords][mlpNetwork.wordEmbeddingSize];
        posEmbeddingGradientHistory = new double[mlpNetwork.numOfPos][mlpNetwork.posEmbeddingSize];
        labelEmbeddingGradientHistory = new double[mlpNetwork.numOfDependencyLabels][mlpNetwork.labelEmbeddingSize];
        hiddenLayerGradientHistory = new double[mlpNetwork.hiddenLayerSize][mlpNetwork.hiddenLayerIntSize];
        hiddenLayerBiasGradientHistory = new double[mlpNetwork.hiddenLayerSize];
        softmaxLayerGradientHistory = new double[mlpNetwork.softmaxLayerSize][mlpNetwork.hiddenLayerSize];
        softmaxLayerBiasGradientHistory = new double[mlpNetwork.softmaxLayerSize];
    }

    public void cost(ArrayList<NeuralTrainingInstance> instances, int batchSize) throws Exception {
        samples += batchSize;
        gradiants = new NetworkMatrices(mlpNetwork.numOfWords, mlpNetwork.wordEmbeddingSize, mlpNetwork.numOfPos, mlpNetwork.posEmbeddingSize,
                mlpNetwork
                .numOfDependencyLabels, mlpNetwork.labelEmbeddingSize, mlpNetwork.hiddenLayerSize,
                mlpNetwork.hiddenLayerIntSize, mlpNetwork.softmaxLayerSize);

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
                    for (int i = 0; i < hidden.length; i++) {
                        hidden[i] += mlpNetwork.saved[j][id][i];
                    }
                } else {
                    for (int i = 0; i < hidden.length; i++) {
                        for (int k = 0; k < embedding.length; k++) {
                            hidden[i] += hiddenLayer[i][offset + k] * embedding[k];
                        }
                    }
                }
                offset += embedding.length;
            }

            double[] reluHidden = new double[hidden.length];
            for (int i = 0; i < hidden.length; i++) {
                hidden[i] += hiddenLayerBias[i];
                //relu
                reluHidden[i] = Math.max(0, hidden[i]);
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
            confusionMatrix[gold][argmax] += 1;

            double[] reluGradW = new double[reluHidden.length];
            for (int i = 0; i < probs.length; i++) {
                if (label[i] >= 0) {
                    double delta = (-label[i] + probs[i]) / batchSize;
                    gradiants.modify(EmbeddingTypes.SOFTMAXBIAS, i, -1, delta);
                    for (int h = 0; h < reluHidden.length; h++) {
                        gradiants.modify(EmbeddingTypes.SOFTMAX, i, h, delta * reluHidden[h]);
                        reluGradW[h] += delta * softmaxLayer[i][h];
                    }
                }
            }

            double[] hiddenGrad = new double[hidden.length];
            for (int h = 0; h < reluHidden.length; h++) {
                hiddenGrad[h] = (reluHidden[h] == 0. ? 0 : reluGradW[h]);
                gradiants.modify(EmbeddingTypes.HIDDENLAYERBIAS, h, -1, hiddenGrad[h]);
            }

            offset = 0;
            for (int index = 0; index < mlpNetwork.numberOfWordEmbeddingLayers; index++) {
                double[] embeddings = wordEmbeddings[features[index]];
                for (int h = 0; h < reluHidden.length; h++) {
                    for (int k = 0; k < embeddings.length; k++) {
                        gradiants.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, hiddenGrad[h] * embeddings[k]);
                        gradiants.modify(EmbeddingTypes.WORD, features[index], k, hiddenGrad[h] * hiddenLayer[h][offset + k]);
                    }
                }
                offset += embeddings.length;
            }

            for (int index = mlpNetwork.numberOfWordEmbeddingLayers; index < mlpNetwork
                    .numberOfWordEmbeddingLayers + mlpNetwork.numberOfPosEmbeddingLayers; index++) {
                double[] embeddings = posEmbeddings[features[index]];
                for (int h = 0; h < reluHidden.length; h++) {
                    for (int k = 0; k < embeddings.length; k++) {
                        gradiants.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, hiddenGrad[h] * embeddings[k]);
                        gradiants.modify(EmbeddingTypes.POS, features[index], k, hiddenGrad[h] * hiddenLayer[h][offset + k]);
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
                        gradiants.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, hiddenGrad[h] * embeddings[k]);
                        gradiants.modify(EmbeddingTypes.DEPENDENCY, features[index], k, hiddenGrad[h] * hiddenLayer[h][offset + k]);
                    }
                }
                offset += embeddings.length;
            }
        }
    }

    private void regularizeWithL2() throws Exception {
        double regCost = 0.0;
        final double[][] hiddenLayer = mlpNetwork.matrices.getHiddenLayer();
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < hiddenLayer[i].length; j++) {
                regCost += Math.pow(hiddenLayer[i][j], 2);
                gradiants.modify(EmbeddingTypes.HIDDENLAYER, i, j, regularizerCoefficient * 2 * hiddenLayer[i][j]);
            }
        }
        cost += regularizerCoefficient * regCost;
    }

    public void fit(ArrayList<NeuralTrainingInstance> instances, int iteration, boolean print) throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        cost(instances, instances.size());
        regularizeWithL2();
        update();
        mlpNetwork.preCompute();

        if (print) {
            System.out.println(getCurrentTimeStamp() + " ---  iteration " + iteration + " --- size " +
                    samples + " --- Correct " + format.format(100. * correct / samples) + " --- cost: " + format.format(cost / samples));
            cost = 0;
            samples = 0;
            correct = 0;
        }
    }

    private void update() throws Exception {
        double[][] wordEmbeddingGradient = gradiants.getWordEmbedding();
        for (int i = 0; i < mlpNetwork.numOfWords; i++) {
            for (int j = 0; j < mlpNetwork.wordEmbeddingSize; j++) {
                wordEmbeddingGradientHistory[i][j] = momentum * wordEmbeddingGradientHistory[i][j] - wordEmbeddingGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.WORD, i, j, learningRate * wordEmbeddingGradientHistory[i][j]);
            }
        }

        double[][] posEmbeddingGradient = gradiants.getPosEmbedding();
        for (int i = 0; i < mlpNetwork.numOfPos; i++) {
            for (int j = 0; j < mlpNetwork.posEmbeddingSize; j++) {
                posEmbeddingGradientHistory[i][j] = momentum * posEmbeddingGradientHistory[i][j] - posEmbeddingGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.POS, i, j, learningRate * posEmbeddingGradientHistory[i][j]);
            }
        }

        double[][] labelEmbeddingGradient = gradiants.getLabelEmbedding();
        for (int i = 0; i < mlpNetwork.numOfDependencyLabels; i++) {
            for (int j = 0; j < mlpNetwork.labelEmbeddingSize; j++) {
                labelEmbeddingGradientHistory[i][j] = momentum * labelEmbeddingGradientHistory[i][j] - labelEmbeddingGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.DEPENDENCY, i, j, learningRate * labelEmbeddingGradientHistory[i][j]);
            }
        }

        double[][] hiddenLayerGradient = gradiants.getHiddenLayer();
        for (int i = 0; i < mlpNetwork.hiddenLayerSize; i++) {
            for (int j = 0; j < mlpNetwork.hiddenLayerIntSize; j++) {
                hiddenLayerGradientHistory[i][j] = momentum * hiddenLayerGradientHistory[i][j] - hiddenLayerGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.HIDDENLAYER, i, j, learningRate * hiddenLayerGradientHistory[i][j]);
            }
        }

        double[] hiddenLayerBiasGradient = gradiants.getHiddenLayerBias();
        for (int i = 0; i < mlpNetwork.hiddenLayerSize; i++) {
            hiddenLayerBiasGradientHistory[i] = momentum * hiddenLayerBiasGradientHistory[i] - hiddenLayerBiasGradient[i];
            mlpNetwork.modify(EmbeddingTypes.HIDDENLAYERBIAS, i, -1, learningRate * hiddenLayerBiasGradientHistory[i]);
        }

        double[][] softmaxLayerGradient = gradiants.getSoftmaxLayer();
        for (int i = 0; i < mlpNetwork.softmaxLayerSize; i++) {
            for (int j = 0; j < mlpNetwork.hiddenLayerSize; j++) {
                softmaxLayerGradientHistory[i][j] = momentum * softmaxLayerGradientHistory[i][j] - softmaxLayerGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.SOFTMAX, i, j, learningRate * softmaxLayerGradientHistory[i][j]);
            }
        }

        double[] softmaxLayerBiasGradient = gradiants.getSoftmaxLayerBias();
        for (int i = 0; i < mlpNetwork.softmaxLayerSize; i++) {
            softmaxLayerBiasGradientHistory[i] = momentum * softmaxLayerBiasGradientHistory[i] - softmaxLayerBiasGradient[i];
            mlpNetwork.modify(EmbeddingTypes.SOFTMAXBIAS, i, -1, learningRate * softmaxLayerBiasGradientHistory[i]);
        }
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public String getCurrentTimeStamp() {
        return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date());
    }
}
