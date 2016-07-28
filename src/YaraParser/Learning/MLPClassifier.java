package YaraParser.Learning;

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
    private double[][] wordEmbeddingGradient;
    private double[][] posEmbeddingGradient;
    private double[][] labelEmbeddingGradient;
    private double[][] hiddenLayerGradient;
    private double[] hiddenLayerBiasGradient;
    private double[][] softmaxLayerGradient;
    private double[] softmaxLayerBiasGradient;
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

        wordEmbeddingGradientHistory = new double[mlpNetwork.wordEmbeddings.length][mlpNetwork.wordEmbeddings[0]
                .length];
        posEmbeddingGradientHistory = new double[mlpNetwork.posEmbeddings.length][mlpNetwork.posEmbeddings[0].length];
        labelEmbeddingGradientHistory = new double[mlpNetwork.labelEmbeddings.length][mlpNetwork.labelEmbeddings[0]
                .length];
        hiddenLayerGradientHistory = new double[mlpNetwork.hiddenLayer.length][];
        for (int i = 0; i < hiddenLayerGradientHistory.length; i++) {
            hiddenLayerGradientHistory[i] = new double[mlpNetwork.hiddenLayer[i].length];
        }
        hiddenLayerBiasGradientHistory = new double[mlpNetwork.hiddenLayerBias.length];
        softmaxLayerGradientHistory = new double[mlpNetwork.softmaxLayer.length][mlpNetwork.softmaxLayer[0].length];
        softmaxLayerBiasGradientHistory = new double[mlpNetwork.softmaxLayerBias.length];
    }

    public void cost(ArrayList<NeuralTrainingInstance> instances, int batchSize) {
        samples+= batchSize;
        wordEmbeddingGradient = new double[mlpNetwork.wordEmbeddings.length][mlpNetwork.wordEmbeddings[0].length];
        posEmbeddingGradient = new double[mlpNetwork.posEmbeddings.length][mlpNetwork.posEmbeddings[0].length];
        labelEmbeddingGradient = new double[mlpNetwork.labelEmbeddings.length][mlpNetwork.labelEmbeddings[0].length];
        hiddenLayerGradient = new double[mlpNetwork.hiddenLayer.length][];
        for (int i = 0; i < hiddenLayerGradient.length; i++) {
            hiddenLayerGradient[i] = new double[mlpNetwork.hiddenLayer[i].length];
        }
        hiddenLayerBiasGradient = new double[mlpNetwork.hiddenLayerBias.length];
        softmaxLayerGradient = new double[mlpNetwork.softmaxLayer.length][mlpNetwork.softmaxLayer[0].length];
        softmaxLayerBiasGradient = new double[mlpNetwork.softmaxLayerBias.length];

        for (NeuralTrainingInstance instance : instances) {
            int[] features = instance.getFeatures();
            int label[] = instance.getLabel();

            double[] hidden = new double[mlpNetwork.hiddenLayer.length];

            int offset = 0;
            for (int j = 0; j < features.length; j++) {
                int tok = features[j];
                double[][] embedding;
                if (j < mlpNetwork.numberOfWordEmbeddingLayers)
                    embedding = mlpNetwork.wordEmbeddings;
                else if (j < mlpNetwork.numberOfWordEmbeddingLayers + mlpNetwork.numberOfPosEmbeddingLayers)
                    embedding = mlpNetwork.posEmbeddings;
                else embedding = mlpNetwork.labelEmbeddings;

                if (mlpNetwork.saved != null && (j >= mlpNetwork.numberOfWordEmbeddingLayers || mlpNetwork.maps
                        .preComputeMap.containsKey(tok))) {
                    int id = tok;
                    if (j < mlpNetwork.numberOfWordEmbeddingLayers)
                        id = mlpNetwork.maps.preComputeMap.get(tok);
                    for (int i = 0; i < hidden.length; i++) {
                        hidden[i] += mlpNetwork.saved[j][id][i];
                    }
                } else {
                    for (int i = 0; i < hidden.length; i++) {
                        for (int k = 0; k < embedding[0].length; k++) {
                            hidden[i] += mlpNetwork.hiddenLayer[i][offset + k] * embedding[tok][k];
                        }
                    }
                }
                offset += embedding[0].length;
            }

            double[] reluHidden = new double[hidden.length];
            for (int i = 0; i < hidden.length; i++) {
                hidden[i] += mlpNetwork.hiddenLayerBias[i];
                //relu
                reluHidden[i] = Math.max(0, hidden[i]);
            }

            int argmax = -1;
            int gold = 0;
            double sum = 0;
            double[] scores = new double[mlpNetwork.softmaxLayerBias.length];
            for (int i = 0; i < scores.length; i++) {
                if (label[i] >= 0) {
                    if (label[i] == 1)
                        gold = i;
                    for (int j = 0; j < reluHidden.length; j++) {
                        scores[i] += mlpNetwork.softmaxLayer[i][j] * reluHidden[j];
                    }

                    scores[i] += mlpNetwork.softmaxLayerBias[i];
                    scores[i] = Math.exp(scores[i]);
                    sum += scores[i];

                    if (argmax < 0 || scores[i] > scores[argmax])
                        argmax = i;
                }
            }

            for (int i = 0; i < scores.length; i++) {
                scores[i] /= sum;
            }

            cost += -Math.log(scores[gold]);
            if (argmax == gold)
                correct += 1.0 ;
            confusionMatrix[gold][argmax] += 1;

            double[] reluGradW = new double[reluHidden.length];
            for (int i = 0; i < scores.length; i++) {
                if (label[i] >= 0) {
                    double delta = (-label[i] + scores[i]) / batchSize;
                    softmaxLayerBiasGradient[i] += delta;
                    for (int h = 0; h < reluHidden.length; h++) {
                        softmaxLayerGradient[i][h] += delta * reluHidden[i];
                        reluGradW[h] += delta * mlpNetwork.softmaxLayer[i][h];
                        hiddenLayerBiasGradient[h] += (reluHidden[h] == 0. ? 0 : delta);
                    }
                }
            }

            double[] hiddenGrad = new double[hidden.length];
            for (int h = 0; h < reluHidden.length; h++) {
                hiddenGrad[h] = (reluHidden[h] == 0. ? 0 : reluGradW[h]);
            }

            offset = 0;
            /**
             // todo gradSave
             if (preMap.containsKey(index)) {
             int id = preMap.get(index);
             for (int nodeIndex : ls)
             gradSaved[id][nodeIndex] += gradHidden[nodeIndex];
             }
             */

            for (int index = 0; index < mlpNetwork.numberOfWordEmbeddingLayers; index++) {
                int id = features[index];
                for (int h = 0; h < reluHidden.length; h++) {
                    for (int k = 0; k < mlpNetwork.wordEmbeddings[0].length; k++) {
                        hiddenLayerGradient[h][offset + k] += hiddenGrad[h] * mlpNetwork.wordEmbeddings[id][k];
                        wordEmbeddingGradient[id][k] += hiddenGrad[h] * mlpNetwork.hiddenLayer[h][offset + k];
                    }
                }
                offset += mlpNetwork.wordEmbeddings[0].length;
            }

            for (int index = mlpNetwork.numberOfWordEmbeddingLayers; index < mlpNetwork
                    .numberOfWordEmbeddingLayers + mlpNetwork.numberOfPosEmbeddingLayers; index++) {
                int id = features[index];
                for (int h = 0; h < reluHidden.length; h++) {
                    for (int k = 0; k < mlpNetwork.posEmbeddings[0].length; k++) {
                        hiddenLayerGradient[h][offset + k] += hiddenGrad[h] * mlpNetwork.posEmbeddings[id][k];
                        posEmbeddingGradient[id][k] += hiddenGrad[h] * mlpNetwork.hiddenLayer[h][offset + k];
                    }
                }
                offset += mlpNetwork.posEmbeddings[0].length;
            }
            for (int index = mlpNetwork.numberOfWordEmbeddingLayers + mlpNetwork
                    .numberOfPosEmbeddingLayers; index < mlpNetwork.numberOfWordEmbeddingLayers +
                    mlpNetwork.numberOfPosEmbeddingLayers + mlpNetwork.numberOfLabelEmbeddingLayers; index++) {
                int id = features[index];
                for (int h = 0; h < reluHidden.length; h++) {
                    for (int k = 0; k < mlpNetwork.labelEmbeddings[0].length; k++) {
                        hiddenLayerGradient[h][offset + k] += hiddenGrad[h] * mlpNetwork.labelEmbeddings[id][k];
                        labelEmbeddingGradient[id][k] += hiddenGrad[h] * mlpNetwork.hiddenLayer[h][offset + k];
                    }
                }
                offset += mlpNetwork.labelEmbeddings[0].length;
            }
        }

        double regCost = 0.0;
        for (int i = 0; i < mlpNetwork.hiddenLayer.length; i++) {
            for (int j = 0; j < mlpNetwork.hiddenLayer[i].length; j++) {
                regCost += Math.pow(mlpNetwork.hiddenLayer[i][j], 2);
                hiddenLayerGradient[i][j] += regularizerCoefficient * 2 * mlpNetwork.hiddenLayer[i][j];
            }
        }
        cost += regularizerCoefficient * regCost;
    }

    public void fit(ArrayList<NeuralTrainingInstance> instances, int iteration, boolean print) {
        // todo grad saved
        // todo multithread
        DecimalFormat format = new DecimalFormat("##.00");

        cost(instances, instances.size());
        update();
        mlpNetwork.preCompute();

        if (print) {
            System.out.println("Time " + getCurrentTimeStamp() + " ---  iteration " + iteration + " --- size " +
                    instances.size() + " --- Correct " + format.format(100. * correct/samples) + " --- cost: " + format
                    .format(cost/samples));
            cost = 0;
            samples = 0;
            correct = 0;
        }
    }

    private void update() {
        for (int i = 0; i < mlpNetwork.wordEmbeddings.length; i++) {
            for (int j = 0; j < mlpNetwork.wordEmbeddings[i].length; j++) {
                wordEmbeddingGradientHistory[i][j] = momentum * wordEmbeddingGradientHistory[i][j] -
                        wordEmbeddingGradient[i][j];
                mlpNetwork.wordEmbeddings[i][j] += learningRate * wordEmbeddingGradientHistory[i][j];
            }
        }

        for (int i = 0; i < mlpNetwork.posEmbeddings.length; i++) {
            for (int j = 0; j < mlpNetwork.posEmbeddings[i].length; j++) {
                posEmbeddingGradientHistory[i][j] = momentum * posEmbeddingGradientHistory[i][j] -
                        posEmbeddingGradient[i][j];
                mlpNetwork.posEmbeddings[i][j] += learningRate * posEmbeddingGradientHistory[i][j];
            }
        }

        for (int i = 0; i < mlpNetwork.labelEmbeddings.length; i++) {
            for (int j = 0; j < mlpNetwork.labelEmbeddings[i].length; j++) {
                labelEmbeddingGradientHistory[i][j] = momentum * labelEmbeddingGradientHistory[i][j] -
                        labelEmbeddingGradient[i][j];
                mlpNetwork.labelEmbeddings[i][j] += learningRate * labelEmbeddingGradientHistory[i][j];
            }
        }

        for (int i = 0; i < mlpNetwork.hiddenLayer.length; i++) {
            for (int j = 0; j < mlpNetwork.hiddenLayer[i].length; j++) {
                hiddenLayerGradientHistory[i][j] = momentum * hiddenLayerGradientHistory[i][j] -
                        hiddenLayerGradient[i][j];
                mlpNetwork.hiddenLayer[i][j] += learningRate * hiddenLayerGradientHistory[i][j];
            }
        }

        for (int i = 0; i < mlpNetwork.hiddenLayerBias.length; i++) {
            hiddenLayerBiasGradientHistory[i] = momentum * hiddenLayerBiasGradientHistory[i] -
                    hiddenLayerBiasGradient[i];
            mlpNetwork.hiddenLayerBias[i] += learningRate * hiddenLayerBiasGradientHistory[i];
        }

        for (int i = 0; i < mlpNetwork.softmaxLayer.length; i++) {
            for (int j = 0; j < mlpNetwork.softmaxLayer[i].length; j++) {
                softmaxLayerGradientHistory[i][j] = momentum * softmaxLayerGradientHistory[i][j] -
                        softmaxLayerGradient[i][j];
                mlpNetwork.softmaxLayer[i][j] += learningRate * softmaxLayerGradientHistory[i][j];
            }
        }

        for (int i = 0; i < mlpNetwork.softmaxLayerBias.length; i++) {
            softmaxLayerBiasGradientHistory[i] = momentum * softmaxLayerBiasGradientHistory[i] -
                    softmaxLayerBiasGradient[i];
            mlpNetwork.softmaxLayerBias[i] += learningRate * softmaxLayerBiasGradientHistory[i];
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
