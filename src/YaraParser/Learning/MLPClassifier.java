package YaraParser.Learning;

import YaraParser.Accessories.Pair;
import YaraParser.Learning.Updater.*;
import YaraParser.Structures.EmbeddingTypes;
import YaraParser.Structures.NeuralTrainingInstance;

import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.concurrent.*;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/27/16
 * Time: 10:40 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class MLPClassifier {
    MLPNetwork mlpNetwork;

    /**
     * for multi-threading
     */
    ExecutorService executor;
    CompletionService<Pair<Pair<Double, Double>, Pair<NetworkMatrices, double[][][]>>> pool;
    int numOfThreads;

    /**
     * Keep track of loss function
     */
    double cost = 0.0;
    double correct = 0.0;
    int samples = 0;
    Updater updater;
    /**
     * Gradients
     */
    private NetworkMatrices gradients;
    private double regularizerCoefficient;

    public MLPClassifier(MLPNetwork mlpNetwork, UpdaterType updaterType, double momentum, double learningRate, double regularizerCoefficient, int
            numOfThreads) throws Exception {
        this.mlpNetwork = mlpNetwork;
        if (updaterType == UpdaterType.SGD)
            updater = new SgdWithMomentumUpdater(mlpNetwork, learningRate, momentum);
        else if (updaterType == UpdaterType.ADAGRAD)
            updater = new Adagrad(mlpNetwork, learningRate, 1e-6);
        else
            updater = new Adam(mlpNetwork, learningRate, 0.9, 0.9999, 1e-8);
        this.regularizerCoefficient = regularizerCoefficient;
        this.numOfThreads = numOfThreads;
        executor = Executors.newFixedThreadPool(numOfThreads);
        pool = new ExecutorCompletionService<>(executor);
    }

    private void regularizeWithL2() throws Exception {
        double regCost = 0.0;
        final double[][] wordEmbeddings = mlpNetwork.matrices.getWordEmbedding();
        final double[][] posEmbeddings = mlpNetwork.matrices.getPosEmbedding();
        final double[][] labelEmbeddings = mlpNetwork.matrices.getLabelEmbedding();
        final double[][] hiddenLayer = mlpNetwork.matrices.getHiddenLayer();
        final double[] hiddenLayerBias = mlpNetwork.matrices.getHiddenLayerBias();
        final double[][] softmaxLayer = mlpNetwork.matrices.getSoftmaxLayer();
        final double[] softmaxLayerBias = mlpNetwork.matrices.getSoftmaxLayerBias();

        for (int i = 0; i < mlpNetwork.numOfWords; i++) {
            for (int j = 0; j < mlpNetwork.wordEmbedDim; j++) {
                regCost += Math.pow(wordEmbeddings[i][j], 2);
                gradients.modify(EmbeddingTypes.WORD, i, j, regularizerCoefficient * 2 * wordEmbeddings[i][j]);
            }
        }

        for (int i = 0; i < mlpNetwork.numOfPos; i++) {
            for (int j = 0; j < mlpNetwork.posEmbeddingDim; j++) {
                regCost += Math.pow(posEmbeddings[i][j], 2);
                gradients.modify(EmbeddingTypes.POS, i, j, regularizerCoefficient * 2 * posEmbeddings[i][j]);
            }
        }

        for (int i = 0; i < mlpNetwork.numOfDepLabels; i++) {
            for (int j = 0; j < mlpNetwork.labelEmbeddingSize; j++) {
                regCost += Math.pow(labelEmbeddings[i][j], 2);
                gradients.modify(EmbeddingTypes.DEPENDENCY, i, j, regularizerCoefficient * 2 * labelEmbeddings[i][j]);
            }
        }

        for (int h = 0; h < hiddenLayer.length; h++) {
            regCost += Math.pow(hiddenLayerBias[h], 2);
            gradients.modify(EmbeddingTypes.HIDDENLAYERBIAS, h, -1, regularizerCoefficient * 2 * hiddenLayerBias[h]);
            for (int j = 0; j < hiddenLayer[h].length; j++) {
                regCost += Math.pow(hiddenLayer[h][j], 2);
                gradients.modify(EmbeddingTypes.HIDDENLAYER, h, j, regularizerCoefficient * 2 * hiddenLayer[h][j]);
            }
        }

        for (int i = 0; i < softmaxLayer.length; i++) {
            regCost += Math.pow(softmaxLayerBias[i], 2);
            gradients.modify(EmbeddingTypes.SOFTMAXBIAS, i, -1, regularizerCoefficient * 2 * softmaxLayerBias[i]);
            for (int h = 0; h < softmaxLayer[i].length; h++) {
                regCost += Math.pow(softmaxLayer[i][h], 2);
                gradients.modify(EmbeddingTypes.SOFTMAX, i, h, regularizerCoefficient * 2 * softmaxLayer[i][h]);
            }
        }
        cost += regularizerCoefficient * regCost;
    }

    public void fit(ArrayList<NeuralTrainingInstance> instances, int iteration, boolean print) throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");
        DecimalFormat format4 = new DecimalFormat("##.0000");

        cost(instances);
        regularizeWithL2();
        updater.update(gradients);
        mlpNetwork.preCompute();

        if (print) {
            System.out.println(getCurrentTimeStamp() + " ---  iteration " + iteration + " --- size " +
                    samples + " --- Correct " + format.format(100. * correct / samples) + " --- cost: " + format4.format(cost / samples));
            cost = 0;
            samples = 0;
            correct = 0;
        }
    }

    private void cost(ArrayList<NeuralTrainingInstance> instances) throws Exception {
        int chunkSize = instances.size() / numOfThreads;
        int s = 0;
        int e = Math.min(instances.size(), chunkSize);
        for (int i = 0; i < Math.min(instances.size(), numOfThreads); i++) {
            pool.submit(new CostThread(instances, instances.size(), s, e));
            s = e;
            e = Math.min(instances.size(), e + chunkSize);
        }
        double[][] hiddenLayer = mlpNetwork.matrices.getHiddenLayer();

        Pair<Pair<Double, Double>, Pair<NetworkMatrices, double[][][]>> firstResult = pool.take().get();
        gradients = firstResult.second.first;
        backPropSavedGradients(hiddenLayer, firstResult.second.second);


        cost += firstResult.first.first;
        correct += firstResult.first.second;

        for (int i = 1; i < Math.min(instances.size(), numOfThreads); i++) {
            Pair<Pair<Double, Double>, Pair<NetworkMatrices, double[][][]>> result = pool.take().get();
            gradients.mergeMatricesInPlace(result.second.first);
            backPropSavedGradients(hiddenLayer, firstResult.second.second);


            cost += result.first.first;
            correct += result.first.second;
        }

        samples += instances.size();
    }

    private void backPropSavedGradients(double[][] hiddenLayer, double[][][] savedGradients) throws Exception {
        for (int tok : mlpNetwork.maps.preComputeMap.keySet()) {
            int id = mlpNetwork.maps.preComputeMap.get(tok);
            double[] embedding = mlpNetwork.matrices.getWordEmbedding()[tok];
            int offset = 0;
            for (int index = 0; index < mlpNetwork.numOfWordLayers; index++) {
                for (int h = 0; h < savedGradients[index][id].length; h++) {
                    double delta = savedGradients[index][id][h];
                    for (int k = 0; k < embedding.length; k++) {
                        gradients.modify(EmbeddingTypes.HIDDENLAYER, h, offset, delta * embedding[k]);
                        gradients.modify(EmbeddingTypes.WORD, tok, k, delta * hiddenLayer[h][offset]);
                    }
                }
                offset += embedding.length;
            }
        }

        for (int tok = 0; tok < mlpNetwork.numOfPos; tok++) {
            double[] embedding = mlpNetwork.matrices.getPosEmbedding()[tok];
            int offset = mlpNetwork.numOfWordLayers * mlpNetwork.wordEmbedDim;
            for (int index = 0; index < mlpNetwork.numOfPosLayers; index++) {
                for (int h = 0; h < savedGradients[index][tok].length; h++) {
                    double delta = savedGradients[index][tok][h];
                    for (int k = 0; k < embedding.length; k++) {
                        gradients.modify(EmbeddingTypes.HIDDENLAYER, h, offset, delta * embedding[k]);
                        gradients.modify(EmbeddingTypes.POS, tok, k, delta * hiddenLayer[h][offset]);
                    }
                }
                offset += embedding.length;
            }
        }

        for (int tok = 0; tok < mlpNetwork.numOfDepLabels; tok++) {
            double[] embedding = mlpNetwork.matrices.getLabelEmbedding()[tok];
            int offset = mlpNetwork.numOfWordLayers * mlpNetwork.wordEmbedDim + mlpNetwork.numOfDepLayers * mlpNetwork.posEmbeddingDim;
            for (int index = 0; index < mlpNetwork.numOfDepLayers; index++) {
                for (int h = 0; h < savedGradients[index][tok].length; h++) {
                    double delta = savedGradients[index][tok][h];
                    for (int k = 0; k < embedding.length; k++) {
                        gradients.modify(EmbeddingTypes.HIDDENLAYER, h, offset, delta * embedding[k]);
                        gradients.modify(EmbeddingTypes.DEPENDENCY, tok, k, delta * hiddenLayer[h][offset]);
                    }
                }
                offset += embedding.length;
            }
        }
    }

    public double getLearningRate() {
        return updater.getLearningRate();
    }

    public void setLearningRate(double learningRate) {
        updater.setLearningRate(learningRate);
    }

    public String getCurrentTimeStamp() {
        return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date());
    }

    public void shutDownLiveThreads() {
        boolean isTerminated = executor.isTerminated();
        while (!isTerminated) {
            executor.shutdownNow();
            isTerminated = executor.isTerminated();
        }
    }

    public class CostThread implements Callable<Pair<Pair<Double, Double>, Pair<NetworkMatrices, double[][][]>>> {
        ArrayList<NeuralTrainingInstance> instances;
        int batchSize;
        int s;
        int e;

        public CostThread(ArrayList<NeuralTrainingInstance> instances, int batchSize, int s, int e) {
            this.instances = instances;
            this.batchSize = batchSize;
            this.s = s;
            this.e = e;
        }

        @Override
        public Pair<Pair<Double, Double>, Pair<NetworkMatrices, double[][][]>> call() throws Exception {
            NetworkMatrices gradients = new NetworkMatrices(mlpNetwork.numOfWords, mlpNetwork.wordEmbedDim, mlpNetwork.numOfPos,
                    mlpNetwork.posEmbeddingDim, mlpNetwork.numOfDepLabels, mlpNetwork.labelEmbeddingSize, mlpNetwork.hiddenLayerDim,
                    mlpNetwork.hiddenLayerIntDim, mlpNetwork.softmaxLayerDim);
            double[][][] savedGradients = instantiateSavedGradients();

            double cost = 0;
            double correct = 0;

            for (int instanceIndex = s; instanceIndex < e; instanceIndex++) {
                NeuralTrainingInstance instance = instances.get(instanceIndex);
                int[] features = instance.getFeatures();
                int[] label = instance.getLabel();

                double[] hidden = new double[mlpNetwork.hiddenLayerDim];

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
                    if (j < mlpNetwork.numOfWordLayers)
                        embedding = wordEmbeddings[tok];
                    else if (j < mlpNetwork.numOfWordLayers + mlpNetwork.numOfPosLayers)
                        embedding = posEmbeddings[tok];
                    else
                        embedding = labelEmbeddings[tok];

                    if (mlpNetwork.saved != null && (j >= mlpNetwork.numOfWordLayers || mlpNetwork.maps.preComputeMap.containsKey(tok))) {
                        int id = tok;
                        if (j < mlpNetwork.numOfWordLayers)
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
                for (int index = 0; index < mlpNetwork.numOfWordLayers; index++) {
                    if (mlpNetwork.maps.preComputeMap.containsKey(features[index])) {
                        for (int h = 0; h < reluHidden.length; h++) {
                            savedGradients[index][mlpNetwork.maps.preComputeMap.get(features[index])][h] += hiddenGrad[h];
                        }
                    } else {
                        double[] embeddings = wordEmbeddings[features[index]];
                        for (int h = 0; h < reluHidden.length; h++) {
                            for (int k = 0; k < embeddings.length; k++) {
                                gradients.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, hiddenGrad[h] * embeddings[k]);
                                gradients.modify(EmbeddingTypes.WORD, features[index], k, hiddenGrad[h] * hiddenLayer[h][offset + k]);
                            }
                        }
                    }
                    offset += mlpNetwork.wordEmbedDim;
                }

                for (int index = mlpNetwork.numOfWordLayers; index < mlpNetwork.numOfWordLayers + mlpNetwork.numOfPosLayers; index++) {
                    for (int h = 0; h < reluHidden.length; h++) {
                        savedGradients[index][features[index]][h] += hiddenGrad[h];
                    }
                }
                for (int index = mlpNetwork.numOfWordLayers + mlpNetwork.numOfPosLayers; index < mlpNetwork.numOfWordLayers +
                        mlpNetwork.numOfPosLayers + mlpNetwork.numOfDepLayers; index++) {
                    for (int h = 0; h < reluHidden.length; h++) {
                        savedGradients[index][features[index]][h] += hiddenGrad[h];
                    }
                }
            }

            return new Pair<>(new Pair<>(cost, correct), new Pair<>(gradients, savedGradients));
        }

        private double[][][] instantiateSavedGradients() {
            double[][][] savedGradients =
                    new double[mlpNetwork.numOfWordLayers + mlpNetwork.numOfPosLayers + mlpNetwork.numOfDepLayers]
                            [mlpNetwork.maps.preComputeMap.size()][mlpNetwork.hiddenLayerDim];
            for (int i = 0; i < mlpNetwork.numOfWordLayers; i++)
                savedGradients[i] = new double[mlpNetwork.maps.preComputeMap.size()][mlpNetwork.hiddenLayerDim];
            for (int i = mlpNetwork.numOfWordLayers; i < mlpNetwork.numOfWordLayers + mlpNetwork
                    .numOfPosLayers; i++)
                savedGradients[i] = new double[mlpNetwork.numOfPos][mlpNetwork.hiddenLayerDim];
            for (int i = mlpNetwork.numOfWordLayers + mlpNetwork.numOfPosLayers;
                 i < mlpNetwork.numOfWordLayers + mlpNetwork.numOfPosLayers + mlpNetwork.numOfDepLayers; i++)
                savedGradients[i] = new double[mlpNetwork.numOfDepLabels][mlpNetwork.hiddenLayerDim];
            return savedGradients;
        }
    }
}
