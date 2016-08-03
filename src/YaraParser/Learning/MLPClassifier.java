package YaraParser.Learning;

import YaraParser.Accessories.Pair;
import YaraParser.Learning.Updater.*;
import YaraParser.Structures.EmbeddingTypes;
import YaraParser.Structures.NeuralTrainingInstance;

import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.*;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/27/16
 * Time: 10:40 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class MLPClassifier {
    MLPNetwork net;

    /**
     * for multi-threading
     */
    ExecutorService executor;
    CompletionService<Pair<Pair<Double, Double>, NetworkMatrices>> pool;
    int numThreads;

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
    private double regCoef;

    public MLPClassifier(MLPNetwork net, UpdaterType updaterType, double momentum, double learningRate, double regCoef, int numThreads) throws
            Exception {
        this.net = net;
        if (updaterType == UpdaterType.SGD)
            updater = new SgdWithMomentumUpdater(net, learningRate, momentum);
        else if (updaterType == UpdaterType.ADAGRAD)
            updater = new Adagrad(net, learningRate, 1e-6);
        else
            updater = new Adam(net, learningRate, 0.9, 0.9999, 1e-8);
        this.regCoef = regCoef;
        this.numThreads = numThreads;
        executor = Executors.newFixedThreadPool(numThreads);
        pool = new ExecutorCompletionService<>(executor);
    }

    private void regularizeWithL2() throws Exception {
        double regCost = 0.0;
        final double[][] wordEmbeddings = net.matrices.getWordEmbedding();
        final double[][] posEmbeddings = net.matrices.getPosEmbedding();
        final double[][] labelEmbeddings = net.matrices.getLabelEmbedding();
        final double[][] hiddenLayer = net.matrices.getHiddenLayer();
        final double[] hiddenLayerBias = net.matrices.getHiddenLayerBias();
        final double[][] softmaxLayer = net.matrices.getSoftmaxLayer();
        final double[] softmaxLayerBias = net.matrices.getSoftmaxLayerBias();

        for (int i = 0; i < net.numWords; i++) {
            for (int j = 0; j < net.wordEmbedDim; j++) {
                regCost += Math.pow(wordEmbeddings[i][j], 2);
                gradients.modify(EmbeddingTypes.WORD, i, j, regCoef * 2 * wordEmbeddings[i][j]);
            }
        }

        for (int i = 0; i < net.numPos; i++) {
            for (int j = 0; j < net.posEmbeddingDim; j++) {
                regCost += Math.pow(posEmbeddings[i][j], 2);
                gradients.modify(EmbeddingTypes.POS, i, j, regCoef * 2 * posEmbeddings[i][j]);
            }
        }

        for (int i = 0; i < net.numDepLabels; i++) {
            for (int j = 0; j < net.labelEmbedDim; j++) {
                regCost += Math.pow(labelEmbeddings[i][j], 2);
                gradients.modify(EmbeddingTypes.DEPENDENCY, i, j, regCoef * 2 * labelEmbeddings[i][j]);
            }
        }

        for (int h = 0; h < hiddenLayer.length; h++) {
            regCost += Math.pow(hiddenLayerBias[h], 2);
            gradients.modify(EmbeddingTypes.HIDDENLAYERBIAS, h, -1, regCoef * 2 * hiddenLayerBias[h]);
            for (int j = 0; j < hiddenLayer[h].length; j++) {
                regCost += Math.pow(hiddenLayer[h][j], 2);
                gradients.modify(EmbeddingTypes.HIDDENLAYER, h, j, regCoef * 2 * hiddenLayer[h][j]);
            }
        }

        for (int i = 0; i < softmaxLayer.length; i++) {
            regCost += Math.pow(softmaxLayerBias[i], 2);
            gradients.modify(EmbeddingTypes.SOFTMAXBIAS, i, -1, regCoef * 2 * softmaxLayerBias[i]);
            for (int h = 0; h < softmaxLayer[i].length; h++) {
                regCost += Math.pow(softmaxLayer[i][h], 2);
                gradients.modify(EmbeddingTypes.SOFTMAX, i, h, regCoef * 2 * softmaxLayer[i][h]);
            }
        }
        cost += regCoef * regCost;
    }

    public void fit(ArrayList<NeuralTrainingInstance> instances, int iteration, boolean print) throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");
        DecimalFormat format4 = new DecimalFormat("##.0000");

        cost(instances);
        regularizeWithL2();
        updater.update(gradients);
        net.preCompute();

        if (print) {
            System.out.println(getCurrentTimeStamp() + " ---  iteration " + iteration + " --- size " +
                    samples + " --- Correct " + format.format(100. * correct / samples) + " --- cost: " + format4.format(cost / samples));
            cost = 0;
            samples = 0;
            correct = 0;
        }
    }

    private void cost(ArrayList<NeuralTrainingInstance> instances) throws Exception {
        submitThreads(instances);
        mergeCosts(instances);
        samples += instances.size();
    }

    private void submitThreads(ArrayList<NeuralTrainingInstance> instances) {
        int chunkSize = instances.size() / numThreads;
        int s = 0;
        int e = Math.min(instances.size(), chunkSize);
        for (int i = 0; i < Math.min(instances.size(), numThreads); i++) {
            pool.submit(new CostThread(instances.subList(s, e), instances.size()));
            s = e;
            e = Math.min(instances.size(), e + chunkSize);
        }
    }

    private void mergeCosts(ArrayList<NeuralTrainingInstance> instances) throws Exception {
        Pair<Pair<Double, Double>, NetworkMatrices> firstResult = pool.take().get();
        gradients = firstResult.second;

        cost += firstResult.first.first;
        correct += firstResult.first.second;

        for (int i = 1; i < Math.min(instances.size(), numThreads); i++) {
            Pair<Pair<Double, Double>, NetworkMatrices> result = pool.take().get();
            gradients.mergeMatricesInPlaceForNonSaved(result.second);
            cost += result.first.first;
            correct += result.first.second;
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

    public class CostThread implements Callable<Pair<Pair<Double, Double>, NetworkMatrices>> {
        List<NeuralTrainingInstance> instances;
        int batchSize;
        NetworkMatrices g;
        double[][][] savedGradients;

        public CostThread(List<NeuralTrainingInstance> instances, int batchSize) {
            this.instances = instances;
            this.batchSize = batchSize;
            g = new NetworkMatrices(net.numWords, net.wordEmbedDim, net.numPos, net.posEmbeddingDim, net.numDepLabels, net.labelEmbedDim,
                    net.hiddenLayerDim, net.hiddenLayerIntDim, net.softmaxLayerDim);
            savedGradients = instantiateSavedGradients();
        }

        private void backPropSavedGradients() throws Exception {
            int offset = 0;
            double[][] hiddenLayer = g.getHiddenLayer();
            double[][] wE = net.matrices.getWordEmbedding();
            double[][] pE = net.matrices.getPosEmbedding();
            double[][] lE = net.matrices.getLabelEmbedding();
            for (int index = 0; index < net.numWordLayers; index++) {
                for (int tok : net.maps.preComputeMap.keySet()) {
                    int id = net.maps.preComputeMap.get(tok);
                    double[] embedding = wE[tok];
                    for (int h = 0; h < savedGradients[index][id].length; h++) {
                        double delta = savedGradients[index][id][h];
                        for (int k = 0; k < embedding.length; k++) {
                            g.modify(EmbeddingTypes.HIDDENLAYER, h, offset, delta * embedding[k]);
                            g.modify(EmbeddingTypes.WORD, tok, k, delta * hiddenLayer[h][offset]);
                        }
                    }
                }
                offset += net.wordEmbedDim;
            }

            int plBorder = net.numWordLayers + net.numPosLayers;
            for (int index = net.numWordLayers; index < net.numWordLayers + net.numPosLayers; index++) {
                for (int tok = 0; tok < net.numPos; tok++) {
                    double[] embedding = index < plBorder ? pE[tok] : lE[tok];
                    for (int h = 0; h < savedGradients[index][tok].length; h++) {
                        double delta = savedGradients[index][tok][h];
                        for (int k = 0; k < embedding.length; k++) {
                            g.modify(EmbeddingTypes.HIDDENLAYER, h, offset, delta * embedding[k]);
                            g.modify(index < plBorder ? EmbeddingTypes.POS : EmbeddingTypes.DEPENDENCY, tok, k, delta * hiddenLayer[h][offset]);
                        }
                    }
                }
                offset += index < plBorder ? net.posEmbeddingDim : net.labelEmbedDim;
            }

        }

        @Override
        public Pair<Pair<Double, Double>, NetworkMatrices> call() throws Exception {
            double cost = 0;
            double correct = 0;

            for (NeuralTrainingInstance instance : instances) {
                int[] features = instance.getFeatures();
                int[] label = instance.getLabel();
                final double[][] softmaxLayer = net.matrices.getSoftmaxLayer();
                final double[] softmaxLayerBias = net.matrices.getSoftmaxLayerBias();
                final double[][] hiddenLayer = net.matrices.getHiddenLayer();
                final double[] hiddenLayerBias = net.matrices.getHiddenLayerBias();
                final double[][] wordEmbeddings = net.matrices.getWordEmbedding();
                final double[][] posEmbeddings = net.matrices.getPosEmbedding();
                final double[][] labelEmbeddings = net.matrices.getLabelEmbedding();

                double[] hidden = new double[net.hiddenLayerDim];
                double[] reluHidden = new double[hidden.length];
                double[] probs = new double[softmaxLayerBias.length];

                int offset = 0;
                for (int j = 0; j < features.length; j++) {
                    int tok = features[j];
                    final double[] embedding;
                    if (j < net.numWordLayers)
                        embedding = wordEmbeddings[tok];
                    else if (j < net.numWordLayers + net.numPosLayers)
                        embedding = posEmbeddings[tok];
                    else
                        embedding = labelEmbeddings[tok];

                    if (net.saved != null && (j >= net.numWordLayers || net.maps.preComputeMap.containsKey(tok))) {
                        int id = tok;
                        if (j < net.numWordLayers)
                            id = net.maps.preComputeMap.get(tok);
                        for (int h = 0; h < hidden.length; h++) {
                            hidden[h] += net.saved[j][id][h];
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

                for (int h = 0; h < hidden.length; h++) {
                    hidden[h] += hiddenLayerBias[h];
                    //relu
                    reluHidden[h] = Math.max(0, hidden[h]);
                }

                int argmax = -1;
                int gold = -1;
                double sum = 0;
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
                        g.modify(EmbeddingTypes.SOFTMAXBIAS, i, -1, delta);
                        for (int h = 0; h < reluHidden.length; h++) {
                            g.modify(EmbeddingTypes.SOFTMAX, i, h, delta * reluHidden[h]);
                            reluGradW[h] += delta * softmaxLayer[i][h];
                        }
                    }
                }

                double[] hiddenGrad = new double[hidden.length];
                for (int h = 0; h < reluHidden.length; h++) {
                    hiddenGrad[h] = (reluHidden[h] == 0. ? 0 : reluGradW[h]);
                    g.modify(EmbeddingTypes.HIDDENLAYERBIAS, h, -1, hiddenGrad[h]);
                }

                offset = 0;
                for (int index = 0; index < net.numWordLayers; index++) {
                    if (net.maps.preComputeMap.containsKey(features[index])) {
                        int id = net.maps.preComputeMap.get(features[index]);
                        for (int h = 0; h < reluHidden.length; h++) {
                            savedGradients[index][id][h] += hiddenGrad[h];
                        }
                    } else {
                        double[] embeddings = wordEmbeddings[features[index]];
                        for (int h = 0; h < reluHidden.length; h++) {
                            for (int k = 0; k < embeddings.length; k++) {
                                g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, hiddenGrad[h] * embeddings[k]);
                                g.modify(EmbeddingTypes.WORD, features[index], k, hiddenGrad[h] * hiddenLayer[h][offset + k]);
                            }
                        }
                    }
                    offset += net.wordEmbedDim;
                }

                // for label and pos
                for (int index = net.numWordLayers; index < savedGradients.length; index++) {
                    int id = features[index];
                    for (int h = 0; h < reluHidden.length; h++) {
                        savedGradients[index][id][h] += hiddenGrad[h];
                    }
                }
            }

            backPropSavedGradients();
            return new Pair<>(new Pair<>(cost, correct), g);
        }

        private double[][][] instantiateSavedGradients() {
            double[][][] savedGradients = new double[net.numWordLayers + net.numPosLayers + net.numDepLayers][][];
            for (int i = 0; i < net.numWordLayers; i++)
                savedGradients[i] = new double[net.maps.preComputeMap.size()][net.hiddenLayerDim];
            for (int i = net.numWordLayers; i < net.numWordLayers + net.numPosLayers; i++)
                savedGradients[i] = new double[net.numPos][net.hiddenLayerDim];
            for (int i = net.numWordLayers + net.numPosLayers;
                 i < net.numWordLayers + net.numPosLayers + net.numDepLayers; i++)
                savedGradients[i] = new double[net.numDepLabels][net.hiddenLayerDim];
            return savedGradients;
        }
    }
}
