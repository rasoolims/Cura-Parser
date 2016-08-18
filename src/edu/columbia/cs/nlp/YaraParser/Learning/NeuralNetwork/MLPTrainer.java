package edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork;

import edu.columbia.cs.nlp.YaraParser.Accessories.Options;
import edu.columbia.cs.nlp.YaraParser.Accessories.Utils;
import edu.columbia.cs.nlp.YaraParser.Learning.Updater.*;
import edu.columbia.cs.nlp.YaraParser.Learning.Updater.Enums.UpdaterType;
import edu.columbia.cs.nlp.YaraParser.Structures.Enums.EmbeddingTypes;
import edu.columbia.cs.nlp.YaraParser.Structures.NeuralTrainingInstance;
import edu.columbia.cs.nlp.YaraParser.Structures.Pair;

import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 7/27/16
 * Time: 10:40 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class MLPTrainer {
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
    Random random;

    /**
     * Gradients
     */
    private NetworkMatrices gradients;
    private double regCoef;
    private double dropoutProb;
    private boolean regularizeAllLayers;

    public MLPTrainer(MLPNetwork net, Options options) throws Exception {
        this.net = net;
        random = new Random();
        this.dropoutProb = options.networkProperties.dropoutProbForHiddenLayer;
        if (options.updaterProperties.updaterType == UpdaterType.SGD)
            updater = new SGD(net, options.updaterProperties.learningRate, options.networkProperties.outputBiasTerm,
                    options.updaterProperties.momentum, options.updaterProperties.sgdType);
        else if (options.updaterProperties.updaterType == UpdaterType.ADAGRAD)
            updater = new Adagrad(net, options.updaterProperties.learningRate, options.networkProperties.outputBiasTerm, 1e-6);
        else if (options.updaterProperties.updaterType == UpdaterType.ADAM)
            updater = new Adam(net, options.updaterProperties.learningRate, options.networkProperties.outputBiasTerm, 0.9, 0.9999, 1e-8);
        else if (options.updaterProperties.updaterType == UpdaterType.ADAMAX)
            updater = new AdaMax(net, options.updaterProperties.learningRate, options.networkProperties.outputBiasTerm, 0.9, 0.9999, 1e-8);
        else
            throw new Exception("Updater not implemented");
        this.regCoef = options.networkProperties.regularization;
        this.numThreads = options.generalProperties.numOfThreads;
        this.regularizeAllLayers = options.networkProperties.regualarizeAllLayers;
        executor = Executors.newFixedThreadPool(numThreads);
        pool = new ExecutorCompletionService<>(executor);
    }

    private void regularizeWithL2() throws Exception {
        double regCost = 0.0;
        final double[][] hiddenLayer = net.matrices.getHiddenLayer();
        final double[] hiddenLayerBias = net.matrices.getHiddenLayerBias();

        for (int h = 0; h < hiddenLayer.length; h++) {
            if (regularizeAllLayers) {
                regCost += Math.pow(hiddenLayerBias[h], 2);
                gradients.modify(EmbeddingTypes.HIDDENLAYERBIAS, h, -1, regCoef * 2 * hiddenLayerBias[h]);
            }
            for (int j = 0; j < hiddenLayer[h].length; j++) {
                regCost += Math.pow(hiddenLayer[h][j], 2);
                gradients.modify(EmbeddingTypes.HIDDENLAYER, h, j, regCoef * 2 * hiddenLayer[h][j]);
            }
        }

        final double[][] secondHiddenLayer = net.matrices.getSecondHiddenLayer();
        final double[] secondHiddenLayerBias = net.matrices.getSecondHiddenLayerBias();

        if (secondHiddenLayer != null) {
            for (int h = 0; h < secondHiddenLayer.length; h++) {
                if (regularizeAllLayers) {
                    regCost += Math.pow(secondHiddenLayerBias[h], 2);
                    gradients.modify(EmbeddingTypes.SECONDHIDDENLAYERBIAS, h, -1, regCoef * 2 * secondHiddenLayerBias[h]);
                }
                for (int j = 0; j < secondHiddenLayer[h].length; j++) {
                    regCost += Math.pow(secondHiddenLayer[h][j], 2);
                    gradients.modify(EmbeddingTypes.SECONDHIDDENLAYER, h, j, regCoef * 2 * secondHiddenLayer[h][j]);
                }
            }
        }


        if (regularizeAllLayers) {
            final double[][] wordEmbeddings = net.matrices.getWordEmbedding();
            final double[][] posEmbeddings = net.matrices.getPosEmbedding();
            final double[][] labelEmbeddings = net.matrices.getLabelEmbedding();
            final double[][] softmaxLayer = net.matrices.getSoftmaxLayer();
            final double[] softmaxLayerBias = net.matrices.getSoftmaxLayerBias();

            for (int i = 0; i < net.numWords; i++) {
                for (int j = 0; j < net.wordEmbedDim; j++) {
                    regCost += Math.pow(wordEmbeddings[i][j], 2);
                    gradients.modify(EmbeddingTypes.WORD, i, j, regCoef * 2 * wordEmbeddings[i][j]);
                }
            }

            for (int i = 0; i < net.numPos; i++) {
                for (int j = 0; j < net.posEmbedDim; j++) {
                    regCost += Math.pow(posEmbeddings[i][j], 2);
                    gradients.modify(EmbeddingTypes.POS, i, j, regCoef * 2 * posEmbeddings[i][j]);
                }
            }

            for (int i = 0; i < net.numDepLabels; i++) {
                for (int j = 0; j < net.depEmbedDim; j++) {
                    regCost += Math.pow(labelEmbeddings[i][j], 2);
                    gradients.modify(EmbeddingTypes.DEPENDENCY, i, j, regCoef * 2 * labelEmbeddings[i][j]);
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
        }
        cost += regCoef * regCost;
    }

    public double fit(List<NeuralTrainingInstance> instances, int iteration, boolean print) throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");
        DecimalFormat format4 = new DecimalFormat("##.0000");

        cost(instances);
        regularizeWithL2();
        updater.update(gradients);
        net.preCompute();

        double acc = correct / samples;
        if (print) {
            System.out.println(getCurrentTimeStamp() + " ---  iteration " + iteration + " --- size " +
                    samples + " --- Correct " + format.format(100. * acc) + " --- cost: " + format4.format(cost / samples));
            cost = 0;
            samples = 0;
            correct = 0;
        }
        return acc;
    }

    public void cost(List<NeuralTrainingInstance> instances) throws Exception {
        submitThreads(instances);
        mergeCosts(instances);
        samples += instances.size();
    }

    private void submitThreads(List<NeuralTrainingInstance> instances) {
        int chunkSize = instances.size() / numThreads;
        int s = 0;
        int e = Math.min(instances.size(), chunkSize);
        for (int i = 0; i < Math.min(instances.size(), numThreads); i++) {
            pool.submit(new CostThread(instances.subList(s, e), instances.size()));
            s = e;
            e = Math.min(instances.size(), e + chunkSize);
        }
    }

    private void mergeCosts(List<NeuralTrainingInstance> instances) throws Exception {
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
        if (Double.isNaN(cost))
            throw new Exception("cost is not a number");
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

    private void backPropSavedGradients(NetworkMatrices g, double[][][] savedGradients, HashSet<Integer>[] wordsSeen) throws Exception {
        int offset = 0;
        double[][] hiddenLayer = net.matrices.getHiddenLayer();
        double[][] wE = net.matrices.getWordEmbedding();
        double[][] pE = net.matrices.getPosEmbedding();
        double[][] lE = net.matrices.getLabelEmbedding();
        for (int index = 0; index < net.getNumWordLayers(); index++) {
            for (int tok : wordsSeen[index]) {
                int id = net.maps.preComputeMap[index].get(tok);
                double[] embedding = wE[tok];
                for (int h = 0; h < hiddenLayer.length; h++) {
                    double delta = savedGradients[index][id][h];
                    for (int k = 0; k < embedding.length; k++) {
                        g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, delta * embedding[k]);
                        g.modify(EmbeddingTypes.WORD, tok, k, delta * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += net.wordEmbedDim;
        }

        for (int index = net.getNumWordLayers(); index < net.getNumWordLayers() + net.getNumPosLayers(); index++) {
            for (int tok = 0; tok < net.numPos; tok++) {
                double[] embedding = pE[tok];
                for (int h = 0; h < hiddenLayer.length; h++) {
                    double delta = savedGradients[index][tok][h];
                    for (int k = 0; k < embedding.length; k++) {
                        g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, delta * embedding[k]);
                        g.modify(EmbeddingTypes.POS, tok, k, delta * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += net.posEmbedDim;
        }

        for (int index = net.getNumWordLayers() + net.getNumPosLayers();
             index < net.getNumWordLayers() + net.getNumPosLayers() + net.getNumDepLayers(); index++) {
            for (int tok = 0; tok < net.numDepLabels; tok++) {
                double[] embedding = lE[tok];
                for (int h = 0; h < hiddenLayer.length; h++) {
                    double delta = savedGradients[index][tok][h];
                    for (int k = 0; k < embedding.length; k++) {
                        g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, delta * embedding[k]);
                        g.modify(EmbeddingTypes.DEPENDENCY, tok, k, delta * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += net.depEmbedDim;
        }
    }

    public Pair<Double, Double> calculateCost(List<NeuralTrainingInstance> instances, int batchSize, NetworkMatrices g, double[][][] savedGradients)
            throws Exception {
        double cost = 0;
        double correct = 0;
        HashSet<Integer>[] featuresSeen = Utils.createHashSetArray(net.getNumWordLayers());

        for (NeuralTrainingInstance instance : instances) {
            int[] features = instance.getFeatures();
            int[] label = instance.getLabel();
            double[] hidden = new double[net.getHiddenLayerDim()];
            final double[][] softmaxLayer = net.getMatrices().getSoftmaxLayer();
            final double[] softmaxLayerBias = net.getMatrices().getSoftmaxLayerBias();
            final double[][] hiddenLayer = net.getMatrices().getHiddenLayer();
            final double[] hiddenLayerBias = net.getMatrices().getHiddenLayerBias();
            final double[][] wordEmbeddings = net.getMatrices().getWordEmbedding();
            final double[][] posEmbeddings = net.getMatrices().getPosEmbedding();
            final double[][] labelEmbeddings = net.getMatrices().getLabelEmbedding();
            final double[][] secondHiddenLayer = net.getMatrices().getSecondHiddenLayer();
            final double[] secondHiddenLayerBias = net.getMatrices().getSecondHiddenLayerBias();

            HashSet<Integer> hiddenNodesToUse = applyDropout(hidden);
            HashSet<Integer> secondHiddenNodesToUse = secondHiddenLayer == null ? null : applyDropout(new double[secondHiddenLayer.length]);
            int offset = 0;
            for (int j = 0; j < features.length; j++) {
                int tok = features[j];
                final double[] embedding;
                if (j < net.getNumWordLayers())
                    embedding = wordEmbeddings[tok];
                else if (j < net.getNumWordLayers() + net.getNumPosLayers())
                    embedding = posEmbeddings[tok];
                else
                    embedding = labelEmbeddings[tok];

                if (net.saved != null && (j >= net.getNumWordLayers() || net.maps.preComputeMap[j].containsKey(tok))) {
                    int id = tok;
                    if (j < net.getNumWordLayers())
                        id = net.maps.preComputeMap[j].get(tok);
                    double[] s = net.saved[j][id];
                    for (int h : hiddenNodesToUse) {
                        hidden[h] += s[h];
                    }
                } else {
                    for (int h : hiddenNodesToUse) {
                        for (int k = 0; k < embedding.length; k++) {
                            hidden[h] += hiddenLayer[h][offset + k] * embedding[k];
                        }
                    }
                }
                offset += embedding.length;
            }

            double[] activationHidden = new double[hidden.length];
            for (int h : hiddenNodesToUse) {
                hidden[h] += hiddenLayerBias[h];
                activationHidden[h] = net.activation.activate(hidden[h]);
            }

            double[] secondHidden = secondHiddenLayer != null ? new double[secondHiddenLayer.length] : null;
            double[] secondActivationHidden = secondHiddenLayer != null ? new double[secondHiddenLayer.length] : null;
            if (secondHiddenLayer != null) {
                for (int h1 : secondHiddenNodesToUse) {
                    for (int h2 : hiddenNodesToUse) {
                        secondHidden[h1] += secondHiddenLayer[h1][h2] * activationHidden[h2];
                    }
                }

                for (int h1 : secondHiddenNodesToUse) {
                    secondHidden[h1] += secondHiddenLayerBias[h1];
                    secondActivationHidden[h1] = net.activation.activate(secondHidden[h1]);
                }
            }

            double[] lastActivation = secondHiddenLayer != null ? secondActivationHidden : activationHidden;
            HashSet<Integer> hiddenToUse = secondHiddenLayer != null ? secondHiddenNodesToUse : hiddenNodesToUse;

            int argmax = -1;
            int gold = -1;
            double sum = 0;
            double[] probs = new double[softmaxLayerBias.length];
            for (int i = 0; i < probs.length; i++) {
                if (label[i] >= 0) {
                    if (label[i] == 1)
                        gold = i;
                    for (int h : hiddenToUse) {
                        probs[i] += softmaxLayer[i][h] * lastActivation[h];
                    }
                    probs[i] += softmaxLayerBias[i];
                    if (argmax < 0 || probs[i] > probs[argmax])
                        argmax = i;
                }
            }

            // We do this to decrease the chance of NAN creation.
            double max = probs[argmax];
            int numActiveLabels = 0;
            for (int i = 0; i < probs.length; i++) {
                if (label[i] >= 0) {
                    // We do this to decrease the chance of NAN creation.
                    probs[i] = Math.exp(probs[i] - max);
                    sum += probs[i];
                    numActiveLabels++;
                }
            }

            for (int i = 0; i < probs.length; i++) {
                if (label[i] >= 0) {
                    if (sum != 0)
                        probs[i] /= sum;
                    else
                        probs[i] = 1.0 / numActiveLabels;
                }
            }

            cost -= Math.log(probs[gold]);
            if (Double.isInfinite(cost))
                throw new Exception("Infinite cost!");
            if (argmax == gold)
                correct += 1.0;

            double[] delta = new double[probs.length];
            for (int i = 0; i < probs.length; i++) {
                if (label[i] >= 0)
                    delta[i] = (-label[i] + probs[i]) / batchSize;
            }
            if (secondHiddenLayer == null) {
                gradientWithOneHiddenLayer(g, savedGradients, featuresSeen, features, label, hidden, softmaxLayer, hiddenLayer,
                        wordEmbeddings, hiddenNodesToUse, activationHidden, delta);
            } else {
                gradientWithTwoHiddenLayers(g, savedGradients, featuresSeen, features, label, hidden, secondHidden, softmaxLayer,
                        hiddenLayer, secondHiddenLayer, wordEmbeddings, hiddenNodesToUse, secondHiddenNodesToUse, activationHidden,
                        secondActivationHidden, delta);
            }
        }

        backPropSavedGradients(g, savedGradients, featuresSeen);
        return new Pair<>(cost, correct);
    }

    private void gradientWithOneHiddenLayer(NetworkMatrices g, double[][][] savedGradients, HashSet<Integer>[] featuresSeen, int[]
            features, int[] label, double[] hidden, double[][] softmaxLayer, double[][] hiddenLayer, double[][] wordEmbeddings, HashSet<Integer>
                                                    hiddenNodesToUse, double[] activationHidden, double[] delta) throws Exception {
        int offset;
        double[] activationGradW = new double[activationHidden.length];
        for (int i = 0; i < delta.length; i++) {
            if (label[i] >= 0) {
                g.modify(EmbeddingTypes.SOFTMAXBIAS, i, -1, delta[i]);
                for (int h : hiddenNodesToUse) {
                    g.modify(EmbeddingTypes.SOFTMAX, i, h, delta[i] * activationHidden[h]);
                    activationGradW[h] += delta[i] * softmaxLayer[i][h];
                }
            }
        }

        double[] hiddenGrad = new double[hidden.length];
        for (int h : hiddenNodesToUse) {
            hiddenGrad[h] = net.activation.gradient(hidden[h], activationGradW[h]);
            g.modify(EmbeddingTypes.HIDDENLAYERBIAS, h, -1, hiddenGrad[h]);
        }

        offset = 0;
        for (int index = 0; index < net.getNumWordLayers(); index++) {
            if (net.maps.preComputeMap[index].containsKey(features[index])) {
                featuresSeen[index].add(features[index]);
                int id = net.maps.preComputeMap[index].get(features[index]);
                for (int h : hiddenNodesToUse) {
                    savedGradients[index][id][h] += hiddenGrad[h];
                }
            } else {
                double[] embeddings = wordEmbeddings[features[index]];
                for (int h : hiddenNodesToUse) {
                    for (int k = 0; k < embeddings.length; k++) {
                        g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, hiddenGrad[h] * embeddings[k]);
                        g.modify(EmbeddingTypes.WORD, features[index], k, hiddenGrad[h] * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += net.wordEmbedDim;
        }

        for (int index = net.getNumWordLayers(); index < net.getNumWordLayers() + net.getNumPosLayers() + net.getNumDepLayers(); index++) {
            for (int h : hiddenNodesToUse) {
                savedGradients[index][features[index]][h] += hiddenGrad[h];
            }
        }
    }

    private void gradientWithTwoHiddenLayers(NetworkMatrices gradient, double[][][] savedGradients, HashSet<Integer>[] featuresSeen,
                                             int[] features, int[] label, double[] hiddenInput, double[] hiddenInput2, double[][] softmaxLayer,
                                             double[][] hiddenLayer, double[][] secondHidden, double[][] wordEmbeddings,
                                             HashSet<Integer> hiddenNodesToUse, HashSet<Integer> secondHiddenNodesToUse,
                                             double[] H1, double[] H2, double[] delta) throws Exception {
        int offset;
        double[] dL_dH2 = new double[H2.length];
        for (int i = 0; i < delta.length; i++) {
            if (label[i] >= 0) {
                gradient.modify(EmbeddingTypes.SOFTMAXBIAS, i, -1, delta[i]);
                for (int k : secondHiddenNodesToUse) {
                    gradient.modify(EmbeddingTypes.SOFTMAX, i, k, delta[i] * H2[k]);
                    dL_dH2[k] += delta[i] * softmaxLayer[i][k];
                }
            }
        }

        double[] df_dH2_input = new double[hiddenInput2.length];
        for (int k : secondHiddenNodesToUse) {
            df_dH2_input[k] = net.activation.gradient(hiddenInput2[k], dL_dH2[k]);
            gradient.modify(EmbeddingTypes.SECONDHIDDENLAYERBIAS, k, -1, df_dH2_input[k]);
        }

        double[] dL_dH1 = new double[H1.length];
        for (int k : secondHiddenNodesToUse) {
            for (int g : hiddenNodesToUse) {
                gradient.modify(EmbeddingTypes.SECONDHIDDENLAYER, k, g, df_dH2_input[k] * H1[g]);
                dL_dH1[g] += df_dH2_input[k] * secondHidden[k][g];
            }
        }


        double[] df_dH1_input = new double[hiddenInput.length];
        for (int h : hiddenNodesToUse) {
            df_dH1_input[h] = net.activation.gradient(hiddenInput[h], dL_dH1[h]);
            gradient.modify(EmbeddingTypes.HIDDENLAYERBIAS, h, -1, df_dH1_input[h]);
        }

        offset = 0;
        for (int index = 0; index < net.getNumWordLayers(); index++) {
            if (net.maps.preComputeMap[index].containsKey(features[index])) {
                featuresSeen[index].add(features[index]);
                int id = net.maps.preComputeMap[index].get(features[index]);
                for (int h : hiddenNodesToUse) {
                    savedGradients[index][id][h] += df_dH1_input[h];
                }
            } else {
                double[] embeddings = wordEmbeddings[features[index]];
                for (int h : hiddenNodesToUse) {
                    for (int k = 0; k < embeddings.length; k++) {
                        gradient.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, df_dH1_input[h] * embeddings[k]);
                        gradient.modify(EmbeddingTypes.WORD, features[index], k, df_dH1_input[h] * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += net.wordEmbedDim;
        }

        for (int index = net.getNumWordLayers(); index < net.getNumWordLayers() + net.getNumPosLayers() + net.getNumDepLayers(); index++) {
            for (int h : hiddenNodesToUse) {
                savedGradients[index][features[index]][h] += df_dH1_input[h];
            }
        }
    }


    private HashSet<Integer> applyDropout(double[] hidden) {
        HashSet<Integer> hiddenNodesToUse = new HashSet<>();
        for (int h = 0; h < hidden.length; h++) {
            if (dropoutProb <= 0 || random.nextDouble() >= dropoutProb)
                hiddenNodesToUse.add(h);
        }
        return hiddenNodesToUse;
    }

    public NetworkMatrices getGradients() {
        return gradients;
    }

    public class CostThread implements Callable<Pair<Pair<Double, Double>, NetworkMatrices>> {
        List<NeuralTrainingInstance> instances;
        int batchSize;
        NetworkMatrices g;
        double[][][] savedGradients;

        public CostThread(List<NeuralTrainingInstance> instances, int batchSize) {
            this.instances = instances;
            this.batchSize = batchSize;
            g = new NetworkMatrices(net.numWords, net.wordEmbedDim, net.numPos, net.posEmbedDim, net.numDepLabels, net.depEmbedDim,
                    net.hiddenLayerDim, net.hiddenLayerIntDim, net.secondHiddenLayerDim, net.softmaxLayerDim);
            savedGradients = instantiateSavedGradients();
        }

        @Override
        public Pair<Pair<Double, Double>, NetworkMatrices> call() throws Exception {
            Pair<Double, Double> costValue = calculateCost(instances, batchSize, g, savedGradients);
            return new Pair<>(costValue, g);
        }

        private double[][][] instantiateSavedGradients() {
            double[][][] savedGradients = new double[net.getNumWordLayers() + net.getNumPosLayers() + net.getNumDepLayers()][][];
            for (int i = 0; i < net.getNumWordLayers(); i++)
                savedGradients[i] = new double[net.maps.preComputeMap[i].size()][net.hiddenLayerDim];
            for (int i = net.getNumWordLayers(); i < net.getNumWordLayers() + net.getNumPosLayers(); i++)
                savedGradients[i] = new double[net.numPos][net.hiddenLayerDim];
            for (int i = net.getNumWordLayers() + net.getNumPosLayers();
                 i < net.getNumWordLayers() + net.getNumPosLayers() + net.getNumDepLayers(); i++)
                savedGradients[i] = new double[net.numDepLabels][net.hiddenLayerDim];
            return savedGradients;
        }
    }
}
