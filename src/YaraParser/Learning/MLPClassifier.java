package YaraParser.Learning;

import YaraParser.Accessories.Pair;
import YaraParser.Learning.Updater.*;
import YaraParser.Structures.EmbeddingTypes;
import YaraParser.Structures.NeuralTrainingInstance;

import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.*;
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
    Random random;

    /**
     * Gradients
     */
    private NetworkMatrices gradients;
    private double regCoef;
    private double dropoutProb;

    public MLPClassifier(MLPNetwork net, UpdaterType updaterType, double momentum, double learningRate, double regCoef, int numThreads, double
            dropoutProb)
            throws Exception {
        this.net = net;
        random = new Random();
        this.dropoutProb = dropoutProb;
        if (updaterType == UpdaterType.SGD)
            updater = new SgdWithMomentumUpdater(net, learningRate, momentum);
        else if (updaterType == UpdaterType.ADAGRAD)
            updater = new Adagrad(net, learningRate, 1e-6);
        else if (updaterType == UpdaterType.ADAM)
            updater = new Adam(net, learningRate, 0.9, 0.9999, 1e-8);
        else if (updaterType == UpdaterType.ADAMAX)
            updater = new AdaMax(net, learningRate, 0.9, 0.9999, 1e-8);
        else
            throw new Exception("Updater not implemented");
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

    public void cost(ArrayList<NeuralTrainingInstance> instances) throws Exception {
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

    private void backPropSavedGradients(NetworkMatrices g, double[][][] savedGradients)
            throws Exception {
        int offset = 0;
        double[][] hiddenLayer = net.matrices.getHiddenLayer();
        double[][] wE = net.matrices.getWordEmbedding();
        double[][] pE = net.matrices.getPosEmbedding();
        double[][] lE = net.matrices.getLabelEmbedding();
        for (int index = 0; index < net.numWordLayers; index++) {
            for (int tok : net.maps.preComputeMap.keySet()) {
                int id = net.maps.preComputeMap.get(tok);
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

        for (int index = net.numWordLayers; index < net.numWordLayers + net.numPosLayers; index++) {
            for (int tok = 0; tok < savedGradients[index].length; tok++) {
                double[] embedding = pE[tok];
                for (int h = 0; h < hiddenLayer.length; h++) {
                    double delta = savedGradients[index][tok][h];
                    for (int k = 0; k < embedding.length; k++) {
                        g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, delta * embedding[k]);
                        g.modify(EmbeddingTypes.POS, tok, k, delta * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += net.posEmbeddingDim;
        }

        for (int index = net.numWordLayers + net.numPosLayers; index < net.numWordLayers + net.numPosLayers + net.numDepLayers; index++) {
            for (int tok = 0; tok < savedGradients[index].length; tok++) {
                double[] embedding = lE[tok];
                for (int h = 0; h < hiddenLayer.length; h++) {
                    double delta = savedGradients[index][tok][h];
                    for (int k = 0; k < embedding.length; k++) {
                        g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, delta * embedding[k]);
                        g.modify(EmbeddingTypes.DEPENDENCY, tok, k, delta * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += net.labelEmbedDim;
        }
    }

    public Pair<Double, Double> calculateCost(List<NeuralTrainingInstance> instances, int batchSize, NetworkMatrices g, double[][][] savedGradients)
            throws Exception {
        double cost = 0;
        double correct = 0;

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

            HashSet<Integer> hiddenNodesToUse = new HashSet<>();
            if (dropoutProb > 0) {
                for (int i = 0; i < hidden.length; i++) {
                    if (dropoutProb <= 0 || random.nextDouble() >= dropoutProb)
                        hiddenNodesToUse.add(i);
                }
            }

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

                if (net.saved != null && (j >= net.numWordLayers || net.maps.preComputeMap.containsKey(tok))) {
                    int id = tok;
                    if (j < net.numWordLayers)
                        id = net.maps.preComputeMap.get(tok);
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

            double[] reluHidden = new double[hidden.length];
            for (int h : hiddenNodesToUse) {
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
                    for (int h : hiddenNodesToUse) {
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
            double[] delta = new double[probs.length];
            for (int i = 0; i < probs.length; i++) {
                if (label[i] >= 0) {
                    delta[i] = (-label[i] + probs[i]) / batchSize;
                    g.modify(EmbeddingTypes.SOFTMAXBIAS, i, -1, delta[i]);
                    for (int h : hiddenNodesToUse) {
                        g.modify(EmbeddingTypes.SOFTMAX, i, h, delta[i] * reluHidden[h]);
                        reluGradW[h] += delta[i] * softmaxLayer[i][h];
                    }
                }
            }

            double[] hiddenGrad = new double[hidden.length];
            for (int h : hiddenNodesToUse) {
                hiddenGrad[h] = (reluHidden[h] == 0. ? 0 : reluGradW[h]);
                g.modify(EmbeddingTypes.HIDDENLAYERBIAS, h, -1, hiddenGrad[h]);
            }

            offset = 0;
            for (int index = 0; index < net.getNumWordLayers(); index++) {
                if (net.maps.preComputeMap.containsKey(features[index])) {
                    int id = net.maps.preComputeMap.get(features[index]);
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

            for (int index = net.getNumWordLayers(); index < net.getNumWordLayers() + net.getNumPosLayers() + net.getNumDepLayers(); index++)
                for (int h : hiddenNodesToUse) {
                    savedGradients[index][features[index]][h] += hiddenGrad[h];
                }
        }

        backPropSavedGradients(g, savedGradients);
        return new Pair<>(cost, correct);
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
            g = new NetworkMatrices(net.numWords, net.wordEmbedDim, net.numPos, net.posEmbeddingDim, net.numDepLabels, net.labelEmbedDim,
                    net.hiddenLayerDim, net.hiddenLayerIntDim, net.softmaxLayerDim);
            savedGradients = instantiateSavedGradients();
        }


        @Override
        public Pair<Pair<Double, Double>, NetworkMatrices> call() throws Exception {
            Pair<Double, Double> costValue = calculateCost(instances, batchSize, g, savedGradients);
            return new Pair<>(costValue, g);
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
