package YaraParser.Learning;

import YaraParser.Accessories.Pair;
import YaraParser.Structures.EmbeddingTypes;
import YaraParser.Structures.NeuralTrainingInstance;

import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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
     * for multi-threading
     */
    ExecutorService executor;
    CompletionService<Pair<Pair<Double, Double>, NetworkMatrices>> pool;
    int numOfThreads;

    /**
     * Keep track of loss function
     */
    double cost = 0.0;
    double correct = 0.0;
    int samples = 0;

    /**
     * Gradients
     */
    private NetworkMatrices gradients;

    /**
     * Gradient histories for momentum update
     */
    private NetworkMatrices gradientHistory;

    private double momentum;
    private double learningRate;
    private double regularizerCoefficient;

    public MLPClassifier(MLPNetwork mlpNetwork, double momentum, double learningRate, double regularizerCoefficient, int numOfThreads) {
        this.mlpNetwork = mlpNetwork;
        this.momentum = momentum;
        this.learningRate = learningRate;
        this.regularizerCoefficient = regularizerCoefficient;
        gradientHistory = new NetworkMatrices(mlpNetwork.numOfWords, mlpNetwork.wordEmbeddingSize, mlpNetwork.numOfPos, mlpNetwork.posEmbeddingSize,
                mlpNetwork.numOfDependencyLabels, mlpNetwork.labelEmbeddingSize, mlpNetwork.hiddenLayerSize,
                mlpNetwork.hiddenLayerIntSize, mlpNetwork.softmaxLayerSize);
        this.numOfThreads = numOfThreads;
        executor = Executors.newFixedThreadPool(numOfThreads);
        pool = new ExecutorCompletionService<>(executor);
    }

    private void regularizeWithL2() throws Exception {
        double regCost = 0.0;
        final double[][] hiddenLayer = mlpNetwork.matrices.getHiddenLayer();
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < hiddenLayer[i].length; j++) {
                regCost += Math.pow(hiddenLayer[i][j], 2);
                gradients.modify(EmbeddingTypes.HIDDENLAYER, i, j, regularizerCoefficient * 2 * hiddenLayer[i][j]);
            }
        }
        cost += regularizerCoefficient * regCost;
    }

    public void fit(ArrayList<NeuralTrainingInstance> instances, int iteration, boolean print) throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");

        cost(instances);
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

    private void cost(ArrayList<NeuralTrainingInstance> instances) throws InterruptedException, java.util.concurrent.ExecutionException {
        int chunkSize = instances.size() / numOfThreads;
        int s = 0;
        int e = Math.min(instances.size(), chunkSize);
        for (int i = 0; i < Math.min(instances.size(), numOfThreads); i++) {
            pool.submit(new CostThread(mlpNetwork, instances.subList(s, e), instances.size()));
            s = e;
            e = Math.min(instances.size(), e + chunkSize);
        }

        Pair<Pair<Double, Double>, NetworkMatrices> firstResult = pool.take().get();
        gradients = firstResult.second;
        cost += firstResult.first.first;
        correct += firstResult.first.second;

        for (int i = 1; i < Math.min(instances.size(), numOfThreads); i++) {
            Pair<Pair<Double, Double>, NetworkMatrices> result = pool.take().get();
            gradients.mergeMatricesInPlace(result.second);
            cost += result.first.first;
            correct += result.first.second;
        }

        samples += instances.size();
    }

    private void update() throws Exception {
        double[][] wordEmbeddingGradient = gradients.getWordEmbedding();
        double[][] wordEmbeddingGradientHistory = gradientHistory.getWordEmbedding();
        for (int i = 0; i < mlpNetwork.numOfWords; i++) {
            for (int j = 0; j < mlpNetwork.wordEmbeddingSize; j++) {
                wordEmbeddingGradientHistory[i][j] = momentum * wordEmbeddingGradientHistory[i][j] - wordEmbeddingGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.WORD, i, j, learningRate * wordEmbeddingGradientHistory[i][j]);
            }
        }

        double[][] posEmbeddingGradient = gradients.getPosEmbedding();
        double[][] posEmbeddingGradientHistory = gradientHistory.getPosEmbedding();
        for (int i = 0; i < mlpNetwork.numOfPos; i++) {
            for (int j = 0; j < mlpNetwork.posEmbeddingSize; j++) {
                posEmbeddingGradientHistory[i][j] = momentum * posEmbeddingGradientHistory[i][j] - posEmbeddingGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.POS, i, j, learningRate * posEmbeddingGradientHistory[i][j]);
            }
        }

        double[][] labelEmbeddingGradient = gradients.getLabelEmbedding();
        double[][] labelEmbeddingGradientHistory = gradientHistory.getLabelEmbedding();
        for (int i = 0; i < mlpNetwork.numOfDependencyLabels; i++) {
            for (int j = 0; j < mlpNetwork.labelEmbeddingSize; j++) {
                labelEmbeddingGradientHistory[i][j] = momentum * labelEmbeddingGradientHistory[i][j] - labelEmbeddingGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.DEPENDENCY, i, j, learningRate * labelEmbeddingGradientHistory[i][j]);
            }
        }

        double[][] hiddenLayerGradient = gradients.getHiddenLayer();
        double[][] hiddenLayerGradientHistory = gradientHistory.getHiddenLayer();
        for (int i = 0; i < mlpNetwork.hiddenLayerSize; i++) {
            for (int j = 0; j < mlpNetwork.hiddenLayerIntSize; j++) {
                hiddenLayerGradientHistory[i][j] = momentum * hiddenLayerGradientHistory[i][j] - hiddenLayerGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.HIDDENLAYER, i, j, learningRate * hiddenLayerGradientHistory[i][j]);
            }
        }

        double[] hiddenLayerBiasGradient = gradients.getHiddenLayerBias();
        double[] hiddenLayerBiasGradientHistory = gradientHistory.getHiddenLayerBias();
        for (int i = 0; i < mlpNetwork.hiddenLayerSize; i++) {
            hiddenLayerBiasGradientHistory[i] = momentum * hiddenLayerBiasGradientHistory[i] - hiddenLayerBiasGradient[i];
            mlpNetwork.modify(EmbeddingTypes.HIDDENLAYERBIAS, i, -1, learningRate * hiddenLayerBiasGradientHistory[i]);
        }

        double[][] softmaxLayerGradient = gradients.getSoftmaxLayer();
        double[][] softmaxLayerGradientHistory = gradientHistory.getSoftmaxLayer();
        for (int i = 0; i < mlpNetwork.softmaxLayerSize; i++) {
            for (int j = 0; j < mlpNetwork.hiddenLayerSize; j++) {
                softmaxLayerGradientHistory[i][j] = momentum * softmaxLayerGradientHistory[i][j] - softmaxLayerGradient[i][j];
                mlpNetwork.modify(EmbeddingTypes.SOFTMAX, i, j, learningRate * softmaxLayerGradientHistory[i][j]);
            }
        }

        double[] softmaxLayerBiasGradient = gradients.getSoftmaxLayerBias();
        double[] softmaxLayerBiasGradientHistory = gradientHistory.getSoftmaxLayerBias();
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

    public void shutDownLiveThreads() {
        boolean isTerminated = executor.isTerminated();
        while (!isTerminated) {
            executor.shutdownNow();
            isTerminated = executor.isTerminated();
        }
    }
}
