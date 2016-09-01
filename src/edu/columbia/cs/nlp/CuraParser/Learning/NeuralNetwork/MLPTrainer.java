package edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork;

import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.FirstHiddenLayer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.Layer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.WordEmbeddingLayer;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.*;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.UpdaterType;
import edu.columbia.cs.nlp.CuraParser.Structures.Enums.EmbeddingTypes;
import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.Structures.NeuralTrainingInstance;
import edu.columbia.cs.nlp.CuraParser.Structures.Pair;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Features.FeatureExtractor;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.ParserType;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcEager;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcStandard;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.text.DecimalFormat;
import java.util.ArrayList;
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
    CompletionService<Pair<Pair<Double, Double>, MLPNetwork>> pool;
    int numThreads;

    /**
     * Keep track of loss function
     */
    double cost = 0.0;
    double correct = 0.0;
    int samples = 0;
    Updater updater;
    Random random;
    ShiftReduceParser parser;
    /**
     * Gradients
     */
    private ArrayList<Layer> gradients;
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

        if (options.generalProperties.parserType == ParserType.ArcStandard)
            parser = new ArcStandard();
        else if (options.generalProperties.parserType == ParserType.ArcEager)
            parser = new ArcEager();
        else
            throw new NotImplementedException();
    }

    private void regularizeWithL2() {
        double regCost = 0.0;

        ArrayList<Layer> layers = net.getLayers();
        for (int i = 0; i < layers.size() - 1; i++) {
            Layer layer = layers.get(i);
            Layer gradient = gradients.get(i);
            for (int d1 = 0; d1 < layer.nOut(); d1++) {
                if (regularizeAllLayers) {
                    regCost += Math.pow(layer.b(d1), 2);
                    gradient.modifyB(d1, regCoef * 2 * layer.b(d1));
                }
                for (int d2 = 0; d2 < layer.nIn(); d2++) {
                    regCost += Math.pow(layer.w(d1, d2), 2);
                    gradient.modifyW(d1, d2, regCoef * 2 * layer.w(d1, d2));
                }
            }
        }

        if (regularizeAllLayers) {
            // regularizing wrt embedding layers.
            FirstHiddenLayer fLayer = (FirstHiddenLayer) layers.get(0);
            FirstHiddenLayer fGradient = (FirstHiddenLayer) gradients.get(0);

            Layer wLayer = fLayer.getWordEmbeddings();
            Layer pLayer = fLayer.getPosEmbeddings();
            Layer dLayer = fLayer.getDepEmbeddings();

            Layer wGrad = fGradient.getWordEmbeddings();
            Layer pGrad = fGradient.getPosEmbeddings();
            Layer dGrad = fGradient.getDepEmbeddings();

            for (int d1 = 0; d1 < wLayer.nOut(); d1++) {
                for (int d2 = 0; d2 < wLayer.nIn(); d2++) {
                    regCost += Math.pow(wLayer.w(d1, d2), 2);
                    wGrad.modifyW(d1, d2, regCoef * 2 * wLayer.w(d1, d2));
                }
            }

            for (int d1 = 0; d1 < pLayer.nOut(); d1++) {
                for (int d2 = 0; d2 < pLayer.nIn(); d2++) {
                    regCost += Math.pow(pLayer.w(d1, d2), 2);
                    pGrad.modifyW(d1, d2, regCoef * 2 * pLayer.w(d1, d2));
                }
            }

            for (int d1 = 0; d1 < dLayer.nOut(); d1++) {
                for (int d2 = 0; d2 < dLayer.nIn(); d2++) {
                    regCost += Math.pow(dLayer.w(d1, d2), 2);
                    dGrad.modifyW(d1, d2, regCoef * 2 * dLayer.w(d1, d2));
                }
            }

            // regularizing wrt last layer.
            Layer layer = layers.get(layers.size() - 1);
            Layer gradient = gradients.get(layers.size() - 1);
            for (int d1 = 0; d1 < layer.nOut(); d1++) {
                regCost += Math.pow(layer.b(d1), 2);
                gradient.modifyB(d1, regCoef * 2 * layer.b(d1));
                for (int d2 = 0; d2 < layer.nIn(); d2++) {
                    regCost += Math.pow(layer.w(d1, d2), 2);
                    gradient.modifyW(d1, d2, regCoef * 2 * layer.w(d1, d2));
                }
            }
        }

        cost += regCoef * regCost;
    }

    public double fit(List instances, int iteration, boolean print) throws Exception {
        DecimalFormat format = new DecimalFormat("##.00");
        DecimalFormat format4 = new DecimalFormat("##.0000");

        cost(instances);
        regularizeWithL2();
        updater.update(gradients);
        ((FirstHiddenLayer) net.getLayers().get(0)).preCompute();

        double acc = correct / samples;
        if (print) {
            System.out.println(Utils.timeStamp() + " ---  iteration " + iteration + " --- size " +
                    samples + " --- Correct " + format.format(100. * acc) + " --- cost: " + format4.format(cost / samples));
            cost = 0;
            samples = 0;
            correct = 0;
        }
        return acc;
    }

    public void cost(List instances) throws Exception {
        submitThreads(instances);
        mergeCosts(instances);
        samples += instances.size();
    }

    private void submitThreads(List instances) {
        int chunkSize = Math.max(1, instances.size() / numThreads);
        int s = 0;
        int e = Math.min(instances.size(), chunkSize);
        for (int i = 0; i < Math.min(instances.size(), numThreads); i++) {
            pool.submit(new CostThread(instances.subList(s, e), instances.size()));
            s = e;
            e = Math.min(instances.size(), e + chunkSize);
        }
    }

    private void mergeCosts(List<NeuralTrainingInstance> instances) throws Exception {
        Pair<Pair<Double, Double>, MLPNetwork> firstResult = pool.take().get();
        gradients = firstResult.second.getLayers();

        cost += firstResult.first.first;
        correct += firstResult.first.second;

        for (int i = 1; i < Math.min(instances.size(), numThreads); i++) {
            Pair<Pair<Double, Double>, MLPNetwork> result = pool.take().get();

            for (int l = 0; l < gradients.size(); l++)
                gradients.get(l).mergeInPlace(result.second.getLayers().get(l));
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

    public void shutDownLiveThreads() {
        boolean isTerminated = executor.isTerminated();
        while (!isTerminated) {
            executor.shutdownNow();
            isTerminated = executor.isTerminated();
        }
    }

    private void backPropSavedGradients(MLPNetwork g, double[][][] savedGradients, HashSet<Integer>[] wordsSeen) {
        int offset = 0;
        final FirstHiddenLayer firstHiddenLayer = (FirstHiddenLayer) net.layer(0);
        final double[][] hiddenLayer = firstHiddenLayer.getW();
        WordEmbeddingLayer wordEmbeddingLayer = firstHiddenLayer.getWordEmbeddings();
        final double[][] wE = wordEmbeddingLayer.getW();
        final double[][] pE = firstHiddenLayer.getPosEmbeddings().getW();
        final double[][] lE = firstHiddenLayer.getDepEmbeddings().getW();

        for (int index = 0; index < g.getNumWordLayers(); index++) {
            for (int tok : wordsSeen[index]) {
                int id = wordEmbeddingLayer.preComputeId(index, tok);
                double[] embedding = wE[tok];
                for (int h = 0; h < hiddenLayer.length; h++) {
                    double delta = savedGradients[index][id][h];
                    for (int k = 0; k < embedding.length; k++) {
                        g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, delta * embedding[k]);
                        g.modify(EmbeddingTypes.WORD, tok, k, delta * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += wordEmbeddingLayer.dim();
        }

        for (int index = g.getNumWordLayers(); index < g.getNumWordLayers() + g.getNumPosLayers(); index++) {
            for (int tok = 0; tok < firstHiddenLayer.getPosEmbeddings().vocabSize(); tok++) {
                double[] embedding = pE[tok];
                for (int h = 0; h < hiddenLayer.length; h++) {
                    double delta = savedGradients[index][tok][h];
                    for (int k = 0; k < embedding.length; k++) {
                        g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, delta * embedding[k]);
                        g.modify(EmbeddingTypes.POS, tok, k, delta * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += firstHiddenLayer.getPosEmbeddings().dim();
        }

        for (int index = g.getNumWordLayers() + g.getNumPosLayers();
             index < g.getNumWordLayers() + g.getNumPosLayers() + g.getNumDepLayers(); index++) {
            for (int tok = 0; tok < firstHiddenLayer.getDepEmbeddings().vocabSize(); tok++) {
                double[] embedding = lE[tok];
                for (int h = 0; h < hiddenLayer.length; h++) {
                    double delta = savedGradients[index][tok][h];
                    for (int k = 0; k < embedding.length; k++) {
                        g.modify(EmbeddingTypes.HIDDENLAYER, h, offset + k, delta * embedding[k]);
                        g.modify(EmbeddingTypes.DEPENDENCY, tok, k, delta * hiddenLayer[h][offset + k]);
                    }
                }
            }
            offset += firstHiddenLayer.getDepEmbeddings().dim();
        }
    }

    /**
     * @param instances      this is actually a List of NeuralTrainingInstance(s).
     * @param batchSize
     * @param g
     * @param savedGradients
     * @return
     * @throws Exception
     */
    public Pair<Double, Double> cost(List instances, int batchSize, MLPNetwork g, double[][][] savedGradients)
            throws Exception {
        double cost = 0;
        double correct = 0;
        HashSet<Integer>[] featuresSeen = Utils.createHashSetArray(g.getNumWordLayers());

        double[][] features = new double[instances.size()][];
        double[][] labels = new double[instances.size()][];
        ArrayList<double[][]> activations = new ArrayList<>();
        ArrayList<double[][]> zs = new ArrayList<>();
        HashSet<Integer>[] hiddenNodesToUse = dropout(instances.size(), net.layer(0).nOut());
        HashSet<Integer>[] finalHiddenNodesToUse = hiddenNodesToUse;

        for (int i = 0; i < instances.size(); i++) {
            features[i] = ((NeuralTrainingInstance) instances.get(i)).getFeatures();
            labels[i] = ((NeuralTrainingInstance) instances.get(i)).getLabel();
        }

        zs.add(features);
        activations.add(features);
        int lIndex = 0;
        zs.add(net.layer(lIndex).forward(activations.get(activations.size() - 1), hiddenNodesToUse));
        activations.add(net.layer(lIndex++).activate(zs.get(zs.size() - 1), false));
        if (g.getLayers().size() >= 3) {
            HashSet<Integer>[] secondHiddenNodesToUse = dropout(instances.size(), net.layer(lIndex).nOut());
            zs.add(net.layer(lIndex).forward(activations.get(activations.size() - 1), secondHiddenNodesToUse, hiddenNodesToUse));
            activations.add(net.layer(lIndex++).activate(zs.get(zs.size() - 1), false));
            finalHiddenNodesToUse = secondHiddenNodesToUse;
        }
        zs.add(net.layer(lIndex).forward(activations.get(activations.size() - 1), labels, false, false));
        activations.add(net.layer(lIndex++).activate(zs.get(zs.size() - 1), false));
        double[][] probs = activations.get(activations.size() - 1);
        for (int i = 0; i < probs.length; i++) {
            int argmax = Utils.argmax(probs[i]);
            int gold = ((NeuralTrainingInstance) instances.get(i)).gold();
            double goldProb = probs[i][gold] == 0 ? 1e-120 : probs[i][gold];
            cost += -Math.log(goldProb);
            if (argmax == gold) correct++;
        }

        double[][] delta = new double[probs.length][probs[0].length];
        for (int i = 0; i < probs.length; i++) {
            for (int j = 0; j < probs[0].length; j++) {
                if (labels[i][j] >= 0)
                    delta[i][j] = (-labels[i][j] + probs[i][j]) / batchSize;
            }
        }

        // Back-propagating for the last layer.
        double[][] lastHiddenActivation = activations.get(activations.size() - 2);
        for (int i = 0; i < delta.length; i++) {
            for (int j = 0; j < delta[i].length; j++) {
                if (labels[i][j] >= 0 && delta[i][j] != 0) {
                    g.modify(net.numLayers() - 1, j, -1, delta[i][j]);
                    for (int h : finalHiddenNodesToUse[i]) {
                        g.modify(net.numLayers() - 1, j, h, delta[i][j] * lastHiddenActivation[i][h]);
                    }
                }
            }
        }

        // backwarding to all other layers.
        for (int i = net.numLayers() - 2; i >= 0; i--) {
            delta = g.layer(i).backward(delta, i, zs.get(i + 1), activations.get(i), activations.get(i + 1), featuresSeen, savedGradients, net);
        }

        backPropSavedGradients(g, savedGradients, featuresSeen);
        return new Pair<>(cost, correct);
    }

    private HashSet<Integer>[] dropout(int numOfInstances, int size) {
        HashSet<Integer>[] hiddenNodesToUse = new HashSet[numOfInstances];
        for (int i = 0; i < hiddenNodesToUse.length; i++)
            hiddenNodesToUse[i] = dropout(size);
        return hiddenNodesToUse;
    }

    private HashSet<Integer> dropout(int size) {
        HashSet<Integer> hiddenNodesToUse = new HashSet<>();
        for (int h = 0; h < size; h++) {
            if (dropoutProb <= 0 || random.nextDouble() >= dropoutProb)
                hiddenNodesToUse.add(h);
        }
        return hiddenNodesToUse;
    }

    public ArrayList<Layer> getGradients() {
        return gradients;
    }

    /**
     * @param instances      It is actually a list of Pair<Configuration, ArrayList<Configuration>>.
     * @param batchSize
     * @param g    Gradients that are calculated.
     * @param savedGradients   For pre-computation.
     * @return
     * @throws Exception
     */
    public Pair<Double, Double> beamCost(List instances, int batchSize, MLPNetwork g, double[][][] savedGradients) throws Exception {
        double cost = 0;
        double correct = 0;
        HashSet<Integer>[] featuresSeen = Utils.createHashSetArray(g.getNumWordLayers());

        for (int i = 0; i < instances.size(); i++) {
            Pair<Configuration, ArrayList<Configuration>> currInstance = (Pair<Configuration, ArrayList<Configuration>>) instances.get(i);
            Configuration gold = currInstance.first;
            ArrayList<Configuration> beam = currInstance.second;
            boolean rootFirst = gold.sentence.getWords()[gold.sentence.size() - 1] != IndexMaps.RootIndex;
            Configuration initialConfig = new Configuration(gold.sentence, rootFirst);


            // For each beam, each action get the inputs and activations.
            double[][][][] inputs = new double[beam.size()][beam.get(0).actionHistory.size()][net.numLayers() + 1][];
            double[][][][] activations = new double[beam.size()][beam.get(0).actionHistory.size()][net.numLayers() + 1][];
            double denom = 0;
            double[] beamDenom = new double[beam.size()];
            double maxBDenom = Double.NEGATIVE_INFINITY;

            int goldElement = -1;

            for (int b = 0; b < beam.size(); b++) {
                ArrayList<Integer> actions = beam.get(b).actionHistory;
                Configuration curConfig = initialConfig.clone();
                // Finding the gold element index.
                if (goldElement == -1 && beam.get(b).equals(gold))
                    goldElement = b;

                for (int a = 0; a < actions.size(); a++) {
                    int action = actions.get(a);
                    double[] feats = FeatureExtractor.extractFeatures(curConfig, net.maps.labelNullIndex, parser);
                    inputs[b][a][0] = feats;
                    activations[b][a][0] = feats;

                    for (int l = 1; l < net.numLayers() + 1; l++) {
                        inputs[b][a][l] = net.layer(l - 1).forward(activations[b][a][l - 1]);
                        activations[b][a][l] = net.layer(l - 1).activate(inputs[b][a][l], false);
                    }

                    beamDenom[b] += activations[b][a][net.numLayers()][action >= 2 ? action - 1 : action];
                    parser.advance(curConfig, action, net.depLabels.size());
                }
                if (beamDenom[b] >= maxBDenom)
                    maxBDenom = beamDenom[b];
            }

            for (int b = 0; b < beam.size(); b++) {
                beamDenom[b] -= maxBDenom;
                denom += Math.exp(beamDenom[b]);
            }

            if (goldElement == 0)
                correct++;

            assert goldElement != -1;
            for (int b = 0; b < beam.size(); b++) {
                ArrayList<Integer> actions = beam.get(b).actionHistory;

                // Finding the gold element index.
                int indicator = goldElement == b ? 1 : 0;
                if (goldElement == b) {
                    cost += Math.log(denom) - beamDenom[b];
                    if (Double.isInfinite(cost))
                        throw new Exception("Infinite cost!");
                }

                double beamProb = Math.exp(beamDenom[b]) / denom;
                double curDelta = (-indicator + beamProb) / batchSize;
                for (int a = 0; a < actions.size(); a++) {
                    int label = actions.get(a) >= 2 ? actions.get(a) - 1 : actions.get(a);
                    double[] delta = new double[net.getNumOutputs()];
                    delta[label] = curDelta;

                    // Modifying the bias term
                    g.modify(net.numLayers() - 1, label, -1, delta[label]);

                    double[] lastHiddenActivation = activations[b][a][net.numLayers() - 1];
                    if (delta[label] != 0.0) {
                        for (int h = 0; h < lastHiddenActivation.length; h++) {
                            g.modify(net.numLayers() - 1, label, h, delta[label] * lastHiddenActivation[h]);
                        }

                        for (int l = net.numLayers() - 2; l >= 0; l--) {
                            delta = g.layer(l).backward(delta, l, inputs[b][a][l + 1], activations[b][a][l], activations[b][a][l + 1], featuresSeen,
                                    savedGradients, net);
                        }
                    }
                }
            }
        }

        backPropSavedGradients(g, savedGradients, featuresSeen);
        return new Pair<>(cost, correct);
    }

    public class CostThread implements Callable<Pair<Pair<Double, Double>, MLPNetwork>> {
        List<Object> instances;
        int batchSize;
        MLPNetwork g;
        double[][][] savedGradients;

        public CostThread(List<Object> instances, int batchSize) {
            this.instances = instances;
            this.batchSize = batchSize;
            g = net.clone(true, false);
            savedGradients = net.instantiateSavedGradients();
        }

        @Override
        public Pair<Pair<Double, Double>, MLPNetwork> call() throws Exception {
            Pair<Double, Double> costValue = null;
            if (instances.get(0) instanceof NeuralTrainingInstance)
                costValue = cost(instances, batchSize, g, savedGradients);
            else
                costValue = beamCost(instances, batchSize, g, savedGradients);

            return new Pair<>(costValue, g);
        }
    }
}
