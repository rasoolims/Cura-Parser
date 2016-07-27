package YaraParser.TransitionBasedSystem.Trainer;

import YaraParser.Accessories.CoNLLReader;
import YaraParser.Accessories.Evaluator;
import YaraParser.Accessories.Options;
import YaraParser.Accessories.Pair;
import YaraParser.Learning.MLPNetwork;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.NNInfStruct;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Parser.KBeamArcEagerParser;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.zip.GZIPOutputStream;

public class StaticNeuralTrainer {
    private static double bestAcc = 0;

    private static void initializeWordEmbeddingLayers(IndexMaps maps,
                                                      org.deeplearning4j.nn.layers.feedforward.embedding
                                                              .EmbeddingLayer layer) {
        int filled = 0;
        INDArray weights = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray rows = Nd4j.createUninitialized(new int[]{maps.vocabSize() + 2, weights.size(1)}, 'c');

        for (int i = 0; i < maps.vocabSize() + 2; i++) {
            double[] embeddings = maps.embeddings(i);
            if (embeddings != null) {
                INDArray newArray = Nd4j.create(embeddings);
                rows.putRow(i, newArray);
                filled++;
            } else {
                rows.putRow(i, weights.getRow(i));
            }
        }
        layer.setParam("W", rows);
        System.out.println("filled " + filled + " out of " + maps.vocabSize() + " vectors manually!");
    }

    public static void trainStaticNeural(ArcEagerBeamTrainer trainer, IndexMaps maps,
                                         int wordDimension, int posDimension, int depDimension,
                                         int possibleOutputs, ArrayList<Integer> dependencyRelations, Options
                                                 options) throws Exception {
        int vocab1Size = maps.vocabSize() + 2;
        int vocab2Size = maps.posSize() + 2;
        int vocab3Size = maps.relSize() + 2;

        double learningRate = options.learningRate;
        int batchSize = options.batchSize;

        CoNLLReader reader = new CoNLLReader(options.inputFile);
        ArrayList<GoldConfiguration> trainDataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled,
                options.rootFirst, options.lowercase, maps);
        int dataSize = 0;
        for (GoldConfiguration d : trainDataSet)
            dataSize += d.getSentence().size() * 2;
        dataSize /= batchSize;

        Collections.shuffle(trainDataSet);
        String trainFile = trainer.createStaticTrainingDataForNeuralNet(trainDataSet, options.inputFile + ".csv", -1);
        MultiDataSetIterator trainIter = readMultiDataSetIterator(trainFile, batchSize, possibleOutputs);

        MultiDataSetIterator devIter = null;
        ArrayList<GoldConfiguration> devDataSet = null;
        if (options.devPath != null && options.devPath.length() > 0) {
            CoNLLReader devReader = new CoNLLReader(options.devPath);
            devDataSet = devReader.readData(Integer.MAX_VALUE, false, options.labeled, options.rootFirst, options
                    .lowercase, maps);
            String devFile = trainer.createStaticTrainingDataForNeuralNet(devDataSet, options.devPath + ".csv", -1);
            devIter = readMultiDataSetIterator(devFile, batchSize, possibleOutputs);
        }

        ComputationGraph net = constructNetwork(options, learningRate, vocab1Size, vocab2Size, vocab3Size,
                wordDimension, posDimension, depDimension, possibleOutputs, maps);
        ComputationGraph avgNet = constructNetwork(options, learningRate, vocab1Size, vocab2Size, vocab3Size,
                wordDimension, posDimension, depDimension, possibleOutputs, maps);

        int step = 0;
        int decayStep = (int) (options.decayStep * dataSize);
        decayStep = decayStep == 0 ? 1 : decayStep;
        System.out.println("decay step is " + decayStep);
        int noImprovement = 0;

        for (int iter = 0; iter < options.trainingIter; iter++) {
            System.out.println(iter + "th iteration");
            trainIter.reset();
            while (trainIter.hasNext()) {
                INDArray wArrBefore = net.getLayer(0).getParam("W").mul(18);
                INDArray wArr2Before = net.getLayer(19).getParam("W").mul(18);
                INDArray wArr3Before = net.getLayer(38).getParam("W").mul(11);

                // making sure that we don't learn bias for embeddings
                for (int i = 0; i < 49; i++)
                    (net.getLayer(i)).conf().setLearningRateByParam("b", 0.0);

                net.fit(trainIter.next());

                shareEmbedding(wArrBefore, net, 0, 19);
                shareEmbedding(wArr2Before, net, 19, 38);
                shareEmbedding(wArr3Before, net, 38, 49);

                step++;
                double ratio = Math.min(0.9999, (double) step / (9 + step));

                averageLayers(avgNet, net, step, ratio, 0, 19);
                averageLayers(avgNet, net, step, ratio, 19, 38);
                averageLayers(avgNet, net, step, ratio, 38, 49);
                averageLayers(avgNet, net, step, ratio, 49, 50);
                averageLayers(avgNet, net, step, ratio, 50, 51);

                if (step % decayStep == 0) {
                    for (int i = 0; i < 51; i++) {
                        double lr = (net.getLayer(i)).conf().getLearningRateByParam("W");
                        (net.getLayer(i)).conf().setLearningRateByParam("W", lr * 0.96);
                        if (i > 48) {
                            lr = (net.getLayer(i)).conf().getLearningRateByParam("b");
                            (net.getLayer(i)).conf().setLearningRateByParam("b", lr * 0.96);
                        }
                    }
                    double lr = (net.getLayer(0)).conf().getLearningRateByParam("W");
                    System.out.println("learning rate:" + lr);
                    System.out.println("avg decay:" + Math.min(0.9999, (double) step / (9 + step)));

                    if (devIter != null) {
                        System.out.println("\nevaluate of dev avg");
                        noImprovement = evaluate(avgNet, maps, dependencyRelations, options, true, noImprovement);

                    }
                }
            }

            System.out.println("Reshuffling the data!");
            Collections.shuffle(trainDataSet);
            trainFile = trainer.createStaticTrainingDataForNeuralNet(trainDataSet, options.inputFile + ".csv", 0.1);
            trainIter = readMultiDataSetIterator(trainFile, batchSize, possibleOutputs);
            System.gc();

            if (noImprovement >= 10) {
                System.out.println("\nEarly stop...!");
                break;
            }
        }

        if (devIter == null) {
            System.out.println("Saving the new model for the final iteration ");
            MLPNetwork mlpNetwork = new MLPNetwork(new NNInfStruct(net, dependencyRelations.size(), maps,
                    dependencyRelations, options));
            saveModel(mlpNetwork, options.modelFile);
        }
    }

    private static void saveModel(MLPNetwork network, String modelPath) throws IOException {
        FileOutputStream fos = new FileOutputStream(modelPath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);
        writer.writeObject(network);
        writer.close();
    }

    private static MultiDataSetIterator readMultiDataSetIterator(String path, int batchSize, int possibleOutputs)
            throws IOException, InterruptedException {
        RecordReader rr = new CSVRecordReader(0, ",");
        rr.initialize(new FileSplit(new File(path)));
        int index = 1;
        MultiDataSetIterator iterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
                .addReader("fields_labels", rr)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addInput("fields_labels", index, index++)
                .addOutputOneHot("fields_labels", 0, possibleOutputs)
                .build();
        return iterator;
    }

    private static ComputationGraph constructNetwork(Options options, double learningRate, int vocab1Size,
                                                     int vocab2Size, int vocab3Size, int wordDimension,
                                                     int posDimension, int depDimension, int possibleOutputs,
                                                     IndexMaps maps) {
        NeuralNetConfiguration.Builder confBuilder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(learningRate).updater(Updater.NESTEROVS)
                .momentum(0.9).regularization(true).l2(0.0001).l1(0);


        String[] embeddingLayerNames = new String[49];
        for (int e = 0; e < embeddingLayerNames.length; e++) {
            embeddingLayerNames[e] = "L" + (e + 1);
        }

        int lIndex = 0;
        int vIndex = 0;
        ComputationGraphConfiguration confComplex = confBuilder.graphBuilder()
                .addInputs("s0w", "s1w", "s2w", "s3w", "b0w", "b1w", "b2w", "b3w", "b0l1w", "b0l2w", "s0l1w", "s0l2w",
                        "sr1w", "s0r2w", "sh0w", "sh1w", "b0llw", "s0llw", "s0rrw",
                        "s0p", "s1p", "s2p", "s3p", "b0p", "b1p", "b2p", "b3p", "b0l1p", "b0l2p", "s0l1p", "s0l2p",
                        "sr1p", "s0r2p", "sh0p", "sh1p", "b0llp", "s0llp", "s0rrp",
                        "s0l", "sh0l", "s0l1l", "sr1l", "s0l2l", "s0r2l", "b0l1l", "b0l2l", "b0lll", "s0lll", "s0rrl")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "s0w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "s1w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "s2w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "s3w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "b0w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "b1w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "b2w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "b3w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "b0l1w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "b0l2w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "s0l1w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "s0l2w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "sr1w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "s0r2w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "sh0w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "sh1w")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "b0llw")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "s0llw")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab1Size, wordDimension), "s0rrw")

                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "s0p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "s1p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "s2p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "s3p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "b0p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "b1p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "b2p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "b3p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "b0l1p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "b0l2p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "s0l1p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "s0l2p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "sr1p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "s0r2p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "sh0p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "sh1p")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "b0llp")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "s0llp")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab2Size, posDimension), "s0rrp")

                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab3Size, depDimension), "s0l")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab3Size, depDimension), "sh0l")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab3Size, depDimension), "s0l1l")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab3Size, depDimension), "sr1l")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab3Size, depDimension), "s0l2l")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab3Size, depDimension), "s0r2l")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab3Size, depDimension), "b0l1l")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab3Size, depDimension), "b0l2l")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab3Size, depDimension), "b0lll")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab3Size, depDimension), "s0lll")
                .addLayer(embeddingLayerNames[lIndex++], embeddingLayerBuilder(vocab3Size, depDimension), "s0rrl")
                .addVertex("concat", new MergeVertex(), embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++],
                        embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++], embeddingLayerNames[vIndex++])
                .addLayer("h1", new DenseLayer.Builder().nIn(19 * (wordDimension + posDimension) + 11 * depDimension)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.01)).biasInit(0.2)
                        //  .dropOut(dropoutProb)
                        .nOut(options.hiddenLayer1Size).activation("relu").build(), "concat")
                //   .addLayer("h2", new DenseLayer.Builder().nIn(options.hiddenLayer1Size)
                //          .weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0,0.01)).biasInit(0.2)
                //             .nOut(options.hiddenLayer2Size).activation("relu").build(), "h1")
                .addLayer("out", new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.01)).biasInit(0)
                        .nIn(options.hiddenLayer1Size).nOut(possibleOutputs).activation("softmax").build(), "h1")
                .setOutputs("out")
                .backprop(true).build();


        ComputationGraph net = new ComputationGraph(confComplex);

        net.init();
        net.setListeners(new ScoreIterationListener(10));
        // making sure that we don't learn bias for embeddings
        for (int i = 0; i < 49; i++)
            (net.getLayer(i)).conf().setLearningRateByParam("b", 0.0);

        if (maps.hasEmbeddings()) {
            System.out.println("Initializing with pre-trained word vectors");
            org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer layer =
                    (org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer) net.getLayer(0);
            initializeWordEmbeddingLayers(maps, layer);
        }

        INDArray wArr = net.getLayer(0).getParam("W");
        INDArray bArr = net.getLayer(0).getParam("b");
        for (int i = 1; i < 19; i++) {
            net.getLayer(i).setParam("W", wArr);
            net.getLayer(i).setParam("b", bArr);
        }
        INDArray wArr2 = net.getLayer(19).getParam("W");
        INDArray bArr2 = net.getLayer(19).getParam("b");
        for (int i = 20; i < 38; i++) {
            net.getLayer(i).setParam("W", wArr2);
            net.getLayer(i).setParam("b", bArr2);
        }
        INDArray wArr3 = net.getLayer(38).getParam("W");
        INDArray bArr3 = net.getLayer(38).getParam("b");
        for (int i = 39; i < 49; i++) {
            net.getLayer(i).setParam("W", wArr3);
            net.getLayer(i).setParam("b", bArr3);
        }

        return net;
    }

    private static EmbeddingLayer embeddingLayerBuilder(int inDim, int outDim) {
        double stdDev = 1.0 / Math.pow(outDim, 0.5);
        return new EmbeddingLayer.Builder().nIn(inDim).nOut(outDim).activation("identity")
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, stdDev)).biasInit(0.0).build();
    }

    private static int evaluate(final ComputationGraph net, final IndexMaps maps,
                                final ArrayList<Integer> dependencyRelations, final Options options, boolean save,
                                int noImprovement)
            throws Exception {
        MLPNetwork mlpNetwork = new MLPNetwork(new NNInfStruct(net, dependencyRelations.size(), maps,
                dependencyRelations, options));

        KBeamArcEagerParser.parseNNConllFileNoParallel(mlpNetwork, options.devPath, options.modelFile + ".tmp",
                options.beamWidth, 1, false, "");
        Pair<Double, Double> eval = Evaluator.evaluate(options.devPath, options.modelFile + ".tmp", options
                .punctuations);

        if (save) {
            double acc = eval.first;
            if (acc > bestAcc) {
                bestAcc = acc;
                noImprovement = 0;
                System.out.println("Saving the new model for iteration \n\n");
                saveModel(mlpNetwork, options.modelFile);
            } else {
                noImprovement++;
            }
        }
        return noImprovement;
    }

    private static void shareEmbedding(INDArray wArrBefore, ComputationGraph net, int s, int e) {
        INDArray wArr = net.getLayer(s).getParam("W");
        for (int i = s + 1; i < e; i++) {
            wArr.addi(net.getLayer(i).getParam("W"));
        }
        wArr = wArr.subi(wArrBefore);
        for (int i = s; i < e; i++) {
            net.getLayer(i).setParam("W", wArr);
        }
    }

    private static void averageLayers(ComputationGraph avgNet, ComputationGraph net, int step, double ratio, int s,
                                      int e) {
        if (step > 1) {
            avgNet.getLayer(s).getParam("W").muli(ratio).addi(net.getLayer(s).getParam("W").mul(1 - ratio));
            if (s > 48)
                avgNet.getLayer(s).getParam("b").muli(ratio).addi(net.getLayer(s).getParam("b").mul(1 -
                        ratio));
        } else {
            avgNet.getLayer(s).getParam("W").muli(0).addi(net.getLayer(s).getParam("W").mul(1 - ratio));
            if (s > 48)
                avgNet.getLayer(s).getParam("b").muli(0).addi(net.getLayer(s).getParam("b").mul(1 - ratio));
        }

        // all other layers
        for (int i = s + 1; i < e; i++) {
            avgNet.getLayer(i).setParam("W", avgNet.getLayer(s).getParam("W"));
            if (s > 48)
                avgNet.getLayer(i).setParam("b", avgNet.getLayer(s).getParam("b"));
        }
    }
}
