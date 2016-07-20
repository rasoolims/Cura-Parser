package YaraParser.TransitionBasedSystem.Trainer;

import YaraParser.Accessories.CoNLLReader;
import YaraParser.Accessories.Options;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.NNInfStruct;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.stepfunctions.NegativeDefaultStepFunction;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorService;
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
        // Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        int batchSize = options.batchSize;

        CoNLLReader reader = new CoNLLReader(options.inputFile);
        ArrayList<GoldConfiguration> trainDataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled,
                options.rootFirst, options.lowercase, maps);
        int dataSize = 0;
        for(GoldConfiguration d: trainDataSet)
            dataSize+= d.getSentence().size()*2;
        dataSize/= batchSize;

        Collections.shuffle(trainDataSet);
        String[] trainFiles = trainer.createStaticTrainingDataForNeuralNet(trainDataSet, options.inputFile + ".csv", -1);
        MultiDataSetIterator trainIter = readMultiDataSetIterator(trainFiles, batchSize, possibleOutputs);

        MultiDataSetIterator devIter = null;
        ArrayList<GoldConfiguration> devDataSet = null;
        if (options.devPath != null && options.devPath.length() > 0) {
            CoNLLReader devReader = new CoNLLReader(options.devPath);
            devDataSet = devReader.readData(Integer.MAX_VALUE, false, options.labeled, options.rootFirst, options
                    .lowercase, maps);
            String[] devFiles = trainer.createStaticTrainingDataForNeuralNet(devDataSet, options.devPath + ".csv", -1);
            devIter = readMultiDataSetIterator(devFiles, batchSize, possibleOutputs);
        }

        ComputationGraph net = constructNetwork(options, learningRate, vocab1Size, vocab2Size, vocab3Size,
                wordDimension, posDimension, depDimension, possibleOutputs, options.dropout,maps);
        ComputationGraph avgNet = constructNetwork(options, learningRate, vocab1Size, vocab2Size, vocab3Size,
                wordDimension, posDimension, depDimension, possibleOutputs, options.dropout,maps);


        int step = 0;
        int decayStep = (int)(options.decayStep*dataSize);
        decayStep = decayStep==0? 1: decayStep;
        System.out.println("decay step is "+decayStep);
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
                for (int i = 0; i < 51; i++) {
                    if (step > 1) {
                        avgNet.getLayer(i).getParam("W").muli(ratio).addi(net.getLayer(i).getParam("W").mul(1 - ratio));
                        if (i > 48)
                            avgNet.getLayer(i).getParam("b").muli(ratio).addi(net.getLayer(i).getParam("b").mul(1 - ratio));
                    } else {
                        avgNet.getLayer(i).getParam("W").muli(0).addi(net.getLayer(i).getParam("W").mul(1 - ratio));
                        if (i > 48)
                            avgNet.getLayer(i).getParam("b").muli(0).addi(net.getLayer(i).getParam("b").mul(1 - ratio));
                    }
                }

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
                        System.out.println("\nevaluate of dev");
                        evaluate(net, devIter, maps, dependencyRelations, options, true, noImprovement);

                        System.out.println("\nevaluate of dev avg");
                        evaluate(avgNet, devIter, maps, dependencyRelations, options, true, noImprovement);
                    }
                }
                System.gc();
            }


            System.out.println("Reshuffling the data!");
            Collections.shuffle(trainDataSet);
            trainFiles = trainer.createStaticTrainingDataForNeuralNet(trainDataSet, options.inputFile + ".csv", 0.1);

            trainIter = readMultiDataSetIterator(trainFiles, batchSize, possibleOutputs);

            if (devIter != null) {
                System.out.println("\nevaluate of dev");
                noImprovement = evaluate(net, devIter, maps, dependencyRelations, options, true, noImprovement);

                System.out.println("\nevaluate of dev avg");
                evaluate(avgNet, devIter, maps, dependencyRelations, options, true, noImprovement);

                if(noImprovement>=20) {
                    System.out.println("\nEarly stop...!");
                    break;
                }
            }
        }

        if (devIter == null) {
            System.out.println("Saving the new model for the final iteration ");
            saveModel(maps, dependencyRelations, options, net);
        }
    }

    private static void saveModel(IndexMaps maps, ArrayList<Integer> dependencyRelations, Options options,
                                  ComputationGraph net) throws IOException {
        FileOutputStream fos = new FileOutputStream(options.modelFile);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);
        ModelSerializer.writeModel(net, options.modelFile+".net",false);
        writer.writeObject(new NNInfStruct(options.modelFile+".net", dependencyRelations.size(), maps, dependencyRelations, options));
        writer.writeObject(net);
        writer.close();
    }

    private static MultiDataSetIterator readMultiDataSetIterator(String[] path, int batchSize, int possibleOutputs)
            throws IOException, InterruptedException {
        int numLinesToSkip = 0;
        String fileDelimiter = ",";
        RecordReader[] featuresReader = new RecordReader[path.length - 1];
        for (int i = 0; i < featuresReader.length; i++) {
            featuresReader[i] = new CSVRecordReader(numLinesToSkip, fileDelimiter);
            featuresReader[i].initialize(new FileSplit(new File(path[i])));
        }

        RecordReader labelsReader = new CSVRecordReader(numLinesToSkip, fileDelimiter);
        String labelsCsvPath = path[path.length - 1];
        labelsReader.initialize(new FileSplit(new File(labelsCsvPath)));

        int ind = 0;
        MultiDataSetIterator iterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
                .addReader("s0w", featuresReader[ind++])
                .addReader("s1w", featuresReader[ind++])
                .addReader("s2w", featuresReader[ind++])
                .addReader("s3w", featuresReader[ind++])
                .addReader("b0w", featuresReader[ind++])
                .addReader("b1w", featuresReader[ind++])
                .addReader("b2w", featuresReader[ind++])
                .addReader("b3w", featuresReader[ind++])
                .addReader("b0l1w", featuresReader[ind++])
                .addReader("b0l2w", featuresReader[ind++])
                .addReader("s0l1w", featuresReader[ind++])
                .addReader("s0l2w", featuresReader[ind++])
                .addReader("sr1w", featuresReader[ind++])
                .addReader("s0r2w", featuresReader[ind++])
                .addReader("sh0w", featuresReader[ind++])
                .addReader("sh1w", featuresReader[ind++])
                .addReader("b0llw", featuresReader[ind++])
                .addReader("s0llw", featuresReader[ind++])
                .addReader("s0rrw", featuresReader[ind++])

                .addReader("s0p", featuresReader[ind++])
                .addReader("s1p", featuresReader[ind++])
                .addReader("s2p", featuresReader[ind++])
                .addReader("s3p", featuresReader[ind++])
                .addReader("b0p", featuresReader[ind++])
                .addReader("b1p", featuresReader[ind++])
                .addReader("b2p", featuresReader[ind++])
                .addReader("b3p", featuresReader[ind++])
                .addReader("b0l1p", featuresReader[ind++])
                .addReader("b0l2p", featuresReader[ind++])
                .addReader("s0l1p", featuresReader[ind++])
                .addReader("s0l2p", featuresReader[ind++])
                .addReader("sr1p", featuresReader[ind++])
                .addReader("s0r2p", featuresReader[ind++])
                .addReader("sh0p", featuresReader[ind++])
                .addReader("sh1p", featuresReader[ind++])
                .addReader("b0llp", featuresReader[ind++])
                .addReader("s0llp", featuresReader[ind++])
                .addReader("s0rrp", featuresReader[ind++])

                .addReader("s0l", featuresReader[ind++])
                .addReader("sh0l", featuresReader[ind++])
                .addReader("s0l1l", featuresReader[ind++])
                .addReader("sr1l", featuresReader[ind++])
                .addReader("s0l2l", featuresReader[ind++])
                .addReader("s0r2l", featuresReader[ind++])
                .addReader("b0l1l", featuresReader[ind++])
                .addReader("b0l2l", featuresReader[ind++])
                .addReader("b0lll", featuresReader[ind++])
                .addReader("s0lll", featuresReader[ind++])
                .addReader("s0rrl", featuresReader[ind++])
                .addReader("csvLabels", labelsReader)
                .addInput("s0w")
                .addInput("s1w")
                .addInput("s2w")
                .addInput("s3w")
                .addInput("b0w")
                .addInput("b1w")
                .addInput("b2w")
                .addInput("b3w")
                .addInput("b0l1w")
                .addInput("b0l2w")
                .addInput("s0l1w")
                .addInput("s0l2w")
                .addInput("sr1w")
                .addInput("s0r2w")
                .addInput("sh0w")
                .addInput("sh1w")
                .addInput("b0llw")
                .addInput("s0llw")
                .addInput("s0rrw")

                .addInput("s0p")
                .addInput("s1p")
                .addInput("s2p")
                .addInput("s3p")
                .addInput("b0p")
                .addInput("b1p")
                .addInput("b2p")
                .addInput("b3p")
                .addInput("b0l1p")
                .addInput("b0l2p")
                .addInput("s0l1p")
                .addInput("s0l2p")
                .addInput("sr1p")
                .addInput("s0r2p")
                .addInput("sh0p")
                .addInput("sh1p")
                .addInput("b0llp")
                .addInput("s0llp")
                .addInput("s0rrp")

                .addInput("s0l")
                .addInput("sh0l")
                .addInput("s0l1l")
                .addInput("sr1l")
                .addInput("s0l2l")
                .addInput("s0r2l")
                .addInput("b0l1l")
                .addInput("b0l2l")
                .addInput("b0lll")
                .addInput("s0lll")
                .addInput("s0rrl")
                .addOutputOneHot("csvLabels", 0, possibleOutputs)
                .build();
        return iterator;
    }

    private static ComputationGraph constructNetwork(Options options, double learningRate, int vocab1Size,
                                                     int vocab2Size, int vocab3Size, int wordDimension,
                                                     int posDimension, int depDimension, int possibleOutputs, boolean
                                                             dropout, IndexMaps maps) {
        NeuralNetConfiguration.Builder confBuilder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).miniBatch(true).iterations(1)
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
        for(int i=0;i<49;i++)
            (net.getLayer(i)).conf().setLearningRateByParam("b",0.0);

        if (maps.hasEmbeddings()) {
            System.out.println("Initializing with pre-trained word vectors");
            org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer layer =
                    (org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer) net.getLayer(0);
            initializeWordEmbeddingLayers(maps, layer);
        }

        INDArray wArr = net.getLayer(0).getParam("W");
        INDArray bArr = net.getLayer(0).getParam("b");
        for (int i = 1; i < 19; i++) {
            net.getLayer(i).setParam("W",wArr);
            net.getLayer(i).setParam("b",bArr);
        }
        INDArray wArr2 = net.getLayer(19).getParam("W");
        INDArray bArr2 = net.getLayer(19).getParam("b");
        for (int i = 20; i < 38; i++) {
            net.getLayer(i).setParam("W",wArr2);
            net.getLayer(i).setParam("b",bArr2);
        }
        INDArray wArr3 = net.getLayer(38).getParam("W");
        INDArray bArr3 = net.getLayer(38).getParam("b");
        for (int i = 39; i < 49; i++) {
            net.getLayer(i).setParam("W",wArr3);
            net.getLayer(i).setParam("b",bArr3);
        }

        return net;
    }

    private static EmbeddingLayer embeddingLayerBuilder(int inDim, int outDim) {
        double stdDev = 1.0 / Math.pow(outDim,0.5);
        return new EmbeddingLayer.Builder().nIn(inDim).nOut(outDim).activation("identity")
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, stdDev)).biasInit(0.0).build();
    }

    private static int evaluate(final ComputationGraph net, MultiDataSetIterator iter, final IndexMaps maps,
                                 final ArrayList<Integer> dependencyRelations, final Options options, boolean save,
                                 int noImprovement)
            throws Exception {
        iter.reset();
        Evaluation evaluation = new Evaluation(2 * (dependencyRelations.size() + 1));

       /**
        INDArray wEArr = net.getLayer(0).getParam("W");
        double[][] E_W = new double[wEArr.rows()][wEArr.columns()];
        for(int i=0;i<E_W.length;i++){
            for(int j=0;j<E_W[i].length;j++)
                E_W[i][j] = wEArr.getRow(i).getColumn(j).getDouble(0);
        }

        INDArray eWB = net.getLayer(0).getParam("b");
        double[] E_W_B = new double[eWB.columns()];
        for(int i=0;i<E_W_B.length;i++){
            E_W_B[i] = eWB.getColumn(i).getDouble(0);
        }

        INDArray pEArr = net.getLayer(20).getParam("W");
        double[][] E_P = new double[pEArr.rows()][pEArr.columns()];
        for(int i=0;i<E_P.length;i++){
            for(int j=0;j<E_P[i].length;j++)
                E_P[i][j] = pEArr.getRow(i).getColumn(j).getDouble(0);
        }

        INDArray ePB = net.getLayer(20).getParam("b");
        double[] E_P_B = new double[ePB.columns()];
        for(int i=0;i<E_P_B.length;i++){
            E_P_B[i] = ePB.getColumn(i).getDouble(0);
        }

        INDArray lEArr = net.getLayer(39).getParam("W");
        double[][] E_L = new double[lEArr.rows()][lEArr.columns()];
        for(int i=0;i<E_L.length;i++){
            for(int j=0;j<E_L[i].length;j++)
                E_L[i][j] = lEArr.getRow(i).getColumn(j).getDouble(0);
        }

        INDArray eLB = net.getLayer(49).getParam("b");
        double[] E_L_B = new double[eLB.columns()];
        for(int i=0;i<E_L_B.length;i++){
            E_L_B[i] = eLB.getColumn(i).getDouble(0);
        }

       INDArray hW = net.getLayer(49).getParam("W");
        double[][] H_W = new double[hW.rows()][hW.columns()];
        for(int i=0;i<H_W.length;i++){
            for(int j=0;j<H_W[i].length;j++)
                H_W[i][j] = hW.getRow(i).getColumn(j).getDouble(0);
        }

        INDArray hB = net.getLayer(49).getParam("b");
        double[] H_B = new double[hB.columns()];
        for(int i=0;i<H_B.length;i++){
            H_B[i] = hB.getColumn(i).getDouble(0);
        }

        INDArray sW = net.getLayer(50).getParam("W");
        double[][] S_W = new double[sW.rows()][sW.columns()];
        for(int i=0;i<S_W.length;i++){
            for(int j=0;j<S_W[i].length;j++)
                S_W[i][j] = sW.getRow(i).getColumn(j).getDouble(0);
        }

        INDArray sB = net.getLayer(50).getParam("b");
        double[] S_B = new double[sB.columns()];
        for(int i=0;i<S_B.length;i++){
            S_B[i] = sB.getColumn(i).getDouble(0);
        }
        **/

        while (iter.hasNext()) {
            MultiDataSet t = iter.next();
            INDArray[] features = t.getFeatures();
            INDArray labels = t.getLabels()[0];
            INDArray predicted = net.output(false, features)[0];
            evaluation.eval(labels, predicted);

            /**
             * get actual vals
             */
            /**
                int[] feats = new int[features.length];
                for(int i=0;i<feats.length;i++)
                    feats[i]=features[i].getInt(0);

                double[] hidden = new double[H_W[0].length];

                int offset = 0;
                for(int j=0;j<feats.length;j++){
                    int tok = feats[j];
                    double[][] embedding = null;
                    if(j<19)
                        embedding = E_W;
                    else if(j<38)
                        embedding = E_P;
                    else embedding = E_L;

                    for(int i=0;i<hidden.length;i++){
                        for(int k=0;k<embedding[0].length;k++){
                            hidden[i]+= H_W[offset+k][i]*embedding[tok][k];
                        }
                    }
                    offset+= embedding[0].length;
                }

                for(int i=0;i<hidden.length;i++){
                    hidden[i]+= H_B[i];
                    //relu
                    hidden[i] = Math.max(0,hidden[i]);
                }

                double[] probs = new double[S_B.length];
                double sum = 0;
                for(int i=0;i<probs.length;i++){
                    for(int j=0;j<hidden.length;j++){
                        probs[i]+=S_W[j][i]* hidden[j];
                    }
                    probs[i]+= S_B[i];
                    probs[i] = Math.exp(probs[i]);
                    sum+= probs[i];
                }

                for(int i=0;i<probs.length;i++){
                   probs[i]/=sum;
                }
                double[] nd4jP = new double[predicted.columns()];
                for(int i=0;i<nd4jP.length;i++)
                    nd4jP[i]=predicted.getColumn(i).getDouble(0);
                **/
        }
        System.out.println("acc: " + evaluation.accuracy());
        System.out.println("precision: " + evaluation.precision());
        System.out.println("recall: " + evaluation.recall());
        System.out.println("f1 score: " + evaluation.f1() + "\n");

        if (save) {
            double acc = evaluation.accuracy();
            if (acc > bestAcc) {
                bestAcc = acc;
                noImprovement = 0;
                System.out.println("Saving the new model for iteration \n\n");
                saveModel(maps, dependencyRelations, options, net);
            } else {
                noImprovement++;
            }
        }
        return noImprovement;
    }

    private static void shareEmbedding(INDArray wArrBefore, ComputationGraph net, int s, int e){
        INDArray wArr = net.getLayer(s).getParam("W");
        for (int i = s + 1; i < e; i++) {
            wArr.addi(net.getLayer(i).getParam("W"));
        }
        wArr = wArr.subi(wArrBefore);
        for (int i = s; i < e; i++) {
            net.getLayer(i).setParam("W", wArr);
        }
    }
}
