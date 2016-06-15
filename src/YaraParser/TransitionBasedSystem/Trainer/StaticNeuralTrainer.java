package YaraParser.TransitionBasedSystem.Trainer;

import YaraParser.Accessories.CoNLLReader;
import YaraParser.Accessories.Options;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.NNInfStruct;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Parser.KBeamArcEagerParser;
import net.didion.jwnl.data.Exc;
import org.apache.commons.collections.map.HashedMap;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPOutputStream;

public class StaticNeuralTrainer {


    private  static void initializeWordEmbeddingLayers(IndexMaps maps,
                                                org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer layer) {
        int filled = 0;
        INDArray weights = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray rows = Nd4j.createUninitialized(new int[]{maps.vocabSize() + 2, weights.size(1)}, 'c');
        for (int i = 0; i < maps.vocabSize() + 2; i++) {
            double[] embeddings = maps.embeddings(i);
            if (embeddings != null & i>2) {
                INDArray newArray = Nd4j.create(embeddings);
                rows.putRow(i, newArray);
                filled++;
            } else {
                rows.putRow(i, weights.getRow(i));
            }
        }
        layer.setParam("W", rows);
        System.out.println("filled "+filled+" out of "+maps.vocabSize()+" vectors manually!");
    }

    public static void trainStaticNeural(String[] trainFeatPath, String[] devFeatPath, IndexMaps maps,
                                         int wordDimension, int posDimension, int depDimension,
                                         int h1Dimension, int h2Dimension, int possibleOutputs, int nEpochs
            , String modelPath, String conllPath, ArrayList<Integer> dependencyRelations, Options options) throws Exception {
        int vocab1Size = maps.vocabSize() + 2;
        int vocab2Size = maps.posSize() + 2;
        int vocab3Size = maps.relSize() + 2;
        //  vocab1Size =13, vocab2Size=8,   vocab3Size=2
        // wordDimension = 64,   posDimension=32,  depDimension=32
        // h1Dimension =100, possibleOutputs=4,  nEpochs=30

        double learningRate = 0.01;
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        int batchSize = 1000;

        MultiDataSetIterator trainIter = readMultiDataSetIterator(trainFeatPath, batchSize, possibleOutputs);
        MultiDataSetIterator devIter = readMultiDataSetIterator(devFeatPath, 1, possibleOutputs);

        EmbeddingLayer wordLayer1 = new EmbeddingLayer.Builder().nIn(vocab1Size).nOut(wordDimension).activation("identity").updater(Updater.NESTEROVS).build();
        EmbeddingLayer wordLayer2 = new EmbeddingLayer.Builder().nIn(vocab1Size).nOut(wordDimension).activation("identity").updater(Updater.NESTEROVS).build();
        EmbeddingLayer wordLayer3 = new EmbeddingLayer.Builder().nIn(vocab1Size).nOut(wordDimension).activation("identity").updater(Updater.NESTEROVS).build();
        EmbeddingLayer wordLayer4 = new EmbeddingLayer.Builder().nIn(vocab1Size).nOut(wordDimension).activation("identity").updater(Updater.NESTEROVS).build();

        Map<Integer, Double> momentumSchedule = new HashedMap();
        double m = .96;
        for(int i=1;i<100000;i++){
           momentumSchedule.put(i,m);
            m*=0.96;
        }


        NeuralNetConfiguration.Builder confBuilder =  new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .momentum(0.96).regularization(true).l2(0.0001);//.dropOut(0.5) ;
        confBuilder.setMomentumSchedule(momentumSchedule);

        ComputationGraphConfiguration confComplex = confBuilder.graphBuilder()
                .addInputs("s0w", "b0w", "b1w", "b2w", "s0p", "b0p", "b1p", "b2p", "s0l", "sh0l")
                .addLayer("L1", wordLayer1, "s0w")
                .addLayer("L2", wordLayer2, "b0w")
                .addLayer("L3", wordLayer3, "b1w")
                .addLayer("L4", wordLayer4, "b2w")
                .addLayer("L5", new EmbeddingLayer.Builder().nIn(vocab2Size).nOut(posDimension).activation("identity").updater(Updater.NESTEROVS).build(), "s0p")
                .addLayer("L6", new EmbeddingLayer.Builder().nIn(vocab2Size).nOut(posDimension).activation("identity").updater(Updater.NESTEROVS).build(), "b0p")
                .addLayer("L7", new EmbeddingLayer.Builder().nIn(vocab2Size).nOut(posDimension).activation("identity").updater(Updater.NESTEROVS).build(), "b1p")
                .addLayer("L8", new EmbeddingLayer.Builder().nIn(vocab2Size).nOut(posDimension).activation("identity").updater(Updater.NESTEROVS).build(), "b2p")
                .addLayer("L9", new EmbeddingLayer.Builder().nIn(vocab3Size).nOut(depDimension).activation("identity").updater(Updater.NESTEROVS).build(), "s0l")
                .addLayer("L10", new EmbeddingLayer.Builder().nIn(vocab3Size).nOut(depDimension).activation("identity").updater(Updater.NESTEROVS).build(), "sh0l")
                .addVertex("concat", new MergeVertex(), "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10")
                .addLayer("h1", new DenseLayer.Builder().nIn(4 * (wordDimension + posDimension) + 2 * depDimension)
                        .nOut(h1Dimension).activation("relu").updater(Updater.NESTEROVS).build(), "concat")
                //.addLayer("h2", new DenseLayer.Builder().nIn(h1Dimension)
                //        .nOut(h2Dimension).activation("relu").build(), "h1")
                .addLayer("out", new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nIn(h1Dimension).nOut(possibleOutputs).activation("softmax").updater(Updater.NESTEROVS).build(), "h1")
                .setOutputs("out")
                .backprop(true).build();


        ComputationGraph net = new ComputationGraph(confComplex);
        net.init();
        net.setListeners(new ScoreIterationListener(100));


        if(maps.hasEmbeddings()) {
            for (int i = 0; i < 4; i++) {
                System.out.println("Initializing with pre-trained word vectors");
                org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer layer =
                        (org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer) net.getLayer(i);
                initializeWordEmbeddingLayers(maps, layer);
            }
        }
        DecimalFormat format = new DecimalFormat("##.00");
        double bestAcc = 0;
        for(int iter=0;iter<nEpochs;iter++) {

            System.out.println(iter+"th iteration");
            while (trainIter.hasNext())
                 net.fit(trainIter.next());
            trainIter.reset();

            try {
                devIter.reset();
            }catch (Exception ex){
                devIter = readMultiDataSetIterator(devFeatPath, 1, possibleOutputs);
            }
                int cor = 0;
            int all = 0;
            while (devIter.hasNext()) {
                MultiDataSet t = devIter.next();
                INDArray[] features = t.getFeatures();
                INDArray[] predicted = net.output(features);

                double max = Double.NEGATIVE_INFINITY;
                int argmax = 0;
                int gold = 0;

                for (int i = 0; i < predicted[0].length(); i++) {
                    double val = predicted[0].getDouble(i);
                    if (val >= max) {
                        argmax = i;
                        max = val;
                    }
                    if (t.getLabels(0).getDouble(i) == 1) {
                        gold = i;
                    }
                }

                if (argmax == gold) {
                    cor++;
                    all++;
                }else {
                    all++;
                }
            }
            double acc = (double) cor / all;
            System.out.println("acc: "+ format.format(100.*acc));

            CoNLLReader reader = new CoNLLReader(conllPath);
            ArrayList<GoldConfiguration> goldConfigurations =  reader.readData(Integer.MAX_VALUE,true,false,false, false,maps);
            double uas = 0;
            int las = 0;
            int a = 0;
            for(GoldConfiguration configuration:goldConfigurations){
                Configuration finalParse =  KBeamArcEagerParser.parseNeural(net,configuration.getSentence(),false,maps,dependencyRelations,1);
                // System.out.println(finalParse.score);

                for(int i=1;i<finalParse.sentence.size();i++){
                    a++;
                    if(finalParse.state.getHead(i)==configuration.head(i)){
                        uas++;
                    }
                }
            }

            uas =  uas / a;
            System.out.println("UAS: "+format.format(100.*uas));

            if(acc>bestAcc){
                bestAcc = acc;
                System.out.println("Saving the new model for iteration "+iter);
                FileOutputStream fos = new FileOutputStream(modelPath);
                GZIPOutputStream gz = new GZIPOutputStream(fos);
                ObjectOutput writer = new ObjectOutputStream(gz);
                writer.writeObject(new NNInfStruct(net,dependencyRelations.size(),maps,dependencyRelations,options));
                writer.writeObject(net);
                writer.close();
            }
        }

        /*
        EarlyStoppingModelSaver<ComputationGraph> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(nEpochs),
                        new ScoreImprovementEpochTerminationCondition(5))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(7, TimeUnit.DAYS),
                        new MaxScoreIterationTerminationCondition(7.5))  //Initial score is ~2.5
                .scoreCalculator(new DataSetLossCalculatorCG(devIter, true))
                .modelSaver(saver)
                .build();


        IEarlyStoppingTrainer trainer = new EarlyStoppingGraphTrainer(esConf, net, trainIter, new LoggingEarlyStoppingListener());
        EarlyStoppingResult result = trainer.fit();

        System.out.println(result.getTerminationDetails());



        devIter.reset();
        int cor = 0;
        int all = 0;
        while (devIter.hasNext()) {
            MultiDataSet t = devIter.next();
            INDArray[] features = t.getFeatures();
            INDArray[] predicted = net.output(features);

            double max = Double.NEGATIVE_INFINITY;
            int argmax = 0;
            int gold = 0;

            for (int i = 0; i < predicted[0].length(); i++) {
                double val = predicted[0].getDouble(i);
                if (val >= max) {
                    argmax = i;
                    max = val;
                }
                if (t.getLabels(0).getDouble(i) == 1) {
                    gold = i;
                }
            }

            if (argmax == gold) {
                cor++;
                all++;
            }else {
                all++;
            }
        }
        System.out.println("acc: "+ format.format(100.*(float) cor / all));


        CoNLLReader reader = new CoNLLReader(conllPath);
        ArrayList<GoldConfiguration> goldConfigurations =  reader.readData(10000,true,false,false, false,maps);
        double uas = 0;
        int las = 0;
        int a = 0;
        for(GoldConfiguration configuration:goldConfigurations){
            Configuration finalParse =  KBeamArcEagerParser.parseNeural(net,configuration.getSentence(),false,maps,dependencyRelations,1);
            // System.out.println(finalParse.score);

            for(int i=1;i<finalParse.sentence.size();i++){
                a++;
                if(finalParse.state.getHead(i)==configuration.head(i)){
                    uas++;
                }
            }
        }

        uas =  uas / a;
        System.out.println("UAS: "+ format.format(100.*uas));
        */
    }

    public static MultiDataSetIterator readMultiDataSetIterator(String[] path, int batchSize, int possibleOutputs) throws IOException, InterruptedException {
        int numLinesToSkip = 0;
        String fileDelimiter = ",";
        RecordReader[] featuresReader = new RecordReader[10];
        for (int i = 0; i < featuresReader.length; i++) {
            featuresReader[i] = new CSVRecordReader(numLinesToSkip, fileDelimiter);
            featuresReader[i].initialize(new FileSplit(new File(path[i])));
        }


        RecordReader labelsReader = new CSVRecordReader(numLinesToSkip, fileDelimiter);
        String labelsCsvPath = path[10];
        labelsReader.initialize(new FileSplit(new File(labelsCsvPath)));

        MultiDataSetIterator iterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
                .addReader("s0w", featuresReader[0])
                .addReader("b0w", featuresReader[1])
                .addReader("b1w", featuresReader[2])
                .addReader("b2w", featuresReader[3])
                .addReader("s0p", featuresReader[4])
                .addReader("b0p", featuresReader[5])
                .addReader("b1p", featuresReader[6])
                .addReader("b2p", featuresReader[7])
                .addReader("s0l", featuresReader[8])
                .addReader("sh0l", featuresReader[9])
                .addReader("csvLabels", labelsReader)
                .addInput("s0w") //Input: all columns from input reader
                .addInput("b0w")
                .addInput("b1w")
                .addInput("b2w")
                .addInput("s0p")
                .addInput("b0p")
                .addInput("b1p")
                .addInput("b2p")
                .addInput("s0l")
                .addInput("sh0l")
                .addOutputOneHot("csvLabels", 0, possibleOutputs)   //Output 2: column 4 -> convert to one-hot for classification
                .build();
        return iterator;

    }

    private static class LoggingEarlyStoppingListener implements EarlyStoppingListener<ComputationGraph> {

        private static Logger log = LoggerFactory.getLogger(LoggingEarlyStoppingListener.class);
        private int onStartCallCount = 0;
        private int onEpochCallCount = 0;
        private int onCompletionCallCount = 0;

        @Override
        public void onStart(EarlyStoppingConfiguration esConfig, ComputationGraph net) {
            log.info("EarlyStopping: onStart called");
            onStartCallCount++;
        }

        @Override
        public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration esConfig, ComputationGraph net) {
            log.info("EarlyStopping: onEpoch called (epochNum={}, score={}}", epochNum, score);
            onEpochCallCount++;
        }

        @Override
        public void onCompletion(EarlyStoppingResult esResult) {
            log.info("EarlyStopping: onCompletion called (result: {})", esResult);
            onCompletionCallCount++;
        }
    }

}
