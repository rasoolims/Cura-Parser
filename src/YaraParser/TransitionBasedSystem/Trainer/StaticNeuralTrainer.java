package YaraParser.TransitionBasedSystem.Trainer;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;

public class StaticNeuralTrainer {

    public StaticNeuralTrainer(String trainPath,int vocab1Size,  int vocab2Size , int vocab3Size,
                               int wordDimension, int posDimension, int depDimension,
                               int h1Dimension, int possibleOutputs, int nEpochs) throws Exception{
        //  vocab1Size =13, vocab2Size=8,   vocab3Size=2
        // wordDimension = 64,   posDimension=32,  depDimension=32
        // h1Dimension =100, possibleOutputs=4,  nEpochs=30

        double learningRate = 0.01;
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        int batchSize = 10;

        ComputationGraphConfiguration confComplex = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.SGD).momentum(0.9).regularization(true).l2(0.0001)
                .graphBuilder()
                .addInputs("s0w", "b0w", "b1w", "b2w", "s0p","b0p","b1p","b2p","s0l","sh0l")
                .addLayer("L1", new EmbeddingLayer.Builder().nIn(vocab1Size).nOut(wordDimension).build(), "s0w")
                .addLayer("L2", new EmbeddingLayer.Builder().nIn(vocab1Size).nOut(wordDimension).build(), "b0w")
                .addLayer("L3", new EmbeddingLayer.Builder().nIn(vocab1Size).nOut(wordDimension).build(), "b1w")
                .addLayer("L4", new EmbeddingLayer.Builder().nIn(vocab1Size).nOut(wordDimension).build(), "b2w")
                .addLayer("L5", new EmbeddingLayer.Builder().nIn(vocab2Size).nOut(posDimension).build(), "s0p")
                .addLayer("L6", new EmbeddingLayer.Builder().nIn(vocab2Size).nOut(posDimension).build(), "b0p")
                .addLayer("L7", new EmbeddingLayer.Builder().nIn(vocab2Size).nOut(posDimension).build(), "b1p")
                .addLayer("L8", new EmbeddingLayer.Builder().nIn(vocab2Size).nOut(posDimension).build(), "b2p")
                .addLayer("L9", new EmbeddingLayer.Builder().nIn(vocab3Size).nOut(depDimension).build(), "s0l")
                .addLayer("L10", new EmbeddingLayer.Builder().nIn(vocab3Size).nOut(depDimension).build(), "sh0l")
                .addVertex("concat", new MergeVertex(), "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10")
                .addLayer("h1", new DenseLayer.Builder().nIn(4*(wordDimension+posDimension)+2*depDimension)
                        .nOut(h1Dimension).activation("relu").build(), "concat")
                .addLayer("out", new OutputLayer.Builder().nIn(h1Dimension).nOut(possibleOutputs).activation("softmax").build(), "h1")
                .setOutputs("out")
                .build();


        int numLinesToSkip = 0;
        String fileDelimiter = ",";
        RecordReader rr = new CSVRecordReader(numLinesToSkip,fileDelimiter);
        rr.initialize(new FileSplit(new File(trainPath)));
        MultiDataSetIterator trainIterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
                .addReader("s0w",rr)
                .addReader("b0w",rr)
                .addReader("b1w",rr)
                .addReader("b2w",rr)
                .addReader("s0p",rr)
                .addReader("b0p",rr)
                .addReader("b1p",rr)
                .addReader("b2p",rr)
                .addReader("s0l",rr)
                .addReader("sh0l",rr)
                .addInput("s0w",1,1)
                .addInput("b0w",2,2)
                .addInput("b1w",3,3)
                .addInput("b2w",4,4)
                .addInput("s0p",5,5)
                .addInput("b0p",6,6)
                .addInput("b1p",7,7)
                .addInput("b2p",8,8)
                .addInput("s0l",9,9)
                .addInput("sh0l",10,10)
                .addReader("out",rr)
                .addOutput("out",0,0)
                .build();

        ComputationGraph net = new ComputationGraph(confComplex);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        for ( int n = 0; n < nEpochs; n++) {
            net.fit( trainIterator );
        }
    }
}
