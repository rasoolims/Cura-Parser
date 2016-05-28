package YaraParser.TransitionBasedSystem.Trainer;

import YaraParser.Structures.IndexMaps;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.zip.GZIPOutputStream;

public class StaticNeuralTrainer {

    public StaticNeuralTrainer(String[] trainFeatPath, IndexMaps maps,
                               int wordDimension, int posDimension, int depDimension,
                               int h1Dimension, int possibleOutputs, int nEpochs
    ,String modelPath) throws Exception{
        int vocab1Size = maps.vocabSize()+2;
        int vocab2Size = maps.posSize()+2;
        int vocab3Size =maps.relSize()+1;
        //  vocab1Size =13, vocab2Size=8,   vocab3Size=2
        // wordDimension = 64,   posDimension=32,  depDimension=32
        // h1Dimension =100, possibleOutputs=4,  nEpochs=30

        double learningRate = 0.01;
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        int batchSize = 1;

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
        RecordReader[] featuresReader = new RecordReader[10];
        for(int i=0;i<featuresReader.length;i++) {
           featuresReader[i] = new CSVRecordReader(numLinesToSkip, fileDelimiter);
            featuresReader[i].initialize(new FileSplit(new File(trainFeatPath[i])));
        }


        RecordReader labelsReader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
        String labelsCsvPath =trainFeatPath[10];
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

        ComputationGraph net = new ComputationGraph(confComplex);
        net.init();
        net.setListeners(new ScoreIterationListener(100));

        for ( int n = 0; n < nEpochs; n++) {
            net.fit( iterator );
        }



         featuresReader = new RecordReader[10];
        for(int i=0;i<featuresReader.length;i++) {
            featuresReader[i] = new CSVRecordReader(numLinesToSkip, fileDelimiter);
            featuresReader[i].initialize(new FileSplit(new File(trainFeatPath[i])));
        }


         labelsReader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
         labelsCsvPath =trainFeatPath[10];
        labelsReader.initialize(new FileSplit(new File(labelsCsvPath)));

        MultiDataSetIterator testIter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
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

        int cor = 0;
        int all =0;
        while(testIter.hasNext()) {
            MultiDataSet t = testIter.next();
            INDArray[] features = t.getFeatures();
            System.out.print("HI");
            INDArray[] predicted = net.output(features);

            double max = Double.NEGATIVE_INFINITY;
            int argmax = 0;
            int gold = 0;


            for (int i = 0; i < predicted[0].length(); i++) {
              double val =  predicted[0].getDouble(i);
                if(val>=max){
                    argmax = i;
                    max = val;
                }
                if(t.getLabels(0).getDouble(i)==1){
                    gold =i;
                }
            }

            if(argmax==gold)
                cor++;
            else
                System.out.println("-->"+gold+"->"+argmax);
            all++;
        }

        System.out.println((float) cor/all);
        FileOutputStream fos = new FileOutputStream(modelPath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);

        ObjectOutput writer = new ObjectOutputStream(gz);
        writer.writeObject(maps);
        writer.writeObject(net);
        writer.close();
    }
}
