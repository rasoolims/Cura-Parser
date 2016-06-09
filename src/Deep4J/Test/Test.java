package Deep4J.Test;


import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 4/15/16
 * Time: 5:02 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class Test {
    public static final String WORD_VECTORS_PATH = "/Users/msr/Downloads/word_10k.embed";
    public static final String WORD_VECTORS_PATH2 = "/Users/msr/Downloads/dep.embed";
    public static final String WORD_VECTORS_PATH3 = "/Users/msr/Downloads/pos.embed";

    public static void main(String[] args) throws Exception {
        System.out.println("hello world!");

        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH));
        WordVectors wordVectors2 = WordVectorSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH2));
        WordVectors wordVectors3 = WordVectorSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH3));

        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        int batchSize = 50;
        int seed = 123;
        double learningRate = 0.005;
        //Number of epochs (full passes of the data)
        int nEpochs = 30;

        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;
        int numHiddenNodes2 = 100;

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("src/saturn.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("src/saturn.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

        //log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes2)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(numHiddenNodes2).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        for (int n = 0; n < nEpochs; n++) {
            model.fit(trainIter);
        }


        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features, false);

            eval.eval(lables, predicted);

        }


        System.out.println(eval.stats());
        //------------------------------------------------------------------------------------
        //Training is complete. Code that follows is for plotting the data & predictions only


        double xMin = -15;
        double xMax = 15;
        double yMin = -15;
        double yMax = 15;

        //Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
        int nPointsPerAxis = 100;
        double[][] evalPoints = new double[nPointsPerAxis * nPointsPerAxis][2];
        int count = 0;
        for (int i = 0; i < nPointsPerAxis; i++) {
            for (int j = 0; j < nPointsPerAxis; j++) {
                double x = i * (xMax - xMin) / (nPointsPerAxis - 1) + xMin;
                double y = j * (yMax - yMin) / (nPointsPerAxis - 1) + yMin;

                evalPoints[count][0] = x;
                evalPoints[count][1] = y;

                count++;
            }
        }

        INDArray allXYPoints = Nd4j.create(evalPoints);
        INDArray predictionsAtXYPoints = model.output(allXYPoints);

    }
}
