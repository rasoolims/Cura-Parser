package edu.columbia.cs.nlp.Tests;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 11:07 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

import edu.columbia.cs.nlp.CuraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Learning.Activation.Enums.ActivationType;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.FirstHiddenLayer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.Layer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPTrainer;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.Initializer;
import edu.columbia.cs.nlp.CuraParser.Learning.WeightInit.NormalInit;
import edu.columbia.cs.nlp.CuraParser.Structures.Enums.EmbeddingTypes;
import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.Structures.NeuralTrainingInstance;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.ParserType;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Trainer.GreedyTrainer;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class GradientTest {
    final String txtFilePath = "/tmp/tmp.tmp";
    final String embedFilePath = "/tmp/tmp2.tmp";
    final String conllText = "1\tThe\t_\tDT\tDT\t_\t2\tdet\t_\t_\n" +
            "2\tbill\t_\tNN\tNN\t_\t3\tnsubj\t_\t_\n" +
            "3\tintends\t_\tVBZ\tVBZ\t_\t0\troot\t_\t_\n" +
            "4\tto\t_\tTO\tTO\t_\t5\taux\t_\t_\n" +
            "5\trestrict\t_\tVB\tVB\t_\t3\txcomp\t_\t_\n" +
            "6\tthe\t_\tDT\tDT\t_\t7\tdet\t_\t_\n" +
            "7\tRTC\t_\tNNP\tNNP\t_\t5\tdobj\t_\t_\n" +
            "8\tto\t_\tTO\tTO\t_\t5\tprep\t_\t_\n" +
            "9\tTreasury\t_\tNNP\tNNP\t_\t10\tnn\t_\t_\n" +
            "10\tborrowings\t_\tNNS\tNNS\t_\t8\tpobj\t_\t_\n" +
            "11\tonly\t_\tRB\tRB\t_\t10\tadvmod\t_\t_\n" +
            "12\t,\t_\t,\t,\t_\t3\tpunct\t_\t_\n" +
            "13\tunless\t_\tIN\tIN\t_\t16\tmark\t_\t_\n" +
            "14\tthe\t_\tDT\tDT\t_\t15\tdet\t_\t_\n" +
            "15\tagency\t_\tNN\tNN\t_\t16\tnsubj\t_\t_\n" +
            "16\treceives\t_\tVBZ\tVBZ\t_\t3\tadvcl\t_\t_\n" +
            "17\tspecific\t_\tJJ\tJJ\t_\t19\tamod\t_\t_\n" +
            "18\tcongressional\t_\tJJ\tJJ\t_\t19\tamod\t_\t_\n" +
            "19\tauthorization\t_\tNN\tNN\t_\t16\tdobj\t_\t_\n" +
            "20\t.\t_\t.\t.\t_\t3\tpunct\t_\t_\n" +
            "\n" +
            "\n" +
            "1\tBut\t_\tCC\tCC\t_\t24\tcc\t_\t_\n" +
            "2\tRobert\t_\tNNP\tNNP\t_\t3\tnn\t_\t_\n" +
            "3\tGabele\t_\tNNP\tNNP\t_\t24\tnsubj\t_\t_\n" +
            "4\t,\t_\t,\t,\t_\t3\tpunct\t_\t_\n" +
            "5\tpresident\t_\tNN\tNN\t_\t3\tappos\t_\t_\n" +
            "6\tof\t_\tIN\tIN\t_\t5\tprep\t_\t_\n" +
            "7\tInvest\\/Net\t_\tNNP\tNNP\t_\t6\tpobj\t_\t_\n" +
            "8\t,\t_\t,\t,\t_\t7\tpunct\t_\t_\n" +
            "9\ta\t_\tDT\tDT\t_\t15\tdet\t_\t_\n" +
            "10\tNorth\t_\tNNP\tNNP\t_\t15\tdep\t_\t_\n" +
            "11\tMiami\t_\tNNP\tNNP\t_\t10\tnn\t_\t_\n" +
            "12\t,\t_\t,\t,\t_\t10\tpunct\t_\t_\n" +
            "13\tFla.\t_\tNNP\tNNP\t_\t10\tdep\t_\t_\n" +
            "14\t,\t_\t,\t,\t_\t10\tpunct\t_\t_\n" +
            "15\tcompany\t_\tNN\tNN\t_\t7\tappos\t_\t_\n" +
            "16\tthat\t_\tWDT\tWDT\t_\t17\tnsubj\t_\t_\n" +
            "17\tpackages\t_\tVBZ\tVBZ\t_\t15\trcmod\t_\t_\n" +
            "18\tand\t_\tCC\tCC\t_\t17\tcc\t_\t_\n" +
            "19\tsells\t_\tVBZ\tVBZ\t_\t17\tconj\t_\t_\n" +
            "20\tthe\t_\tDT\tDT\t_\t22\tdet\t_\t_\n" +
            "21\tinsider-trading\t_\tNN\tNN\t_\t22\tnn\t_\t_\n" +
            "22\tdata\t_\tNNS\tNNS\t_\t17\tdobj\t_\t_\n" +
            "23\t,\t_\t,\t,\t_\t3\tpunct\t_\t_\n" +
            "24\tsaid\t_\tVBD\tVBD\t_\t0\troot\t_\t_\n" +
            "25\tthe\t_\tDT\tDT\t_\t26\tdet\t_\t_\n" +
            "26\tproposal\t_\tNN\tNN\t_\t28\tnsubjpass\t_\t_\n" +
            "27\tis\t_\tVBZ\tVBZ\t_\t28\tauxpass\t_\t_\n" +
            "28\tworded\t_\tVBN\tVBN\t_\t24\tccomp\t_\t_\n" +
            "29\tso\t_\tRB\tRB\t_\t30\tadvmod\t_\t_\n" +
            "30\tvaguely\t_\tRB\tRB\t_\t28\tadvmod\t_\t_\n" +
            "31\tthat\t_\tIN\tIN\t_\t35\tdep\t_\t_\n" +
            "32\tkey\t_\tJJ\tJJ\t_\t33\tamod\t_\t_\n" +
            "33\tofficials\t_\tNNS\tNNS\t_\t35\tnsubj\t_\t_\n" +
            "34\tmay\t_\tMD\tMD\t_\t35\taux\t_\t_\n" +
            "35\tfail\t_\tVB\tVB\t_\t30\tccomp\t_\t_\n" +
            "36\tto\t_\tTO\tTO\t_\t37\taux\t_\t_\n" +
            "37\tfile\t_\tVB\tVB\t_\t35\txcomp\t_\t_\n" +
            "38\tthe\t_\tDT\tDT\t_\t39\tdet\t_\t_\n" +
            "39\treports\t_\tNNS\tNNS\t_\t37\tdobj\t_\t_\n" +
            "40\t.\t_\t.\t.\t_\t24\tpunct\t_\t_\n" +
            "\n" +
            "1\tMany\t_\tJJ\tJJ\t_\t2\tamod\t_\t_\n" +
            "2\tinvestors\t_\tNNS\tNNS\t_\t3\tnsubj\t_\t_\n" +
            "3\twrote\t_\tVBD\tVBD\t_\t0\troot\t_\t_\n" +
            "4\tasking\t_\tVBG\tVBG\t_\t3\txcomp\t_\t_\n" +
            "5\tthe\t_\tDT\tDT\t_\t6\tdet\t_\t_\n" +
            "6\tSEC\t_\tNNP\tNNP\t_\t4\tdobj\t_\t_\n" +
            "7\tto\t_\tTO\tTO\t_\t8\taux\t_\t_\n" +
            "8\trequire\t_\tVB\tVB\t_\t4\txcomp\t_\t_\n" +
            "9\tinsiders\t_\tNNS\tNNS\t_\t11\tnsubj\t_\t_\n" +
            "10\tto\t_\tTO\tTO\t_\t11\taux\t_\t_\n" +
            "11\treport\t_\tVB\tVB\t_\t8\txcomp\t_\t_\n" +
            "12\ttheir\t_\tPRP$\tPRP$\t_\t13\tposs\t_\t_\n" +
            "13\tpurchases\t_\tNNS\tNNS\t_\t11\tdobj\t_\t_\n" +
            "14\tand\t_\tCC\tCC\t_\t13\tcc\t_\t_\n" +
            "15\tsales\t_\tNNS\tNNS\t_\t13\tconj\t_\t_\n" +
            "16\timmediately\t_\tRB\tRB\t_\t11\tneg\t_\t_\n" +
            "17\t,\t_\t,\t,\t_\t16\tpunct\t_\t_\n" +
            "18\tnot\t_\tRB\tRB\t_\t16\tdep\t_\t_\n" +
            "19\ta\t_\tDT\tDT\t_\t20\tdet\t_\t_\n" +
            "20\tmonth\t_\tNN\tNN\t_\t21\tnpadvmod\t_\t_\n" +
            "21\tlater\t_\tRB\tRB\t_\t16\tadvmod\t_\t_\n" +
            "22\t.\t_\t.\t.\t_\t3\tpunct\t_\t_\n";


    @Test
    public void TestWordEmbeddingGradients() throws Exception {
        int[] h2Sizes = new int[]{0, 5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                if (type == ActivationType.RandomRelu)
                    continue; // because of randomness we cannot make sure what really had happened.
                writeText();
                Options options = new Options();
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.trainingOptions.trainFile = txtFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options
                        .generalProperties.lowercase, "", 0, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                        .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(0).nOut());
                normalInit.init(network.layer(0).getB());

                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);

                classifier.cost(instances, 1, gradients, savedGradients);

                double eps = 0.000001;
                for (int i = 0; i < network.getNumWordLayers(); i++) {
                    for (int slot = 0; slot < network.getwDim(); slot++) {
                        int tok = (int) instances.get(0).getFeatures()[i];
                        double gradForTok = gradients.getWordEmbedding()[tok][slot] / instances.size();

                        MLPNetwork plusNetwork = network.clone();
                        purturb(plusNetwork, EmbeddingTypes.WORD, tok, slot, eps);
                        plusNetwork.preCompute();
                        MLPNetwork negNetwork = network.clone();
                        purturb(negNetwork, EmbeddingTypes.WORD, tok, slot, -eps);
                        negNetwork.preCompute();
                        double diff = 0;

                        for (NeuralTrainingInstance instance : instances) {
                            double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel(), true);
                            double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel(), true);

                            int goldLabel = instance.gold();
                            diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                        }
                        diff /= (2 * eps * instances.size());

                        System.out.println(gradForTok + "\t" + diff);
                        assert Math.abs(gradForTok - diff) <= 0.0000001;
                    }
                }
            }
        }
    }

    @Test
    public void TestPretrainedWordEmbeddingGradients() throws Exception {
        int[] h2Sizes = new int[]{0, 5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                if (type == ActivationType.RandomRelu)
                    continue; // because of randomness we cannot make sure what really had happened.
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.trainingOptions.trainFile = txtFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options
                        .generalProperties.lowercase, "", 0, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                        .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(0).nOut());
                normalInit.init(network.layer(0).getB());
                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);
                classifier.cost(instances, 1, gradients, savedGradients);

                double eps = 0.000001;
                for (int i = 0; i < network.getNumWordLayers(); i++) {
                    for (int slot = 0; slot < network.getwDim(); slot++) {
                        int tok = (int) instances.get(0).getFeatures()[i];
                        double gradForTok = gradients.getWordEmbedding()[tok][slot] / instances.size();

                        MLPNetwork plusNetwork = network.clone();
                        purturb(plusNetwork, EmbeddingTypes.WORD, tok, slot, eps);
                        plusNetwork.preCompute();
                        MLPNetwork negNetwork = network.clone();
                        purturb(negNetwork, EmbeddingTypes.WORD, tok, slot, -eps);
                        negNetwork.preCompute();
                        double diff = 0;

                        for (NeuralTrainingInstance instance : instances) {
                            double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel(), true);
                            double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel(), true);

                            int goldLabel = instance.gold();
                            diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                        }
                        diff /= (2 * eps * instances.size());

                        if (Math.abs(gradForTok - diff) > 0.0000001)
                            System.out.println(type + "\t" + gradForTok + "\t" + diff);
                        assert Math.abs(gradForTok - diff) <= 0.0000001;
                    }
                }
            }
        }
    }

    @Test
    public void TestWordEmbeddingUpdates() throws Exception {
        int[] h2Sizes = new int[]{0, 5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.trainingOptions.trainFile = txtFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options
                        .generalProperties.lowercase, "", 1, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                        .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(0).nOut());
                normalInit.init(network.layer(0).getB());
                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);
                classifier.cost(instances, 1, gradients, savedGradients);

                double[] oovEmbedding = Utils.clone(network.getWordEmbedding()[0]);
                double[] nullEmbedding = Utils.clone(network.getWordEmbedding()[1]);
                double[] rootEmbedding = Utils.clone(network.getWordEmbedding()[2]);
                double[] simpleWordEmbedding = Utils.clone(network.getWordEmbedding()[3]);
                for (int i = 0; i < 3; i++) {
                    classifier.fit(instances, i, true);
                }

                assert !Utils.equals(network.getWordEmbedding()[0], oovEmbedding);
                assert !Utils.equals(network.getWordEmbedding()[1], nullEmbedding);
                assert !Utils.equals(network.getWordEmbedding()[2], rootEmbedding);
                assert !Utils.equals(network.getWordEmbedding()[3], simpleWordEmbedding);
            }
        }
    }

    @Test
    public void TestPretrainedtWordEmbeddingUpdates() throws Exception {
        int[] h2Sizes = new int[]{0, 5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.trainingOptions.trainFile = txtFilePath;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options
                        .generalProperties.lowercase, "", 1, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                        .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(0).nOut());
                normalInit.init(network.layer(0).getB());
                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);
                classifier.cost(instances, 1, gradients, savedGradients);

                double[] oovEmbedding = Utils.clone(network.getWordEmbedding()[0]);
                double[] nullEmbedding = Utils.clone(network.getWordEmbedding()[1]);
                double[] rootEmbedding = Utils.clone(network.getWordEmbedding()[2]);
                double[] simpleWordEmbedding = Utils.clone(network.getWordEmbedding()[3]);
                for (int i = 0; i < 3; i++) {
                    classifier.fit(instances, i, true);
                }

                assert !Utils.equals(network.getWordEmbedding()[0], oovEmbedding);
                assert !Utils.equals(network.getWordEmbedding()[1], nullEmbedding);
                assert !Utils.equals(network.getWordEmbedding()[2], rootEmbedding);
                assert !Utils.equals(network.getWordEmbedding()[3], simpleWordEmbedding);
            }
        }
    }

    @Test
    public void TestPOSEmbeddingGradients() throws Exception {
        int[] h2Sizes = new int[]{0, 5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                if (type == ActivationType.RandomRelu)
                    continue; // because of randomness we cannot make sure what really had happened.
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.trainingOptions.trainFile = txtFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options
                        .generalProperties.lowercase, "", 0, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                        .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(0).nOut());
                normalInit.init(network.layer(0).getB());
                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);
                classifier.cost(instances, 1, gradients, savedGradients);

                double eps = 0.000001;
                for (int i = network.getNumWordLayers(); i < network.getNumWordLayers() + network.getNumPosLayers(); i++) {
                    for (int slot = 0; slot < network.getpDim(); slot++) {
                        int tok = (int) instances.get(0).getFeatures()[i];
                        double gradForTok = gradients.getPosEmbedding()[tok][slot] / instances.size();

                        MLPNetwork plusNetwork = network.clone();
                        purturb(plusNetwork, EmbeddingTypes.POS, tok, slot, eps);
                        plusNetwork.preCompute();
                        MLPNetwork negNetwork = network.clone();
                        purturb(negNetwork, EmbeddingTypes.POS, tok, slot, -eps);
                        negNetwork.preCompute();
                        double diff = 0;

                        for (NeuralTrainingInstance instance : instances) {
                            double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel(), true);
                            double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel(), true);

                            int goldLabel = instance.gold();
                            diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                        }
                        diff /= (2 * eps * instances.size());

                        System.out.println(gradForTok + "\t" + diff);
                        assert Math.abs(gradForTok - diff) <= 0.0000001;
                    }
                }
            }
        }
    }

    @Test
    public void TestLabelEmbeddingGradients() throws Exception {
        int[] h2Sizes = new int[]{0, 5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                if (type == ActivationType.RandomRelu)
                    continue; // because of randomness we cannot make sure what really had happened.
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.trainingOptions.trainFile = txtFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options
                        .generalProperties.lowercase, "", 0, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                        .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(0).nOut());
                normalInit.init(network.layer(0).getB());
                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);
                classifier.cost(instances, 1, gradients, savedGradients);

                double eps = 0.000001;
                for (int i = network.getNumWordLayers() + network.getNumPosLayers();
                     i < network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers(); i++) {
                    for (int slot = 0; slot < network.getDepDim(); slot++) {
                        int tok = (int) instances.get(0).getFeatures()[i];
                        double gradForTok = gradients.getDepEmbedding()[tok][slot] / instances.size();

                        MLPNetwork plusNetwork = network.clone();
                        purturb(plusNetwork, EmbeddingTypes.DEPENDENCY, tok, slot, eps);
                        plusNetwork.preCompute();
                        MLPNetwork negNetwork = network.clone();
                        purturb(negNetwork, EmbeddingTypes.DEPENDENCY, tok, slot, -eps);
                        negNetwork.preCompute();
                        double diff = 0;

                        for (NeuralTrainingInstance instance : instances) {
                            double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel(), true);
                            double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel(), true);

                            int goldLabel = instance.gold();
                            diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                        }
                        diff /= (2 * eps * instances.size());

                        System.out.println(type + "\t" + gradForTok + "\t" + diff);
                        assert Math.abs(gradForTok - diff) <= 0.0000001;
                    }
                }
            }
        }
    }

    @Test
    public void TestHiddenLayerGradients() throws Exception {
        int[] h2Sizes = new int[]{0, 5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                if (type == ActivationType.RandomRelu)
                    continue; // because of randomness we cannot make sure what really had happened.
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.trainingOptions.trainFile = txtFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options
                        .generalProperties.lowercase, "", 0, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                        .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(0).nOut());
                normalInit.init(network.layer(0).getB());
                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);
                classifier.cost(instances, 1, gradients, savedGradients);

                double eps = 0.000001;
                for (int i = 0; i < network.layer(0).nOut(); i++) {
                    for (int slot = 0; slot < network.layer(0).nIn(); slot++) {
                        double gradForTok = gradients.layer(0).getW()[i][slot] / instances.size();

                        MLPNetwork plusNetwork = network.clone();
                        purturb(plusNetwork, EmbeddingTypes.HIDDENLAYER, i, slot, eps);
                        plusNetwork.preCompute();
                        MLPNetwork negNetwork = network.clone();
                        purturb(negNetwork, EmbeddingTypes.HIDDENLAYER, i, slot, -eps);
                        negNetwork.preCompute();
                        double diff = 0;

                        for (NeuralTrainingInstance instance : instances) {
                            double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel(), true);
                            double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel(), true);

                            int goldLabel = instance.gold();
                            diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                        }
                        diff /= (2 * eps * instances.size());

                        System.out.println(gradForTok + "\t" + diff);
                        assert Math.abs(gradForTok - diff) <= 0.0000001;
                    }
                }
            }
        }
    }

    @Test
    public void TestHiddenLayerBiasGradients() throws Exception {
        int[] h2Sizes = new int[]{0, 5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                if (type == ActivationType.RandomRelu)
                    continue; // because of randomness we cannot make sure what really had happened.
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.trainingOptions.trainFile = txtFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options
                        .generalProperties.lowercase, "", 0, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                        .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(0).nOut());
                normalInit.init(network.layer(0).getB());
                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);
                classifier.cost(instances, 1, gradients, savedGradients);

                double eps = 0.000001;
                for (int i = 0; i < network.layer(0).nOut(); i++) {
                    int tok = i;
                    double gradForTok = gradients.layer(0).getB()[tok] / instances.size();

                    MLPNetwork plusNetwork = network.clone();
                    purturb(plusNetwork, EmbeddingTypes.HIDDENLAYERBIAS, tok, -1, eps);
                    plusNetwork.preCompute();
                    MLPNetwork negNetwork = network.clone();
                    purturb(negNetwork, EmbeddingTypes.HIDDENLAYERBIAS, tok, -1, -eps);
                    negNetwork.preCompute();
                    double diff = 0;

                    for (NeuralTrainingInstance instance : instances) {
                        double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel(), true);
                        double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel(), true);

                        int goldLabel = instance.gold();
                        diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                    }
                    diff /= (2 * eps * instances.size());

                    System.out.println(gradForTok + "\t" + diff);
                    assert Math.abs(gradForTok - diff) <= 0.0000001;
                }
            }
        }
    }

    @Test
    public void TestSecondHiddenLayerGradients() throws Exception {
        int[] h2Sizes = new int[]{5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                if (type == ActivationType.RandomRelu)
                    continue; // because of randomness we cannot make sure what really had happened.
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.trainingOptions.trainFile = txtFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options
                        .generalProperties.lowercase, "", 0, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                        .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(1).nOut());
                normalInit.init(network.layer(1).getB());
                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);
                classifier.cost(instances, 1, gradients, savedGradients);

                double eps = 0.000001;
                for (int i = 0; i < network.layer(1).nOut(); i++) {
                    for (int slot = 0; slot < network.layer(1).nIn(); slot++) {
                        double gradForTok = gradients.layer(1).getW()[i][slot] / instances.size();

                        MLPNetwork plusNetwork = network.clone();
                        purturb(plusNetwork, EmbeddingTypes.SECONDHIDDENLAYER, i, slot, eps);
                        plusNetwork.preCompute();
                        MLPNetwork negNetwork = network.clone();
                        purturb(negNetwork, EmbeddingTypes.SECONDHIDDENLAYER, i, slot, -eps);
                        negNetwork.preCompute();
                        double diff = 0;

                        for (NeuralTrainingInstance instance : instances) {
                            double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel(), true);
                            double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel(), true);

                            int goldLabel = instance.gold();
                            diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                        }
                        diff /= (2 * eps * instances.size());

                        System.out.println(gradForTok + "\t" + diff);
                        assert Math.abs(gradForTok - diff) <= 0.0000001;
                    }
                }
            }
        }
    }

    @Test
    public void TestSecondHiddenLayerBiasGradients() throws Exception {
        int[] h2Sizes = new int[]{5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                if (type == ActivationType.RandomRelu)
                    continue; // because of randomness we cannot make sure what really had happened.
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.trainingOptions.trainFile = txtFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options
                        .generalProperties.lowercase, "", 0, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                        .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(1).nOut());
                normalInit.init(network.layer(1).getB());
                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);
                classifier.cost(instances, 1, gradients, savedGradients);

                double eps = 0.000001;
                for (int i = 0; i < network.layer(1).nOut(); i++) {
                    double gradForTok = gradients.layer(1).getB()[i] / instances.size();

                    MLPNetwork plusNetwork = network.clone();
                    purturb(plusNetwork, EmbeddingTypes.SECONDHIDDENLAYERBIAS, i, -1, eps);
                    plusNetwork.preCompute();
                    MLPNetwork negNetwork = network.clone();
                    purturb(negNetwork, EmbeddingTypes.SECONDHIDDENLAYERBIAS, i, -1, -eps);
                    negNetwork.preCompute();
                    double diff = 0;

                    for (NeuralTrainingInstance instance : instances) {
                        double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel(), true);
                        double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel(), true);

                        int goldLabel = instance.gold();
                        diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                    }
                    diff /= (2 * eps * instances.size());

                    System.out.println(gradForTok + "\t" + diff);
                    assert Math.abs(gradForTok - diff) <= 0.0000001;
                }
            }
        }
    }


    @Test
    public void TestSoftmaxLayerGradients() throws Exception {
        int[] h2Sizes = new int[]{0, 5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                if (type == ActivationType.RandomRelu)
                    continue; // because of randomness we cannot make sure what really had happened.
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.trainingOptions.trainFile = txtFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options
                        .generalProperties.lowercase, "", 0, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                        .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(0).nOut());
                normalInit.init(network.layer(0).getB());
                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);
                classifier.cost(instances, 1, gradients, savedGradients);

                double eps = 0.000001;
                for (int i = 0; i < network.getNumOutputs(); i++) {
                    for (int slot = 0; slot < network.layer(network.numLayers() - 1).nIn(); slot++) {
                        double gradForTok = gradients.layer(gradients.numLayers() - 1).getW()[i][slot] / instances.size();

                        MLPNetwork plusNetwork = network.clone();
                        purturb(plusNetwork, EmbeddingTypes.SOFTMAX, i, slot, eps);

                        plusNetwork.preCompute();
                        MLPNetwork negNetwork = network.clone();
                        purturb(negNetwork, EmbeddingTypes.SOFTMAX, i, slot, -eps);
                        plusNetwork.preCompute();
                        double diff = 0;

                        for (NeuralTrainingInstance instance : instances) {
                            double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel(), true);
                            double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel(), true);

                            int goldLabel = instance.gold();
                            diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                        }
                        diff /= (2 * eps * instances.size());

                        System.out.println(gradForTok + "\t" + diff);
                        assert Math.abs(gradForTok - diff) <= 0.0000001;
                    }
                }
            }
        }
    }

    @Test
    public void TestSoftmaxLayerBiasGradients() throws Exception {
        int[] h2Sizes = new int[]{0, 5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                if (type == ActivationType.RandomRelu)
                    continue; // because of randomness we cannot make sure what really had happened.
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.networkProperties.outputBiasTerm = true;
                options.networkProperties.outputBiasTerm = true;
                options.trainingOptions.trainFile = txtFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options
                        .generalProperties.lowercase, "", 0, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                        .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(0).nOut());
                normalInit.init(network.layer(0).getB());
                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);
                classifier.cost(instances, 1, gradients, savedGradients);

                double eps = 0.000001;
                for (int i = 0; i < network.getNumOutputs(); i++) {
                    double gradForTok = gradients.layer(gradients.numLayers() - 1).getB()[i] / instances.size();

                    MLPNetwork plusNetwork = network.clone();
                    purturb(plusNetwork, EmbeddingTypes.SOFTMAXBIAS, i, -1, eps);
                    plusNetwork.preCompute();
                    MLPNetwork negNetwork = network.clone();
                    purturb(negNetwork, EmbeddingTypes.SOFTMAXBIAS, i, -1, -eps);
                    negNetwork.preCompute();
                    double diff = 0;

                    for (NeuralTrainingInstance instance : instances) {
                        double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel(), true);
                        double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel(), true);

                        int goldLabel = instance.gold();
                        diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                    }
                    diff /= (2 * eps * instances.size());

                    if (Math.abs(gradForTok - diff) > 0.0000001)
                        System.out.println(type + "\t" + gradForTok + "\t" + diff);
                    assert Math.abs(gradForTok - diff) <= 0.0000001;
                }
            }
        }
    }

    @Test
    public void TestMultiThreadGradients() throws Exception {
        int[] h2Sizes = new int[]{0, 5, 10, 15};
        for (int h2Size : h2Sizes) {
            for (ActivationType type : ActivationType.values()) {
                if (type == ActivationType.RandomRelu)
                    continue; // because of randomness we cannot make sure what really had happened.
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.networkProperties.hiddenLayer2Size = h2Size;
                options.trainingOptions.trainFile = txtFilePath;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled,
                        options.generalProperties.lowercase, "", 0, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled,
                        options.generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
                // to make sure that RELU can be really effective.
                Initializer normalInit = new NormalInit(new Random(), network.layer(0).nOut());
                normalInit.init(network.layer(0).getB());
                double[][][] savedGradients = network.instantiateSavedGradients();
                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();
                MLPNetwork gradients = network.clone(true, false);
                classifier.cost(instances, instances.size(), gradients, savedGradients);
                MLPTrainer classifierMultiThread = new MLPTrainer(network, options);
                classifierMultiThread.cost(instances);

                final ArrayList<Layer> gradientsMultiThread = classifierMultiThread.getGradients();
                double eps = 1e-15;

                ArrayList<Layer> allMatrices1 = gradients.getLayers();

                for (int i = 0; i < allMatrices1.size(); i++) {
                    double[][] m1 = allMatrices1.get(i).getW();
                    double[][] m2 = gradientsMultiThread.get(i).getW();
                    if (m1 == null) continue;

                    for (int j = 0; j < m1.length; j++)
                        for (int h = 0; h < m1[j].length; h++) {
                            if (Math.abs(m1[j][h] - m2[j][h]) > eps)
                                System.out.println(i + "\t" + j + "\t" + h + "\t" + m1[j][h] + "\t" + m2[j][h]);
                            assert Math.abs(m1[j][h] - m2[j][h]) <= eps;
                        }
                }

                for (int i = 0; i < allMatrices1.size(); i++) {
                    double[] v1 = allMatrices1.get(i).getB();
                    double[] v2 = gradientsMultiThread.get(i).getB();
                    if (v1 == null) continue;

                    for (int j = 0; j < v1.length; j++) {
                        if (Math.abs(v1[j] - v2[j]) > eps)
                            System.out.println(i + "\t" + j + "\t" + v1[j] + "\t" + v2[j]);
                        assert Math.abs(v1[j] - v2[j]) <= eps;
                    }
                }
            }
        }
    }

    private void purturb(MLPNetwork network, EmbeddingTypes type, int tokNum, int slotNum, double eps) {
        if (type == EmbeddingTypes.WORD || type == EmbeddingTypes.POS || type == EmbeddingTypes.DEPENDENCY) {
            ((FirstHiddenLayer) network.layer(0)).modify(type, tokNum, slotNum, eps);
        } else if (type == EmbeddingTypes.HIDDENLAYER) {
            network.layer(0).modifyW(tokNum, slotNum, eps);
        } else if (type == EmbeddingTypes.SECONDHIDDENLAYER) {
            network.layer(1).modifyW(tokNum, slotNum, eps);
        } else if (type == EmbeddingTypes.SOFTMAX) {
            network.layer(network.numLayers() - 1).modifyW(tokNum, slotNum, eps);
        } else if (type == EmbeddingTypes.HIDDENLAYERBIAS) {
            network.layer(0).modifyB(tokNum, eps);
        } else if (type == EmbeddingTypes.SECONDHIDDENLAYERBIAS) {
            network.layer(1).modifyB(tokNum, eps);
        } else if (type == EmbeddingTypes.SOFTMAXBIAS) {
            network.layer(network.numLayers() - 1).modifyB(tokNum, eps);
        }
    }

    private void writeText() throws Exception {
        BufferedWriter writer = new BufferedWriter(new FileWriter(txtFilePath));
        writer.write(conllText);
        writer.close();
    }

    private void writeWordEmbedText() throws Exception {
        String embedText = "the\t0.1\t-.01\t.5\t.6\t-.36\t.001\t.45\t-.4\nto\t0.1\t-.01\t.3\t-.6\t-.56\t.021\t.41\t.4\nfail\t0.3\t-.011\t.51\t" +
                ".26\t-.36\t.1\t-.45\t-.4\n";
        BufferedWriter writer = new BufferedWriter(new FileWriter(embedFilePath));
        writer.write(embedText);
        writer.close();
    }
}
