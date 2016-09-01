package edu.columbia.cs.nlp.Tests;

import edu.columbia.cs.nlp.CuraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.FirstHiddenLayer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.Layer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.Layers.WordEmbeddingLayer;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
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

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/5/16
 * Time: 10:49 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class MLPNetworkTest {
    final String txtFilePath = "/tmp/tmp.tmp";
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
    public void testClone() throws Exception {
        writeText();
        Options options = new Options();
        options.networkProperties.hiddenLayer1Size = 10;
        options.generalProperties.inputFile = txtFilePath;
        IndexMaps maps = CoNLLReader.createIndices(options.generalProperties.inputFile, options.generalProperties.labeled,
                options.generalProperties.lowercase, "", 0, false);
        ArrayList<Integer> dependencyLabels = new ArrayList<>();
        for (int lab = 0; lab < maps.relSize(); lab++)
            dependencyLabels.add(lab);
        CoNLLReader reader = new CoNLLReader(options.generalProperties.inputFile);
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled,
                options.generalProperties.rootFirst, options.generalProperties.lowercase, maps);
        int wDim = 8;
        int pDim = 4;
        int lDim = 6;
        GreedyTrainer trainer = new GreedyTrainer(options, dependencyLabels,
                maps.labelNullIndex, maps.rareWords);
        List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
        maps.constructPreComputeMap(instances, 22, 10000);

        MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
        network.preCompute();
        MLPNetwork clonedNetwork = network.clone();
        clonedNetwork.emptyPrecomputedMap();

        assert clonedNetwork.numLayers() == network.numLayers();
        for (int i = 0; i < clonedNetwork.numLayers(); i++) {
            Layer layer1 = network.layer(i);
            Layer layer2 = clonedNetwork.layer(i);
            assert Utils.equals(layer1.getW(), layer2.getW());
            if (layer1.getB() != null)
                assert Utils.equals(layer1.getB(), layer2.getB());

            if (layer1 instanceof FirstHiddenLayer) {
                WordEmbeddingLayer wordEmbeddingLayer1 = ((FirstHiddenLayer) layer1).getWordEmbeddings();
                Layer posEmbeddingLayer1 = ((FirstHiddenLayer) layer1).getPosEmbeddings();
                Layer depEmbeddingLayer1 = ((FirstHiddenLayer) layer1).getDepEmbeddings();

                WordEmbeddingLayer wordEmbeddingLayer2 = ((FirstHiddenLayer) layer2).getWordEmbeddings();
                Layer posEmbeddingLayer2 = ((FirstHiddenLayer) layer2).getPosEmbeddings();
                Layer depEmbeddingLayer2 = ((FirstHiddenLayer) layer2).getDepEmbeddings();

                assert Utils.equals(wordEmbeddingLayer1.getW(), wordEmbeddingLayer2.getW());
                assert Utils.equals(posEmbeddingLayer1.getW(), posEmbeddingLayer2.getW());
                assert Utils.equals(depEmbeddingLayer1.getW(), depEmbeddingLayer2.getW());
            }
        }
    }


    @Test
    public void testCloneAllZero() throws Exception {
        writeText();
        Options options = new Options();
        options.networkProperties.hiddenLayer1Size = 10;
        options.generalProperties.inputFile = txtFilePath;
        IndexMaps maps = CoNLLReader.createIndices(options.generalProperties.inputFile, options.generalProperties.labeled,
                options.generalProperties.lowercase, "", 0, false);
        ArrayList<Integer> dependencyLabels = new ArrayList<>();
        for (int lab = 0; lab < maps.relSize(); lab++)
            dependencyLabels.add(lab);
        CoNLLReader reader = new CoNLLReader(options.generalProperties.inputFile);
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled,
                options.generalProperties.rootFirst, options.generalProperties.lowercase, maps);
        int wDim = 8;
        int pDim = 4;
        int lDim = 6;
        GreedyTrainer trainer = new GreedyTrainer(options, dependencyLabels,
                maps.labelNullIndex, maps.rareWords);
        List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
        maps.constructPreComputeMap(instances, 22, 10000);

        MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
        network.preCompute();
        MLPNetwork clonedNetwork = network.clone(true, false);

        assert clonedNetwork.numLayers() == network.numLayers();
        for (int i = 0; i < clonedNetwork.numLayers(); i++) {
            Layer layer2 = clonedNetwork.layer(i);
            assert Utils.allZero(layer2.getW());
            assert Utils.allZero(layer2.getB());

            if (layer2 instanceof FirstHiddenLayer) {
                WordEmbeddingLayer wordEmbeddingLayer2 = ((FirstHiddenLayer) layer2).getWordEmbeddings();
                Layer posEmbeddingLayer2 = ((FirstHiddenLayer) layer2).getPosEmbeddings();
                Layer depEmbeddingLayer2 = ((FirstHiddenLayer) layer2).getDepEmbeddings();

                assert Utils.allZero(wordEmbeddingLayer2.getW());
                assert Utils.allZero(posEmbeddingLayer2.getW());
                assert Utils.allZero(depEmbeddingLayer2.getW());
            }
        }
    }

    @Test
    public void testSaved() throws Exception {
        writeText();
        Options options = new Options();
        options.networkProperties.hiddenLayer1Size = 10;
        options.generalProperties.inputFile = txtFilePath;
        options.generalProperties.parserType = ParserType.ArcEager;
        IndexMaps maps = CoNLLReader.createIndices(options.generalProperties.inputFile, options.generalProperties.labeled,
                options.generalProperties.lowercase, "", 0, false);
        ArrayList<Integer> dependencyLabels = new ArrayList<>();
        for (int lab = 0; lab < maps.relSize(); lab++)
            dependencyLabels.add(lab);
        CoNLLReader reader = new CoNLLReader(options.generalProperties.inputFile);
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled,
                options.generalProperties.rootFirst, options.generalProperties.lowercase, maps);
        int wDim = 8;
        int pDim = 4;
        int lDim = 6;
        GreedyTrainer trainer = new GreedyTrainer(options, dependencyLabels,
                maps.labelNullIndex, maps.rareWords);
        List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
        maps.constructPreComputeMap(instances, 22, 10000);

        MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
        network.preCompute();
        MLPNetwork clonedNetwork = network.clone();
        clonedNetwork.emptyPrecomputedMap();

        double eps = 1e-15;
        for (NeuralTrainingInstance instance : instances) {
            double[] preComputedOutput = network.output(instance.getFeatures(), instance.getLabel());
            double[] regularOutput = clonedNetwork.output(instance.getFeatures(), instance.getLabel());

            for (int i = 0; i < regularOutput.length; i++) {
                double diff = Math.abs(regularOutput[i] - preComputedOutput[i]);
                if (diff > 0)
                    System.out.println(i + "\t" + regularOutput[i] + "\t" + preComputedOutput[i] + "\t" + diff);
                assert diff <= eps;
            }
        }
    }

    @Test
    public void testSavedWithTwoHiddenLayers() throws Exception {
        writeText();
        int[] h2sizes = new int[]{5, 10, 15};

        for (int h2size : h2sizes) {
            Options options = new Options();
            options.networkProperties.hiddenLayer1Size = 10;
            options.networkProperties.hiddenLayer2Size = h2size;
            options.generalProperties.inputFile = txtFilePath;
            options.generalProperties.parserType = ParserType.ArcEager;
            IndexMaps maps = CoNLLReader.createIndices(options.generalProperties.inputFile, options.generalProperties.labeled,
                    options.generalProperties.lowercase, "", 0, false);
            ArrayList<Integer> dependencyLabels = new ArrayList<>();
            for (int lab = 0; lab < maps.relSize(); lab++)
                dependencyLabels.add(lab);
            CoNLLReader reader = new CoNLLReader(options.generalProperties.inputFile);
            ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled,
                    options.generalProperties.rootFirst, options.generalProperties.lowercase, maps);
            int wDim = 8;
            int pDim = 4;
            int lDim = 6;
            GreedyTrainer trainer = new GreedyTrainer(options, dependencyLabels,
                    maps.labelNullIndex, maps.rareWords);
            List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
            maps.constructPreComputeMap(instances, 22, 10000);

            MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
            network.preCompute();
            MLPNetwork clonedNetwork = network.clone();
            clonedNetwork.emptyPrecomputedMap();

            double eps = 1e-15;
            for (NeuralTrainingInstance instance : instances) {
                double[] preComputedOutput = network.output(instance.getFeatures(), instance.getLabel());
                double[] regularOutput = clonedNetwork.output(instance.getFeatures(), instance.getLabel());

                for (int i = 0; i < regularOutput.length; i++) {
                    double diff = Math.abs(regularOutput[i] - preComputedOutput[i]);
                    if (diff > 0)
                        System.out.println(i + "\t" + regularOutput[i] + "\t" + preComputedOutput[i] + "\t" + diff);
                    assert diff <= eps;
                }
            }
        }
    }

    @Test
    public void testOutput() throws Exception {
        writeText();
        Options options = new Options();
        options.networkProperties.hiddenLayer1Size = 10;
        options.networkProperties.hiddenLayer2Size = 0;
        options.generalProperties.inputFile = txtFilePath;
        options.networkProperties.outputBiasTerm = true;
        options.generalProperties.parserType = ParserType.ArcEager;
        IndexMaps maps = CoNLLReader.createIndices(options.generalProperties.inputFile, options.generalProperties.labeled,
                options.generalProperties.lowercase, "", 0, false);
        ArrayList<Integer> dependencyLabels = new ArrayList<>();
        for (int lab = 0; lab < maps.relSize(); lab++)
            dependencyLabels.add(lab);
        CoNLLReader reader = new CoNLLReader(options.generalProperties.inputFile);
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled,
                options.generalProperties.rootFirst, options.generalProperties.lowercase, maps);
        int wDim = 4;
        int pDim = 4;
        int lDim = 4;

        GreedyTrainer trainer = new GreedyTrainer(options, dependencyLabels,
                maps.labelNullIndex, maps.rareWords);
        List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
        maps.constructPreComputeMap(instances, 22, 10000);

        MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);

        double[][] w = network.getWordEmbedding();
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[i].length; j++)
                w[i][j] = 1;
        }

        double[][] p = network.getPosEmbedding();
        for (int i = 0; i < p.length; i++) {
            for (int j = 0; j < p[i].length; j++)
                p[i][j] = 1;
        }

        double[][] l = network.getPosEmbedding();
        for (int i = 0; i < l.length; i++) {
            for (int j = 0; j < l[i].length; j++)
                l[i][j] = 1;
        }

        double[] hBias = network.layer(0).getB();
        for (int i = 0; i < hBias.length; i++)
            hBias[i] = 1;

        double[][] hW = network.layer(0).getW();
        for (int i = 0; i < hW.length; i++) {
            for (int j = 0; j < hW[i].length; j++)
                hW[i][j] = 1;
        }

        double[] sBias = network.layer(1).getB();
        for (int i = 0; i < sBias.length; i++)
            sBias[i] = 1;

        double[][] sW = network.layer(1).getW();
        for (int i = 0; i < sW.length; i++) {
            for (int j = 0; j < sW[i].length; j++)
                sW[i][j] = 1;
        }

        network.preCompute();

        for (NeuralTrainingInstance instance : instances) {
            double[] preComputedOutput = network.output(instance.getFeatures(), new double[network.getNumOutputs()], true);

            for (int i = 0; i < preComputedOutput.length; i++)
                assert preComputedOutput[i] == Math.log(1.0 / network.getNumOutputs());
        }
    }

    @Test
    public void testForward() throws Exception {
        writeText();
        Options options = new Options();
        options.networkProperties.hiddenLayer1Size = 10;
        options.generalProperties.inputFile = txtFilePath;
        options.networkProperties.outputBiasTerm = true;
        options.generalProperties.parserType = ParserType.ArcEager;
        IndexMaps maps = CoNLLReader.createIndices(options.generalProperties.inputFile, options.generalProperties.labeled,
                options.generalProperties.lowercase, "", 0, false);
        ArrayList<Integer> dependencyLabels = new ArrayList<>();
        for (int lab = 0; lab < maps.relSize(); lab++)
            dependencyLabels.add(lab);
        CoNLLReader reader = new CoNLLReader(options.generalProperties.inputFile);
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled,
                options.generalProperties.rootFirst, options.generalProperties.lowercase, maps);
        int wDim = 4;
        int pDim = 4;
        int lDim = 4;

        GreedyTrainer trainer = new GreedyTrainer(options, dependencyLabels,
                maps.labelNullIndex, maps.rareWords);
        List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);
        maps.constructPreComputeMap(instances, 22, 10000);

        MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);

        double[][] w = network.getWordEmbedding();
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[i].length; j++)
                w[i][j] = 3;
        }

        double[][] p = network.getPosEmbedding();
        for (int i = 0; i < p.length; i++) {
            for (int j = 0; j < p[i].length; j++)
                p[i][j] = 2;
        }

        double[][] l = network.getDepEmbedding();
        for (int i = 0; i < l.length; i++) {
            for (int j = 0; j < l[i].length; j++)
                l[i][j] = 1;
        }

        double[] hBias = network.layer(0).getB();
        for (int i = 0; i < hBias.length; i++)
            hBias[i] = 4;

        double[][] hW = network.layer(0).getW();
        for (int i = 0; i < hW.length; i++) {
            for (int j = 0; j < hW[i].length; j++)
                hW[i][j] = 1;
        }

        double[] sBias = network.layer(1).getB();
        for (int i = 0; i < sBias.length; i++)
            sBias[i] = 1;

        double[][] sW = network.layer(1).getW();
        for (int i = 0; i < sW.length; i++) {
            for (int j = 0; j < sW[i].length; j++)
                sW[i][j] = 1;
        }
        network.emptyPrecomputedMap();

        double[] f1 = null;
        for (NeuralTrainingInstance instance : instances) {
            double[] f = network.layer(0).forward(instance.getFeatures());
            for (int i = 0; i < f.length; i++) {
                assert f[i] == network.getNumWordLayers() * 3 * network.getwDim() + network.getNumPosLayers() * 2 * network.getpDim() +
                        network.getNumDepLayers() * 1 * network.getDepDim() + 4;
            }
        }

    }

    private void writeText() throws Exception {
        BufferedWriter writer = new BufferedWriter(new FileWriter(txtFilePath));
        writer.write(conllText);
        writer.close();
    }
}
