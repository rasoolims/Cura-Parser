package Tests;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 11:07 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

import YaraParser.Accessories.CoNLLReader;
import YaraParser.Accessories.Options;
import YaraParser.Accessories.Utils;
import YaraParser.Learning.Activation.ActivationType;
import YaraParser.Learning.NeuralNetwork.MLPNetwork;
import YaraParser.Learning.NeuralNetwork.MLPTrainer;
import YaraParser.Learning.NeuralNetwork.NetworkMatrices;
import YaraParser.Learning.Updater.UpdaterType;
import YaraParser.Structures.EmbeddingTypes;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.NeuralTrainingInstance;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Trainer.ArcEagerBeamTrainer;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;


public class GradientTest {

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
    public void TestWordEmbeddingGradients() throws Exception {
        for (ActivationType type : ActivationType.values()) {
            writeText();
            Options options = new Options();
            options.activationType = type;
            options.hiddenLayer1Size = 10;
            options.inputFile = txtFilePath;
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, "", 0);
            ArrayList<Integer> dependencyLabels = new ArrayList<>();
            for (int lab = 0; lab < maps.relSize(); lab++)
                dependencyLabels.add(lab);
            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled, options
                    .rootFirst, options.lowercase, maps);
            int wDim = 8;
            int pDim = 4;
            int lDim = 6;
            MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim);
            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early",
                    options, dependencyLabels, maps);
            List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);

            double[][][] savedGradients = new double[network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers()][][];
            for (int i = 0; i < network.getNumWordLayers(); i++)
                savedGradients[i] = new double[network.maps.preComputeMap.size()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers(); i < network.getNumWordLayers() + network.getNumPosLayers(); i++)
                savedGradients[i] = new double[network.getNumPos()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers() + network.getNumPosLayers();
                 i < network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers(); i++)
                savedGradients[i] = new double[network.getNumDepLabels()][network.getHiddenLayerDim()];

            MLPTrainer classifier = new MLPTrainer(network, UpdaterType.SGD, 0.9, 0.1, 1e-4, 1, 0);
            network.preCompute();

            NetworkMatrices gradients = new NetworkMatrices(network.getNumWords(), network.getWordEmbedDim(), network.getNumPos(), network
                    .getPosEmbeddingDim(), network.getNumDepLabels(), network.getLabelEmbedDim(), network.getHiddenLayerDim(), network
                    .getHiddenLayerIntDim(), network.getSoftmaxLayerDim());
            classifier.calculateCost(instances, 1, gradients, savedGradients);

            double eps = 0.000001;
            for (int i = 0; i < network.getNumWordLayers(); i++) {
                for (int slot = 0; slot < network.getWordEmbedDim(); slot++) {
                    int tok = instances.get(0).getFeatures()[i];
                    double gradForTok = gradients.getWordEmbedding()[tok][slot] / instances.size();

                    MLPNetwork plusNetwork = network.clone();
                    purturb(plusNetwork.getMatrices(), EmbeddingTypes.WORD, tok, slot, eps);
                    plusNetwork.preCompute();
                    MLPNetwork negNetwork = network.clone();
                    purturb(negNetwork.getMatrices(), EmbeddingTypes.WORD, tok, slot, -eps);
                    negNetwork.preCompute();
                    double diff = 0;

                    for (NeuralTrainingInstance instance : instances) {
                        double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel());
                        double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel());

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
    public void TestWordEmbeddingUpdates() throws Exception {
        for (ActivationType type : ActivationType.values()) {
            writeText();
            Options options = new Options();
            options.activationType = type;
            options.hiddenLayer1Size = 10;
            options.inputFile = txtFilePath;
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, "", 1);
            ArrayList<Integer> dependencyLabels = new ArrayList<>();
            for (int lab = 0; lab < maps.relSize(); lab++)
                dependencyLabels.add(lab);
            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled, options
                    .rootFirst, options.lowercase, maps);
            int wDim = 8;
            int pDim = 4;
            int lDim = 6;
            MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim);
            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early",
                    options, dependencyLabels, maps);
            List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);

            double[][][] savedGradients = new double[network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers()][][];
            for (int i = 0; i < network.getNumWordLayers(); i++)
                savedGradients[i] = new double[network.maps.preComputeMap.size()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers(); i < network.getNumWordLayers() + network.getNumPosLayers(); i++)
                savedGradients[i] = new double[network.getNumPos()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers() + network.getNumPosLayers();
                 i < network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers(); i++)
                savedGradients[i] = new double[network.getNumDepLabels()][network.getHiddenLayerDim()];

            MLPTrainer classifier = new MLPTrainer(network, UpdaterType.SGD, 0.9, 0.1, 1e-4, 1, 0);
            network.preCompute();

            NetworkMatrices gradients = new NetworkMatrices(network.getNumWords(), network.getWordEmbedDim(), network.getNumPos(), network
                    .getPosEmbeddingDim(), network.getNumDepLabels(), network.getLabelEmbedDim(), network.getHiddenLayerDim(), network
                    .getHiddenLayerIntDim(), network.getSoftmaxLayerDim());
            classifier.calculateCost(instances, 1, gradients, savedGradients);

            double[] oovEmbedding = Utils.clone(network.getMatrices().getWordEmbedding()[0]);
            double[] nullEmbedding = Utils.clone(network.getMatrices().getWordEmbedding()[1]);
            double[] rootEmbedding = Utils.clone(network.getMatrices().getWordEmbedding()[2]);
            double[] simpleWordEmbedding = Utils.clone(network.getMatrices().getWordEmbedding()[3]);
            for (int i = 0; i < 3; i++) {
                classifier.fit(instances, i, true);
            }

            assert !Utils.equals(network.getMatrices().getWordEmbedding()[0], oovEmbedding);
            assert !Utils.equals(network.getMatrices().getWordEmbedding()[1], nullEmbedding);
            assert !Utils.equals(network.getMatrices().getWordEmbedding()[2], rootEmbedding);
            assert !Utils.equals(network.getMatrices().getWordEmbedding()[3], simpleWordEmbedding);
        }
    }


    @Test
    public void TestPOSEmbeddingGradients() throws Exception {
        for (ActivationType type : ActivationType.values()) {
            writeText();
            Options options = new Options();
            options.activationType = type;
            options.hiddenLayer1Size = 10;
            options.inputFile = txtFilePath;
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, "", 0);
            ArrayList<Integer> dependencyLabels = new ArrayList<>();
            for (int lab = 0; lab < maps.relSize(); lab++)
                dependencyLabels.add(lab);
            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled, options
                    .rootFirst, options.lowercase, maps);
            int wDim = 8;
            int pDim = 4;
            int lDim = 6;
            MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim);
            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early",
                    options, dependencyLabels, maps);
            List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);

            double[][][] savedGradients = new double[network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers()][][];
            for (int i = 0; i < network.getNumWordLayers(); i++)
                savedGradients[i] = new double[network.maps.preComputeMap.size()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers(); i < network.getNumWordLayers() + network.getNumPosLayers(); i++)
                savedGradients[i] = new double[network.getNumPos()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers() + network.getNumPosLayers();
                 i < network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers(); i++)
                savedGradients[i] = new double[network.getNumDepLabels()][network.getHiddenLayerDim()];

            MLPTrainer classifier = new MLPTrainer(network, UpdaterType.SGD, 0.9, 0.1, 1e-4, 1, 0);
            network.preCompute();

            NetworkMatrices gradients = new NetworkMatrices(network.getNumWords(), network.getWordEmbedDim(), network.getNumPos(), network
                    .getPosEmbeddingDim(), network.getNumDepLabels(), network.getLabelEmbedDim(), network.getHiddenLayerDim(), network
                    .getHiddenLayerIntDim(), network.getSoftmaxLayerDim());
            classifier.calculateCost(instances, 1, gradients, savedGradients);

            double eps = 0.000001;
            for (int i = network.getNumWordLayers(); i < network.getNumWordLayers() + network.getNumPosLayers(); i++) {
                for (int slot = 0; slot < network.getPosEmbeddingDim(); slot++) {
                    int tok = instances.get(0).getFeatures()[i];
                    double gradForTok = gradients.getPosEmbedding()[tok][slot] / instances.size();

                    MLPNetwork plusNetwork = network.clone();
                    purturb(plusNetwork.getMatrices(), EmbeddingTypes.POS, tok, slot, eps);
                    plusNetwork.preCompute();
                    MLPNetwork negNetwork = network.clone();
                    purturb(negNetwork.getMatrices(), EmbeddingTypes.POS, tok, slot, -eps);
                    negNetwork.preCompute();
                    double diff = 0;

                    for (NeuralTrainingInstance instance : instances) {
                        double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel());
                        double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel());

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
    public void TestLabelEmbeddingGradients() throws Exception {
        for (ActivationType type : ActivationType.values()) {
            writeText();
            Options options = new Options();
            options.activationType = type;
            options.hiddenLayer1Size = 10;
            options.inputFile = txtFilePath;
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, "", 0);
            ArrayList<Integer> dependencyLabels = new ArrayList<>();
            for (int lab = 0; lab < maps.relSize(); lab++)
                dependencyLabels.add(lab);
            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled, options
                    .rootFirst, options.lowercase, maps);
            int wDim = 8;
            int pDim = 4;
            int lDim = 6;
            MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim);
            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early",
                    options, dependencyLabels, maps);
            List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);

            double[][][] savedGradients = new double[network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers()][][];
            for (int i = 0; i < network.getNumWordLayers(); i++)
                savedGradients[i] = new double[network.maps.preComputeMap.size()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers(); i < network.getNumWordLayers() + network.getNumPosLayers(); i++)
                savedGradients[i] = new double[network.getNumPos()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers() + network.getNumPosLayers();
                 i < network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers(); i++)
                savedGradients[i] = new double[network.getNumDepLabels()][network.getHiddenLayerDim()];

            MLPTrainer classifier = new MLPTrainer(network, UpdaterType.SGD, 0.9, 0.1, 1e-4, 1, 0);
            network.preCompute();

            NetworkMatrices gradients = new NetworkMatrices(network.getNumWords(), network.getWordEmbedDim(), network.getNumPos(), network
                    .getPosEmbeddingDim(), network.getNumDepLabels(), network.getLabelEmbedDim(), network.getHiddenLayerDim(), network
                    .getHiddenLayerIntDim(), network.getSoftmaxLayerDim());
            classifier.calculateCost(instances, 1, gradients, savedGradients);

            double eps = 0.000001;
            for (int i = network.getNumWordLayers() + network.getNumPosLayers(); i < network.getNumWordLayers() + network.getNumPosLayers() + network
                    .getNumDepLayers(); i++) {
                for (int slot = 0; slot < network.getLabelEmbedDim(); slot++) {
                    int tok = instances.get(0).getFeatures()[i];
                    double gradForTok = gradients.getLabelEmbedding()[tok][slot] / instances.size();

                    MLPNetwork plusNetwork = network.clone();
                    purturb(plusNetwork.getMatrices(), EmbeddingTypes.DEPENDENCY, tok, slot, eps);
                    plusNetwork.preCompute();
                    MLPNetwork negNetwork = network.clone();
                    purturb(negNetwork.getMatrices(), EmbeddingTypes.DEPENDENCY, tok, slot, -eps);
                    negNetwork.preCompute();
                    double diff = 0;

                    for (NeuralTrainingInstance instance : instances) {
                        double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel());
                        double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel());

                        int goldLabel = instance.gold();
                        diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                    }
                    diff /= (2 * eps * instances.size());

                    System.out.println(plusNetwork.activationType + "\t" + gradForTok + "\t" + diff);
                    assert Math.abs(gradForTok - diff) <= 0.0000001;
                }
            }
        }
    }

    @Test
    public void TestHiddenLayerGradients() throws Exception {
        for (ActivationType type : ActivationType.values()) {
            writeText();
            Options options = new Options();
            options.activationType = type;
            options.inputFile = txtFilePath;
            options.hiddenLayer1Size = 10;
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, "", 0);
            ArrayList<Integer> dependencyLabels = new ArrayList<>();
            for (int lab = 0; lab < maps.relSize(); lab++)
                dependencyLabels.add(lab);
            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled, options
                    .rootFirst, options.lowercase, maps);
            int wDim = 8;
            int pDim = 4;
            int lDim = 6;
            MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim);
            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early",
                    options, dependencyLabels, maps);
            List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);

            double[][][] savedGradients = new double[network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers()][][];
            for (int i = 0; i < network.getNumWordLayers(); i++)
                savedGradients[i] = new double[network.maps.preComputeMap.size()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers(); i < network.getNumWordLayers() + network.getNumPosLayers(); i++)
                savedGradients[i] = new double[network.getNumPos()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers() + network.getNumPosLayers();
                 i < network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers(); i++)
                savedGradients[i] = new double[network.getNumDepLabels()][network.getHiddenLayerDim()];

            MLPTrainer classifier = new MLPTrainer(network, UpdaterType.SGD, 0.9, 0.1, 1e-4, 1, 0);
            network.preCompute();

            NetworkMatrices gradients = new NetworkMatrices(network.getNumWords(), network.getWordEmbedDim(), network.getNumPos(), network
                    .getPosEmbeddingDim(), network.getNumDepLabels(), network.getLabelEmbedDim(), network.getHiddenLayerDim(), network
                    .getHiddenLayerIntDim(), network.getSoftmaxLayerDim());
            classifier.calculateCost(instances, 1, gradients, savedGradients);

            double eps = 0.000001;
            for (int i = 0; i < network.getHiddenLayerDim(); i++) {
                for (int slot = 0; slot < network.getHiddenLayerIntDim(); slot++) {
                    int tok = i;
                    double gradForTok = gradients.getHiddenLayer()[tok][slot] / instances.size();

                    MLPNetwork plusNetwork = network.clone();
                    purturb(plusNetwork.getMatrices(), EmbeddingTypes.HIDDENLAYER, tok, slot, eps);
                    plusNetwork.preCompute();
                    MLPNetwork negNetwork = network.clone();
                    purturb(negNetwork.getMatrices(), EmbeddingTypes.HIDDENLAYER, tok, slot, -eps);
                    negNetwork.preCompute();
                    double diff = 0;

                    for (NeuralTrainingInstance instance : instances) {
                        double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel());
                        double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel());

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
    public void TestHiddenLayerBiasGradients() throws Exception {
        for (ActivationType type : ActivationType.values()) {
            writeText();
            Options options = new Options();
            options.activationType = type;
            options.inputFile = txtFilePath;
            options.hiddenLayer1Size = 10;
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, "", 0);
            ArrayList<Integer> dependencyLabels = new ArrayList<>();
            for (int lab = 0; lab < maps.relSize(); lab++)
                dependencyLabels.add(lab);
            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled, options
                    .rootFirst, options.lowercase, maps);
            int wDim = 8;
            int pDim = 4;
            int lDim = 6;
            MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim);
            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early",
                    options, dependencyLabels, maps);
            List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);

            double[][][] savedGradients = new double[network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers()][][];
            for (int i = 0; i < network.getNumWordLayers(); i++)
                savedGradients[i] = new double[network.maps.preComputeMap.size()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers(); i < network.getNumWordLayers() + network.getNumPosLayers(); i++)
                savedGradients[i] = new double[network.getNumPos()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers() + network.getNumPosLayers();
                 i < network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers(); i++)
                savedGradients[i] = new double[network.getNumDepLabels()][network.getHiddenLayerDim()];

            MLPTrainer classifier = new MLPTrainer(network, UpdaterType.SGD, 0.9, 0.1, 1e-4, 1, 0);
            network.preCompute();

            NetworkMatrices gradients = new NetworkMatrices(network.getNumWords(), network.getWordEmbedDim(), network.getNumPos(), network
                    .getPosEmbeddingDim(), network.getNumDepLabels(), network.getLabelEmbedDim(), network.getHiddenLayerDim(), network
                    .getHiddenLayerIntDim(), network.getSoftmaxLayerDim());
            classifier.calculateCost(instances, 1, gradients, savedGradients);

            double eps = 0.000001;
            for (int i = 0; i < network.getHiddenLayerDim(); i++) {
                int tok = i;
                double gradForTok = gradients.getHiddenLayerBias()[tok] / instances.size();

                MLPNetwork plusNetwork = network.clone();
                purturb(plusNetwork.getMatrices(), EmbeddingTypes.HIDDENLAYERBIAS, tok, -1, eps);
                plusNetwork.preCompute();
                MLPNetwork negNetwork = network.clone();
                purturb(negNetwork.getMatrices(), EmbeddingTypes.HIDDENLAYERBIAS, tok, -1, -eps);
                negNetwork.preCompute();
                double diff = 0;

                for (NeuralTrainingInstance instance : instances) {
                    double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel());
                    double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel());

                    int goldLabel = instance.gold();
                    diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                }
                diff /= (2 * eps * instances.size());

                System.out.println(gradForTok + "\t" + diff);
                assert Math.abs(gradForTok - diff) <= 0.0000001;
            }
        }
    }

    @Test
    public void TestSoftmaxLayerGradients() throws Exception {
        for (ActivationType type : ActivationType.values()) {
            writeText();
            Options options = new Options();
            options.activationType = type;
            options.inputFile = txtFilePath;
            options.hiddenLayer1Size = 10;
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, "", 0);
            ArrayList<Integer> dependencyLabels = new ArrayList<>();
            for (int lab = 0; lab < maps.relSize(); lab++)
                dependencyLabels.add(lab);
            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled, options
                    .rootFirst, options.lowercase, maps);
            int wDim = 8;
            int pDim = 4;
            int lDim = 6;
            MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim);
            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early",
                    options, dependencyLabels, maps);
            List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);

            double[][][] savedGradients = new double[network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers()][][];
            for (int i = 0; i < network.getNumWordLayers(); i++)
                savedGradients[i] = new double[network.maps.preComputeMap.size()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers(); i < network.getNumWordLayers() + network.getNumPosLayers(); i++)
                savedGradients[i] = new double[network.getNumPos()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers() + network.getNumPosLayers();
                 i < network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers(); i++)
                savedGradients[i] = new double[network.getNumDepLabels()][network.getHiddenLayerDim()];

            MLPTrainer classifier = new MLPTrainer(network, UpdaterType.SGD, 0.9, 0.1, 1e-4, 1, 0);
            network.preCompute();

            NetworkMatrices gradients = new NetworkMatrices(network.getNumWords(), network.getWordEmbedDim(), network.getNumPos(), network
                    .getPosEmbeddingDim(), network.getNumDepLabels(), network.getLabelEmbedDim(), network.getHiddenLayerDim(), network
                    .getHiddenLayerIntDim(), network.getSoftmaxLayerDim());
            classifier.calculateCost(instances, 1, gradients, savedGradients);

            double eps = 0.000001;
            for (int i = 0; i < network.getSoftmaxLayerDim(); i++) {
                for (int slot = 0; slot < network.getHiddenLayerDim(); slot++) {
                    int tok = i;
                    double gradForTok = gradients.getSoftmaxLayer()[tok][slot] / instances.size();

                    MLPNetwork plusNetwork = network.clone();
                    purturb(plusNetwork.getMatrices(), EmbeddingTypes.SOFTMAX, tok, slot, eps);
                    plusNetwork.preCompute();
                    MLPNetwork negNetwork = network.clone();
                    purturb(negNetwork.getMatrices(), EmbeddingTypes.SOFTMAX, tok, slot, -eps);
                    negNetwork.preCompute();
                    double diff = 0;

                    for (NeuralTrainingInstance instance : instances) {
                        double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel());
                        double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel());

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
    public void TestSoftmaxLayerBiasGradients() throws Exception {
        for (ActivationType type : ActivationType.values()) {
            writeText();
            Options options = new Options();
            options.activationType = type;
            options.inputFile = txtFilePath;
            options.hiddenLayer1Size = 10;
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, "", 0);
            ArrayList<Integer> dependencyLabels = new ArrayList<>();
            for (int lab = 0; lab < maps.relSize(); lab++)
                dependencyLabels.add(lab);
            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled, options
                    .rootFirst, options.lowercase, maps);
            int wDim = 8;
            int pDim = 4;
            int lDim = 6;
            MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim);
            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early",
                    options, dependencyLabels, maps);
            List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);

            double[][][] savedGradients = new double[network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers()][][];
            for (int i = 0; i < network.getNumWordLayers(); i++)
                savedGradients[i] = new double[network.maps.preComputeMap.size()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers(); i < network.getNumWordLayers() + network.getNumPosLayers(); i++)
                savedGradients[i] = new double[network.getNumPos()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers() + network.getNumPosLayers();
                 i < network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers(); i++)
                savedGradients[i] = new double[network.getNumDepLabels()][network.getHiddenLayerDim()];

            MLPTrainer classifier = new MLPTrainer(network, UpdaterType.SGD, 0.9, 0.1, 1e-4, 1, 0);
            network.preCompute();

            NetworkMatrices gradients = new NetworkMatrices(network.getNumWords(), network.getWordEmbedDim(), network.getNumPos(), network
                    .getPosEmbeddingDim(), network.getNumDepLabels(), network.getLabelEmbedDim(), network.getHiddenLayerDim(), network
                    .getHiddenLayerIntDim(), network.getSoftmaxLayerDim());
            classifier.calculateCost(instances, 1, gradients, savedGradients);

            double eps = 0.000001;
            for (int i = 0; i < network.getSoftmaxLayerDim(); i++) {
                int tok = i;
                double gradForTok = gradients.getSoftmaxLayerBias()[tok] / instances.size();

                MLPNetwork plusNetwork = network.clone();
                purturb(plusNetwork.getMatrices(), EmbeddingTypes.SOFTMAXBIAS, tok, -1, eps);
                plusNetwork.preCompute();
                MLPNetwork negNetwork = network.clone();
                purturb(negNetwork.getMatrices(), EmbeddingTypes.SOFTMAXBIAS, tok, -1, -eps);
                negNetwork.preCompute();
                double diff = 0;

                for (NeuralTrainingInstance instance : instances) {
                    double[] plusPurturb = plusNetwork.output(instance.getFeatures(), instance.getLabel());
                    double[] negPurturb = negNetwork.output(instance.getFeatures(), instance.getLabel());

                    int goldLabel = instance.gold();
                    diff += -(plusPurturb[goldLabel] - negPurturb[goldLabel]);
                }
                diff /= (2 * eps * instances.size());

                System.out.println(gradForTok + "\t" + diff);
                assert Math.abs(gradForTok - diff) <= 0.0000001;
            }
        }
    }

    @Test
    public void TestMultiThreadGradients() throws Exception {
        for (ActivationType type : ActivationType.values()) {
            writeText();
            Options options = new Options();
            options.activationType = type;
            options.inputFile = txtFilePath;
            options.hiddenLayer1Size = 10;
            IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase, "", 0);
            ArrayList<Integer> dependencyLabels = new ArrayList<>();
            for (int lab = 0; lab < maps.relSize(); lab++)
                dependencyLabels.add(lab);
            CoNLLReader reader = new CoNLLReader(options.inputFile);
            ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.labeled, options
                    .rootFirst, options.lowercase, maps);
            int wDim = 8;
            int pDim = 4;
            int lDim = 6;
            MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim);
            ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early",
                    options, dependencyLabels, maps);
            List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);

            double[][][] savedGradients = new double[network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers()][][];
            for (int i = 0; i < network.getNumWordLayers(); i++)
                savedGradients[i] = new double[network.maps.preComputeMap.size()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers(); i < network.getNumWordLayers() + network.getNumPosLayers(); i++)
                savedGradients[i] = new double[network.getNumPos()][network.getHiddenLayerDim()];
            for (int i = network.getNumWordLayers() + network.getNumPosLayers();
                 i < network.getNumWordLayers() + network.getNumPosLayers() + network.getNumDepLayers(); i++)
                savedGradients[i] = new double[network.getNumDepLabels()][network.getHiddenLayerDim()];

            MLPTrainer classifier = new MLPTrainer(network, UpdaterType.SGD, 0.9, 0.1, 1e-4, 1, 0);
            MLPTrainer classifierMultiThread = new MLPTrainer(network, UpdaterType.SGD, 0.9, 0.1, 1e-4, 4, 0);
            network.preCompute();

            NetworkMatrices gradients = new NetworkMatrices(network.getNumWords(), network.getWordEmbedDim(), network.getNumPos(), network
                    .getPosEmbeddingDim(), network.getNumDepLabels(), network.getLabelEmbedDim(), network.getHiddenLayerDim(), network
                    .getHiddenLayerIntDim(), network.getSoftmaxLayerDim());

            classifier.calculateCost(instances, instances.size(), gradients, savedGradients);
            classifierMultiThread.cost(instances);

            final NetworkMatrices gradientsMultiThread = classifierMultiThread.getGradients();
            double eps = 1e-15;

            ArrayList<double[][]> allMatrices1 = gradients.getAllMatrices();
            ArrayList<double[][]> allMatrices2 = gradientsMultiThread.getAllMatrices();
            ArrayList<double[]> allVectors1 = gradients.getAllVectors();
            ArrayList<double[]> allVectors2 = gradientsMultiThread.getAllVectors();

            for (int i = 0; i < allMatrices1.size(); i++) {
                double[][] m1 = allMatrices1.get(i);
                double[][] m2 = allMatrices2.get(i);

                for (int j = 0; j < m1.length; j++)
                    for (int h = 0; h < m1[j].length; h++) {
                        if (Math.abs(m1[j][h] - m2[j][h]) > eps)
                            System.out.println(i + "\t" + j + "\t" + h + "\t" + m1[j][h] + "\t" + m2[j][h]);
                        assert Math.abs(m1[j][h] - m2[j][h]) <= eps;
                    }
            }

            for (int i = 0; i < allVectors1.size(); i++) {
                double[] v1 = allVectors1.get(i);
                double[] v2 = allVectors2.get(i);

                for (int j = 0; j < v1.length; j++) {
                    if (Math.abs(v1[j] - v2[j]) > eps)
                        System.out.println(i + "\t" + j + "\t" + v1[j] + "\t" + v2[j]);
                    assert Math.abs(v1[j] - v2[j]) <= eps;
                }
            }
        }
    }


    private void purturb(NetworkMatrices matrices, EmbeddingTypes type, int tokNum, int slotNum, double eps) {
        if (type == EmbeddingTypes.WORD) {
            matrices.getWordEmbedding()[tokNum][slotNum] += eps;
        } else if (type == EmbeddingTypes.POS) {
            matrices.getPosEmbedding()[tokNum][slotNum] += eps;
        } else if (type == EmbeddingTypes.DEPENDENCY) {
            matrices.getLabelEmbedding()[tokNum][slotNum] += eps;
        } else if (type == EmbeddingTypes.HIDDENLAYER) {
            matrices.getHiddenLayer()[tokNum][slotNum] += eps;
        } else if (type == EmbeddingTypes.SOFTMAX) {
            matrices.getSoftmaxLayer()[tokNum][slotNum] += eps;
        } else if (type == EmbeddingTypes.HIDDENLAYERBIAS) {
            matrices.getHiddenLayerBias()[tokNum] += eps;
        } else if (type == EmbeddingTypes.SOFTMAXBIAS) {
            matrices.getSoftmaxLayerBias()[tokNum] += eps;
        }
    }

    private void writeText() throws Exception {
        BufferedWriter writer = new BufferedWriter(new FileWriter(txtFilePath));
        writer.write(conllText);
        writer.close();
    }
}
