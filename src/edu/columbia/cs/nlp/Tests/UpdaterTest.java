package edu.columbia.cs.nlp.Tests;

import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/11/16
 * Time: 6:27 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class UpdaterTest {
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
    public void testSGDUpdate() throws Exception {
     /*
        writeText();
        writeWordEmbedText();
        Options options = new Options();
        options.trainingOptions.wordEmbeddingFile = embedFilePath;
        options.trainingOptions.devPath = txtFilePath;
        options.networkProperties.hiddenLayer1Size = 10;
        options.trainingOptions.trainFile = txtFilePath;
        options.generalProperties.modelFile = txtFilePath + ".model";
        options.updaterProperties.learningRate = .1;
        IndexMaps maps = CoNLLReader.createIndices(options.trainingOptions.trainFile, options.generalProperties.labeled, options.generalProperties
                .lowercase, "", 1);
        ArrayList<Integer> dependencyLabels = new ArrayList<>();
        for (int lab = 0; lab < maps.relSize(); lab++)
            dependencyLabels.add(lab);
        CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options
                .generalProperties.rootFirst, options.generalProperties.lowercase, maps);
        int wDim = 8;
        int pDim = 4;
        int lDim = 6;
        BeamTrainer trainer = new BeamTrainer(
                options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
        List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, dataSet.size(), 0);
        maps.constructPreComputeMap(instances, 22, 10000);
        MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);
        network.getMatrices().getWordEmbedding()[0][0] = 0;
        double origValue = network.getMatrices().getWordEmbedding()[0][0];
        double g1 = -0.01;
        double g2 = options.updaterProperties.momentum * g1 + 0.005;
        double g3 = options.updaterProperties.momentum * g2 - 0.003;

        NetworkMatrices gradients = new NetworkMatrices(network.getNumWords(), network.getWordEmbedDim(), network.getNumPos(), network
                .getPosEmbedDim(), network.getNumDepLabels(), network.getDepEmbedDim(), network.getHiddenLayerDim(), network
                .getHiddenLayerIntDim(), network.getSecondHiddenLayerDim(), network.getSoftmaxLayerDim());
        // Test sgd with momentum.
        Updater updater = new SGD(network, options.updaterProperties.learningRate, options.networkProperties.outputBiasTerm,
                options.updaterProperties.momentum, SGDType.MOMENTUM);
        gradients.modify(EmbeddingTypes.WORD, 0, 0, 0.01);
        updater.update(gradients);

        // reset
        gradients = new NetworkMatrices(network.getNumWords(), network.getWordEmbedDim(), network.getNumPos(), network
                .getPosEmbedDim(), network.getNumDepLabels(), network.getDepEmbedDim(), network.getHiddenLayerDim(), network
                .getHiddenLayerIntDim(), network.getSecondHiddenLayerDim(), network.getSoftmaxLayerDim());

        gradients.modify(EmbeddingTypes.WORD, 0, 0, -0.005);
        updater.update(gradients);

        // reset
        gradients = new NetworkMatrices(network.getNumWords(), network.getWordEmbedDim(), network.getNumPos(), network
                .getPosEmbedDim(), network.getNumDepLabels(), network.getDepEmbedDim(), network.getHiddenLayerDim(), network
                .getHiddenLayerIntDim(), network.getSecondHiddenLayerDim(), network.getSoftmaxLayerDim());

        gradients.modify(EmbeddingTypes.WORD, 0, 0, 0.003);
        updater.update(gradients);

        assert updater.getGradientHistory().getWordEmbedding()[0][0] == g3;
        assert network.getMatrices().getWordEmbedding()[0][0] - origValue + options.updaterProperties.learningRate * (g1 + g2 + g3) < 1e-16;
   */
    }

    @Test
    public void testAdamUpdate() throws Exception {
        //todo
    }

    @Test
    public void testAdagradUpdate() throws Exception {
        //todo
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
