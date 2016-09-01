package edu.columbia.cs.nlp.Tests;

import edu.columbia.cs.nlp.CuraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.CuraParser.Accessories.Evaluator;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Learning.Activation.Enums.ActivationType;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPTrainer;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.UpdaterType;
import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.Structures.NeuralTrainingInstance;
import edu.columbia.cs.nlp.CuraParser.Structures.Pair;
import edu.columbia.cs.nlp.CuraParser.Structures.Sentence;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Beam.BeamParser;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.ParserType;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcEager;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcStandard;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Trainer.GreedyTrainer;
import org.junit.Test;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/10/16
 * Time: 7:37 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class ParserTest {
    final String shortConllText = "1\tTerms\t_\tNOUN\t_\t_\t4\tnsubjpass\t_\t_\n" +
            "2\twere\t_\tVERB\t_\t_\t4\tauxpass\t_\t_\n" +
            "3\tn't\t_\tADV\t_\t_\t2\tneg\t_\t_\n" +
            "4\tdisclosed\t_\tVERB\t_\t_\t0\tROOT\t_\t_\n" +
            "5\t.\t_\t.\t_\t_\t4\tp\t_\t_";
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
    final String tmpPath = "/tmp/f.txt";

    private void writeConllFile(String txt) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(tmpPath));
        writer.write(txt);
        writer.close();
    }

    @Test
    public void testArcEagerActions() throws Exception {
        writeConllFile(shortConllText);

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1, false);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, false, false, maps);
        Sentence sentence = dataSet.get(0).getSentence();
        Configuration configuration = new Configuration(sentence, options.generalProperties.rootFirst);
        ShiftReduceParser parser = new ArcEager();

        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.rightArc(configuration.state, maps.dep2Int("neg"));
        parser.reduce(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("auxpass"));
        parser.leftArc(configuration.state, maps.dep2Int("nsubjpass"));
        parser.shift(configuration.state);
        parser.rightArc(configuration.state, maps.dep2Int("p"));
        parser.reduce(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("ROOT"));
        assert configuration.state.isTerminalState();
    }

    @Test
    public void testArcStandardActions() throws Exception {
        writeConllFile(shortConllText);

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1, false);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        options.generalProperties.rootFirst = false;
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, options.generalProperties.rootFirst, false, maps);
        Sentence sentence = dataSet.get(0).getSentence();
        Configuration configuration = new Configuration(sentence, options.generalProperties.rootFirst);
        ShiftReduceParser parser = new ArcStandard();

        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.rightArc(configuration.state, maps.dep2Int("neg"));
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("auxpass"));
        parser.leftArc(configuration.state, maps.dep2Int("nsubjpass"));
        parser.shift(configuration.state);
        parser.rightArc(configuration.state, maps.dep2Int("p"));
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("ROOT"));
        assert configuration.state.isTerminalState();


        reader = new CoNLLReader(tmpPath);
        options.generalProperties.rootFirst = true;
        dataSet = reader.readData(Integer.MAX_VALUE, false, true, options.generalProperties.rootFirst, false, maps);
        sentence = dataSet.get(0).getSentence();
        configuration = new Configuration(sentence, options.generalProperties.rootFirst);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.rightArc(configuration.state, maps.dep2Int("neg"));
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("auxpass"));
        parser.leftArc(configuration.state, maps.dep2Int("nsubjpass"));
        parser.shift(configuration.state);
        parser.rightArc(configuration.state, maps.dep2Int("p"));
        parser.rightArc(configuration.state, maps.dep2Int("ROOT"));
        assert configuration.state.isTerminalState();
    }

    @Test
    public void TestPretrainedtWordEmbeddingUpdates() throws Exception {
        for (UpdaterType updaterType : UpdaterType.values()) {
            for (ActivationType type : ActivationType.values()) {
                if (type == ActivationType.RandomRelu)
                    continue; // because of randomness we cannot make sure what really had happened.
                writeText();
                writeWordEmbedText();
                Options options = new Options();
                options.updaterProperties.updaterType = updaterType;
                options.trainingOptions.wordEmbeddingFile = embedFilePath;
                options.trainingOptions.devPath = txtFilePath;
                options.networkProperties.activationType = type;
                options.networkProperties.hiddenLayer1Size = 10;
                options.generalProperties.inputFile = txtFilePath;
                options.generalProperties.modelFile = txtFilePath + ".model";
                options.generalProperties.parserType = ParserType.ArcEager;
                options.generalProperties.parserType = ParserType.ArcEager;
                IndexMaps maps = CoNLLReader.createIndices(options.generalProperties.inputFile, options.generalProperties.labeled,
                        options.generalProperties.lowercase, "", 1, false);
                ArrayList<Integer> dependencyLabels = new ArrayList<>();
                for (int lab = 0; lab < maps.relSize(); lab++)
                    dependencyLabels.add(lab);
                CoNLLReader reader = new CoNLLReader(options.generalProperties.inputFile);
                ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled,
                        options.generalProperties.rootFirst, options.generalProperties.lowercase, maps);
                int wDim = 8;
                int pDim = 4;
                int lDim = 6;
                GreedyTrainer trainer = new GreedyTrainer(
                        options, dependencyLabels, maps.labelNullIndex, maps.rareWords);
                List<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, dataSet.size(), 0);
                maps.constructPreComputeMap(instances, 22, 10000);
                MLPNetwork network = new MLPNetwork(maps, options, dependencyLabels, wDim, pDim, lDim, ParserType.ArcEager);

                MLPTrainer classifier = new MLPTrainer(network, options);
                network.preCompute();

                for (int i = 1; i <= 100; i++) {
                    double acc = classifier.fit(instances, i, true);

                    if (i % 10 == 0) {
                        BeamParser parser = new BeamParser(network, options.generalProperties.numOfThreads, ParserType.ArcEager);
                        parser.parseConll(options.trainingOptions.devPath, options.generalProperties.modelFile + ".tmp",
                                options.generalProperties.rootFirst, options.generalProperties.beamWidth, options.generalProperties.lowercase,
                                options.generalProperties.numOfThreads, false, "");
                        Pair<Double, Double> evaluator = Evaluator.evaluate(options.trainingOptions.devPath, options.generalProperties.modelFile +
                                        ".tmp",
                                options.generalProperties.punctuations);

                        FileOutputStream fos = new FileOutputStream(options.generalProperties.modelFile);
                        GZIPOutputStream gz = new GZIPOutputStream(fos);
                        ObjectOutput writer = new ObjectOutputStream(gz);
                        writer.writeObject(network);
                        writer.writeObject(options);
                        writer.close();
                        System.out.print("done!\n\n");

                        FileInputStream fis = new FileInputStream(options.generalProperties.modelFile);
                        GZIPInputStream gz2 = new GZIPInputStream(fis);
                        ObjectInput r = new ObjectInputStream(gz2);
                        MLPNetwork mlpNetwork = (MLPNetwork) r.readObject();
                        Options infoptions = (Options) r.readObject();
                        BeamParser loadedParser = new BeamParser(mlpNetwork, options.generalProperties.numOfThreads, ParserType.ArcEager);
                        loadedParser.parseConll(options.trainingOptions.devPath, options.generalProperties.modelFile + ".tmp2",
                                infoptions.generalProperties.rootFirst, options.generalProperties.beamWidth,
                                infoptions.generalProperties.lowercase, options.generalProperties.numOfThreads, false, options.scorePath);
                        Pair<Double, Double> evaluator2 = Evaluator.evaluate(options.trainingOptions.devPath,
                                options.generalProperties.modelFile + ".tmp2", options.generalProperties.punctuations);

                        assert evaluator.equals(evaluator2);
                        if (acc == 1) assert evaluator.first == 100 && evaluator.second == 100;
                    }
                }
            }
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
