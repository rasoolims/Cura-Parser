package Tests;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 2:59 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

import YaraParser.Accessories.CoNLLReader;
import YaraParser.Accessories.Options;
import YaraParser.Learning.AveragedPerceptron;
import YaraParser.Learning.MLPNetwork;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.NeuralTrainingInstance;
import YaraParser.Structures.Sentence;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;
import YaraParser.TransitionBasedSystem.Parser.ArcEager;
import YaraParser.TransitionBasedSystem.Trainer.ArcEagerBeamTrainer;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class FeatureExtractorTest {
    final String conllText = "1\tThe\t_\tDET\t_\t_\t4\tdet\t_\t_\n" +
            "2\tcomplex\t_\tADJ\t_\t_\t4\tamod\t_\t_\n" +
            "3\tfinancing\t_\tNOUN\t_\t_\t4\tcompmod\t_\t_\n" +
            "4\tplan\t_\tNOUN\t_\t_\t10\tnsubj\t_\t_\n" +
            "5\tin\t_\tADP\t_\t_\t4\tadpmod\t_\t_\n" +
            "6\tthe\t_\tDET\t_\t_\t9\tdet\t_\t_\n" +
            "7\tS&L\t_\tNOUN\t_\t_\t9\tcompmod\t_\t_\n" +
            "8\tbailout\t_\tNOUN\t_\t_\t9\tcompmod\t_\t_\n" +
            "9\tlaw\t_\tNOUN\t_\t_\t5\tadpobj\t_\t_\n" +
            "10\tincludes\t_\tVERB\t_\t_\t0\tROOT\t_\t_\n" +
            "11\traising\t_\tVERB\t_\t_\t10\txcomp\t_\t_\n" +
            "12\t$\t_\t.\t_\t_\t11\tdobj\t_\t_\n" +
            "13\t30\t_\tNUM\t_\t_\t14\tnum\t_\t_\n" +
            "14\tbillion\t_\tNUM\t_\t_\t12\tnum\t_\t_\n" +
            "15\tfrom\t_\tADP\t_\t_\t11\tadpmod\t_\t_\n" +
            "16\tdebt\t_\tNOUN\t_\t_\t15\tadpobj\t_\t_\n" +
            "17\tissued\t_\tVERB\t_\t_\t16\tvmod\t_\t_\n" +
            "18\tby\t_\tADP\t_\t_\t17\tadpmod\t_\t_\n" +
            "19\tthe\t_\tDET\t_\t_\t22\tdet\t_\t_\n" +
            "20\tnewly\t_\tADV\t_\t_\t21\tadvmod\t_\t_\n" +
            "21\tcreated\t_\tVERB\t_\t_\t22\tamod\t_\t_\n" +
            "22\tRTC\t_\tNOUN\t_\t_\t18\tadpobj\t_\t_\n" +
            "23\t.\t_\t.\t_\t_\t10\tp\t_\t_";
    final String tmpPath = "/tmp/f.txt";

    @Test
    public void testFeatureExtraction() throws Exception {
        writeConllFile();

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1);
        ArrayList<Integer> dependencyLabels = new ArrayList<>();
        for (int lab : maps.getLabelMap().keySet())
            dependencyLabels.add(lab);

        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, false, false, maps);
        ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer("max_violation", new AveragedPerceptron(72, dependencyLabels.size()),
                options, dependencyLabels, 72, maps);
        MLPNetwork mlpNetwork = new MLPNetwork(maps, options, dependencyLabels, 64);
        ArrayList<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, 1, 0);

        Sentence sentence = dataSet.get(0).getSentence();//dummySentence(5,mlpNetwork.getNumOfWords()-2,mlpNetwork.getNumOfPos()-2);
        Configuration configuration = new Configuration(sentence, options.rootFirst);

        /**
         * actions = shift, shift, left-arc, left-arc, shift, right-arc, reduce, left-arc
         */
        int[] baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps);
        assert baseFeatures[0] == 1;
        assert baseFeatures[4] == maps.getNeuralWordKey(sentence.getWords()[0]);
        assert baseFeatures[19] == 1;
        assert baseFeatures[23] == maps.getNeuralPOSKey(sentence.getTags()[0]);
        assert baseFeatures[40] == 1;
        ArcEager.shift(configuration.state);
        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps);
        assert baseFeatures[0] == maps.getNeuralWordKey(sentence.getWords()[0]);
        assert baseFeatures[4] == maps.getNeuralWordKey(sentence.getWords()[1]);
        assert baseFeatures[19] == maps.getNeuralPOSKey(sentence.getTags()[0]);
        assert baseFeatures[23] == maps.getNeuralPOSKey(sentence.getTags()[1]);
        assert baseFeatures[40] == 1;
        ArcEager.shift(configuration.state);
        ArcEager.leftArc(configuration.state, 1);
        ArcEager.leftArc(configuration.state, 1);
        ArcEager.shift(configuration.state);
        ArcEager.shift(configuration.state);
        ArcEager.rightArc(configuration.state, 2);
        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps);
        assert baseFeatures[38] == maps.getNeuralDepRelationKey(configuration.state.getDependency(5));
        ArcEager.reduce(configuration.state);
        ArcEager.leftArc(configuration.state, 0);
    }

    private void writeConllFile() throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(tmpPath));
        writer.write(conllText);
        writer.close();
    }

    private Sentence dummySentence(int length, int numOfPos, int numOfWords) {
        ArrayList<Integer> tokens = new ArrayList<>();
        ArrayList<Integer> pos = new ArrayList<>();
        ArrayList<Integer> dummyB = new ArrayList<>();

        Random random = new Random(0);
        for (int i = 0; i < length; i++) {
            tokens.add(random.nextInt(numOfWords));
            pos.add(random.nextInt(numOfPos));
            dummyB.add(-1);
        }
        tokens.add(0);
        pos.add(0);
        dummyB.add(-1);
        return new Sentence(tokens, pos, dummyB, dummyB, dummyB);

    }
}
