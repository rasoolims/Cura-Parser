package Tests;

import YaraParser.Accessories.CoNLLReader;
import YaraParser.Accessories.Options;
import YaraParser.Accessories.Utils;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.NeuralTrainingInstance;
import YaraParser.Structures.Sentence;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;
import YaraParser.TransitionBasedSystem.Parser.ArcEager.ArcEager;
import YaraParser.TransitionBasedSystem.Trainer.ArcEagerBeamTrainer;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/10/16
 * Time: 7:39 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class ArcEagerBeamTrainerTest {
    final String shortConllText = "1\tTerms\t_\tNOUN\t_\t_\t4\tnsubjpass\t_\t_\n" +
            "2\twere\t_\tVERB\t_\t_\t4\tauxpass\t_\t_\n" +
            "3\tn't\t_\tADV\t_\t_\t2\tneg\t_\t_\n" +
            "4\tdisclosed\t_\tVERB\t_\t_\t0\tROOT\t_\t_\n" +
            "5\t.\t_\t.\t_\t_\t4\tp\t_\t_";
    final String tmpPath = "/tmp/f.txt";

    private void writeConllFile(String txt) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(tmpPath));
        writer.write(txt);
        writer.close();
    }

    @Test
    public void testInstanceGeneration() throws Exception {
        writeConllFile(shortConllText);

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, false, false, maps);
        Sentence sentence = dataSet.get(0).getSentence();

        ArrayList<Integer> dependencyLabels = new ArrayList<>();
        for (int lab = 0; lab < maps.relSize(); lab++)
            dependencyLabels.add(lab);
        ArcEagerBeamTrainer trainer = new ArcEagerBeamTrainer(options.useMaxViol ? "max_violation" : "early", options, dependencyLabels);
        ArrayList<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, dataSet.size(), 0);
        Configuration configuration = new Configuration(sentence, options.rootFirst);

        int index = 0;
        int[] baseFeatures = FeatureExtractor.extractBaseFeatures(configuration);
        assert instances.get(index).gold() == 0;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        ArcEager.shift(configuration.state);

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration);
        assert instances.get(index).gold() == 0;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        ArcEager.shift(configuration.state);

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration);
        assert instances.get(index).gold() == 2 + maps.dep2Int("neg");
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        ArcEager.rightArc(configuration.state, maps.dep2Int("neg"));

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration);
        assert instances.get(index).gold() == 1;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        ArcEager.reduce(configuration.state);

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration);
        assert instances.get(index).gold() == 2 + dependencyLabels.size() + maps.dep2Int("auxpass");
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        ArcEager.leftArc(configuration.state, maps.dep2Int("auxpass"));

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration);
        assert instances.get(index).gold() == 2 + dependencyLabels.size() + maps.dep2Int("nsubjpass");
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        ArcEager.leftArc(configuration.state, maps.dep2Int("nsubjpass"));

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration);
        assert instances.get(index).gold() == 0;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        ArcEager.shift(configuration.state);

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration);
        assert instances.get(index).gold() == 2 + maps.dep2Int("p");
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        ArcEager.rightArc(configuration.state, maps.dep2Int("p"));

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration);
        assert instances.get(index).gold() == 1;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        ArcEager.reduce(configuration.state);

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration);
        assert instances.get(index).gold() == 2 + dependencyLabels.size() + maps.dep2Int("ROOT");
        assert instances.get(index).gold() == 2 + dependencyLabels.size() + IndexMaps.LabelRootIndex;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        ArcEager.leftArc(configuration.state, maps.dep2Int("ROOT"));

        assert instances.size() == index;
    }
}
