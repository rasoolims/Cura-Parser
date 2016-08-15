package edu.columbia.cs.nlp.Tests;

import edu.columbia.cs.nlp.YaraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.YaraParser.Accessories.Options;
import edu.columbia.cs.nlp.YaraParser.Accessories.Utils;
import edu.columbia.cs.nlp.YaraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.YaraParser.Structures.NeuralTrainingInstance;
import edu.columbia.cs.nlp.YaraParser.Structures.Sentence;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Features.FeatureExtractor;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Parsers.ArcEager;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Trainer.BeamTrainer;
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
        ShiftReduceParser parser = new ArcEager();
        writeConllFile(shortConllText);

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, false, false, maps);
        Sentence sentence = dataSet.get(0).getSentence();

        ArrayList<Integer> dependencyLabels = new ArrayList<>();
        for (int lab = 0; lab < maps.relSize(); lab++)
            dependencyLabels.add(lab);
        BeamTrainer trainer = new BeamTrainer(options.useMaxViol ? "max_violation" : "early", options, dependencyLabels,
                maps.labelNullIndex, maps.rareWords);
        ArrayList<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, dataSet.size(), 0);
        Configuration configuration = new Configuration(sentence, options.rootFirst);

        int index = 0;
        int[] baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 0;
        for (int i = 1; i < instances.get(index).getLabel().length; i++)
            assert instances.get(index).getLabel()[i] == -1;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);

        parser.shift(configuration.state);

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 0;
        assert instances.get(index).getLabel()[1] == -1;
        for (int i = 2; i < instances.get(index).getLabel().length; i++)
            assert instances.get(index).getLabel()[i] == 0;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.shift(configuration.state);

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 2 + maps.dep2Int("neg");
        assert instances.get(index).getLabel()[1] == -1;
        for (int i = 0; i < instances.get(index).getLabel().length; i++)
            if (i != instances.get(index).gold() && i != 1)
                assert instances.get(index).getLabel()[i] == 0;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.rightArc(configuration.state, maps.dep2Int("neg"));

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 1;
        for (int i = 0; i < instances.get(index).getLabel().length; i++) {
            if (i != instances.get(index).gold()) {
                if (i < 2 + dependencyLabels.size())
                    assert instances.get(index).getLabel()[i] == 0;
                else assert instances.get(index).getLabel()[i] == -1;
            }
        }
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.reduce(configuration.state);

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 2 + dependencyLabels.size() + maps.dep2Int("auxpass");
        assert instances.get(index).getLabel()[1] == -1;
        for (int i = 0; i < instances.get(index).getLabel().length; i++) {
            if (i != instances.get(index).gold()) {
                if (i != 1)
                    assert instances.get(index).getLabel()[i] == 0;
            }
        }
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.leftArc(configuration.state, maps.dep2Int("auxpass"));

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 2 + dependencyLabels.size() + maps.dep2Int("nsubjpass");
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.leftArc(configuration.state, maps.dep2Int("nsubjpass"));

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 0;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.shift(configuration.state);

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 2 + maps.dep2Int("p");
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.rightArc(configuration.state, maps.dep2Int("p"));

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 1;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.reduce(configuration.state);

        baseFeatures = FeatureExtractor.extractBaseFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 2 + dependencyLabels.size() + maps.dep2Int("ROOT");
        assert instances.get(index).gold() == 2 + dependencyLabels.size() + maps.LabelRootIndex;
        for (int i = 0; i < instances.get(index).getLabel().length; i++) {
            if (i != instances.get(index).gold()) {
                if (i >= 1 && i < 2 + dependencyLabels.size())
                    assert instances.get(index).getLabel()[i] == -1;
            }
        }
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.leftArc(configuration.state, maps.dep2Int("ROOT"));

        assert instances.size() == index;
    }
}
