package edu.columbia.cs.nlp.Tests;

import edu.columbia.cs.nlp.CuraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Accessories.Utils;
import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.Structures.NeuralTrainingInstance;
import edu.columbia.cs.nlp.CuraParser.Structures.Sentence;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Features.FeatureExtractor;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.ParserType;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcEager;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Trainer.GreedyTrainer;
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

public class StaticTrainerTest {
    final String shortConllText = "1\tTerms\t_\tNOUN\t_\t_\t4\tnsubjpass\t_\t_\n" +
            "2\twere\t_\tVERB\t_\t_\t4\tauxpass\t_\t_\n" +
            "3\tn't\t_\tADV\t_\t_\t2\tneg\t_\t_\n" +
            "4\tdisclosed\t_\tVERB\t_\t_\t0\tROOT\t_\t_\n" +
            "5\t.\t_\t.\t_\t_\t4\tp\t_\t_";
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

    private void writeConllFile(String txt) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(tmpPath));
        writer.write(txt);
        writer.close();
    }

    @Test
    public void testInstanceGenerationForArcEager() throws Exception {
        ShiftReduceParser parser = new ArcEager();
        writeConllFile(shortConllText);

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1, false);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        options.generalProperties.parserType = ParserType.ArcEager;
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, false, false, maps);
        Sentence sentence = dataSet.get(0).getSentence();

        ArrayList<Integer> dependencyLabels = new ArrayList<>();
        for (int lab = 0; lab < maps.relSize(); lab++)
            dependencyLabels.add(lab);
        GreedyTrainer trainer = new GreedyTrainer(options, dependencyLabels,
                maps.labelNullIndex, maps.rareWords);
        ArrayList<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, dataSet.size(), 0);
        Configuration configuration = new Configuration(sentence, options.generalProperties.rootFirst);

        int index = 0;
        double[] baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 0;
        for (int i = 1; i < instances.get(index).getLabel().length; i++)
            assert instances.get(index).getLabel()[i] == -1;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);

        parser.shift(configuration.state);

        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 0;
        assert instances.get(index).getLabel()[1] == -1;
        for (int i = 2; i < instances.get(index).getLabel().length; i++)
            assert instances.get(index).getLabel()[i] == 0;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.shift(configuration.state);

        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 2 + maps.dep2Int("neg");
        assert instances.get(index).getLabel()[1] == -1;
        for (int i = 0; i < instances.get(index).getLabel().length; i++)
            if (i != instances.get(index).gold() && i != 1)
                assert instances.get(index).getLabel()[i] == 0;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.rightArc(configuration.state, maps.dep2Int("neg"));

        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
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

        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
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

        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 2 + dependencyLabels.size() + maps.dep2Int("nsubjpass");
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.leftArc(configuration.state, maps.dep2Int("nsubjpass"));

        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 0;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.shift(configuration.state);

        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 2 + maps.dep2Int("p");
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.rightArc(configuration.state, maps.dep2Int("p"));

        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert instances.get(index).gold() == 1;
        assert Utils.equals(instances.get(index++).getFeatures(), baseFeatures);
        parser.reduce(configuration.state);

        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
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

    @Test
    public void testOracleForArcStandard() throws Exception {
        writeConllFile(conllText);

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1, false);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        options.generalProperties.parserType = ParserType.ArcStandard;
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, options.generalProperties.rootFirst, false, maps);

        ArrayList<Integer> dependencyLabels = new ArrayList<>();
        for (int lab = 0; lab < maps.relSize(); lab++)
            dependencyLabels.add(lab);
        GreedyTrainer trainer = new GreedyTrainer(options, dependencyLabels,
                maps.labelNullIndex, maps.rareWords);
        ArrayList<NeuralTrainingInstance> instances = trainer.getNextInstances(dataSet, 0, dataSet.size(), 0);
        int lIndex = 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("compmod"); // 3<-4
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("amod"); // 2<-4
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("det"); // 1<-4
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("compmod"); // 8<-9
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("compmod"); // 7<-9
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("det"); // 6<-9
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("adpobj"); // 5->9
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("adpmod"); // 4->5
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("nsubj"); // 4<-10
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("num"); // 13<-14
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("num"); // 12->14
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("dobj"); // 11->12
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("advmod"); // 20<-21
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("amod"); // 21<-22
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("det"); // 19<-22
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("adpobj"); // 18->22
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("adpmod"); // 17->18
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("vmod"); // 16->17
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("adpobj"); // 15->16
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("adpmod"); // 11->15
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("xcomp"); // 10->11
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("p"); // 10->23
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("ROOT"); // 10<-ROOT
        assert instances.size() == lIndex;

        options.generalProperties.rootFirst = true;
        lIndex = 0;
        reader = new CoNLLReader(tmpPath);
        dataSet = reader.readData(Integer.MAX_VALUE, false, true, options.generalProperties.rootFirst, false, maps);

        dependencyLabels = new ArrayList<>();
        for (int lab = 0; lab < maps.relSize(); lab++)
            dependencyLabels.add(lab);
        trainer = new GreedyTrainer(options, dependencyLabels,
                maps.labelNullIndex, maps.rareWords);
        instances = trainer.getNextInstances(dataSet, 0, dataSet.size(), 0);
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("compmod"); // 3<-4
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("amod"); // 2<-4
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("det"); // 1<-4
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("compmod"); // 8<-9
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("compmod"); // 7<-9
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("det"); // 6<-9
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("adpobj"); // 5->9
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("adpmod"); // 4->5
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("nsubj"); // 4<-10
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("num"); // 13<-14
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("num"); // 12->14
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("dobj"); // 11->12
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("advmod"); // 20<-21
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("amod"); // 21<-22
        assert instances.get(lIndex++).gold() == 2 + dependencyLabels.size() + maps.dep2Int("det"); // 19<-22
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("adpobj"); // 18->22
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("adpmod"); // 17->18
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("vmod"); // 16->17
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("adpobj"); // 15->16
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("adpmod"); // 11->15
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("xcomp"); // 10->11
        assert instances.get(lIndex++).gold() == 0;
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("p"); // 10->23
        assert instances.get(lIndex++).gold() == 2 + maps.dep2Int("ROOT"); // ROOT -> 10
        assert instances.size() == lIndex;
    }
}
