package edu.columbia.cs.nlp.Tests;

import edu.columbia.cs.nlp.YaraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.YaraParser.Accessories.Options;
import edu.columbia.cs.nlp.YaraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.YaraParser.Structures.Sentence;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Parsers.ArcEager;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Parsers.ArcStandard;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

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
    final String tmpPath = "/tmp/f.txt";

    private void writeConllFile(String txt) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(tmpPath));
        writer.write(txt);
        writer.close();
    }

    @Test
    public void testArcEagerActions() throws Exception {
        writeConllFile(shortConllText);

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, false, false, maps);
        Sentence sentence = dataSet.get(0).getSentence();
        Configuration configuration = new Configuration(sentence, options.rootFirst);
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

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        options.rootFirst = false;
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, options.rootFirst, false, maps);
        Sentence sentence = dataSet.get(0).getSentence();
        Configuration configuration = new Configuration(sentence, options.rootFirst);
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
        options.rootFirst = true;
        dataSet = reader.readData(Integer.MAX_VALUE, false, true, options.rootFirst, false, maps);
        sentence = dataSet.get(0).getSentence();
        configuration = new Configuration(sentence, options.rootFirst);
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
}
