package Tests;

import YaraParser.Accessories.CoNLLReader;
import YaraParser.Accessories.Options;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.Sentence;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Parser.ArcEager;
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

public class ArcEagerTest {
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

        ArcEager.shift(configuration.state);
        ArcEager.shift(configuration.state);
        ArcEager.rightArc(configuration.state, maps.dep2Int("neg"));
        ArcEager.reduce(configuration.state);
        ArcEager.leftArc(configuration.state, maps.dep2Int("auxpass"));
        ArcEager.leftArc(configuration.state, maps.dep2Int("nsubjpass"));
        ArcEager.shift(configuration.state);
        ArcEager.rightArc(configuration.state, maps.dep2Int("p"));
        ArcEager.reduce(configuration.state);
        ArcEager.leftArc(configuration.state, maps.dep2Int("ROOT"));
        assert configuration.state.isTerminalState();
    }

}
