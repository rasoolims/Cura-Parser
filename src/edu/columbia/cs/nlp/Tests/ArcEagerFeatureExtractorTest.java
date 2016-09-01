package edu.columbia.cs.nlp.Tests;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/2/16
 * Time: 2:59 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

import edu.columbia.cs.nlp.CuraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.Structures.Sentence;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Features.FeatureExtractor;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcEager;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class ArcEagerFeatureExtractorTest {
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
    final String shortConllText = "1\tTerms\t_\tNOUN\t_\t_\t4\tnsubjpass\t_\t_\n" +
            "2\twere\t_\tVERB\t_\t_\t4\tauxpass\t_\t_\n" +
            "3\tn't\t_\tADV\t_\t_\t2\tneg\t_\t_\n" +
            "4\tdisclosed\t_\tVERB\t_\t_\t0\tROOT\t_\t_\n" +
            "5\t.\t_\t.\t_\t_\t4\tp\t_\t_";
    final String shortConllOOVText = "1\tX\t_\tNOUN\t_\t_\t4\tnsubjpass\t_\t_\n" +
            "2\tZ\t_\tVERB\t_\t_\t4\tauxpass\t_\t_\n" +
            "3\tY\t_\tADV\t_\t_\t2\tneg\t_\t_\n" +
            "4\tZ\t_\tVERB\t_\t_\t0\tROOT\t_\t_\n" +
            "5\tX\t_\t.\t_\t_\t4\tp\t_\t_";

    final String tmpPath = "/tmp/f.txt";

    @Test
    public void testFeatureExtraction() throws Exception {
        ShiftReduceParser parser = new ArcEager();
        writeConllFile(conllText);

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1, false);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, false, false, maps);
        Sentence sentence = dataSet.get(0).getSentence();
        Configuration configuration = new Configuration(sentence, options.generalProperties.rootFirst);

        assert configuration.sentence.getWords()[12] == maps.word2Int("_NUM_");
        /**
         * actions = shift, shift, left-arc, left-arc, shift, right-arc, reduce, left-arc
         */
        double[] baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == IndexMaps.NullIndex;
        assert baseFeatures[4] == (sentence.getWords()[0]);
        assert baseFeatures[19] == IndexMaps.NullIndex;
        assert baseFeatures[26] == (sentence.getTags()[0]);
        assert baseFeatures[46] == maps.labelNullIndex;
        parser.shift(configuration.state);
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == (sentence.getWords()[0]);
        assert baseFeatures[4] == (sentence.getWords()[1]);
        assert baseFeatures[19] == (sentence.getTags()[0]);
        assert baseFeatures[26] == (sentence.getTags()[1]);
        assert baseFeatures[46] == maps.labelNullIndex;
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, 1);
        parser.leftArc(configuration.state, 1);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.rightArc(configuration.state, 2);
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[44] == (configuration.state.getDependency(5, maps.labelNullIndex));
        parser.reduce(configuration.state);
        int s0 = configuration.state.peek();
        parser.leftArc(configuration.state, maps.dep2Int("adpobj"));
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert configuration.state.getDependency(s0, maps.labelNullIndex) == baseFeatures[50];
        assert configuration.state.getDependency(configuration.state.bufferHead(), maps.labelNullIndex) == maps.labelNullIndex;
    }

    @Test
    public void testFeatureConsistency() throws Exception {
        ShiftReduceParser parser = new ArcEager();
        writeConllFile(shortConllText);

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1, false);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, false, false, maps);
        Sentence sentence = dataSet.get(0).getSentence();
        Configuration configuration = new Configuration(sentence, options.generalProperties.rootFirst);

        double[] baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[1] == IndexMaps.NullIndex;
        assert baseFeatures[23] == IndexMaps.NullIndex;
        assert baseFeatures[46] == maps.labelNullIndex;

        parser.shift(configuration.state);
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == sentence.getWords()[0];
        assert baseFeatures[0] == maps.word2Int("Terms");
        assert baseFeatures[1] == IndexMaps.NullIndex;
        assert baseFeatures[22] == sentence.getTags()[0];
        assert baseFeatures[22] == maps.pos2Int("NOUN");
        assert baseFeatures[23] == IndexMaps.NullIndex;
        assert baseFeatures[46] == maps.labelNullIndex;

        parser.shift(configuration.state);
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == sentence.getWords()[1];
        assert baseFeatures[0] == maps.word2Int("were");
        assert baseFeatures[1] == sentence.getWords()[0];
        assert baseFeatures[1] == maps.word2Int("Terms");
        assert baseFeatures[20] == sentence.getWords()[0];
        assert baseFeatures[20] == maps.word2Int("Terms");
        assert baseFeatures[21] == sentence.getWords()[2];
        assert baseFeatures[21] == maps.word2Int("n't");
        assert baseFeatures[19] == sentence.getWords()[1];
        assert baseFeatures[19] == maps.word2Int("were");
        assert baseFeatures[2] == IndexMaps.NullIndex;
        assert baseFeatures[22] == sentence.getTags()[1];
        assert baseFeatures[22] == maps.pos2Int("VERB");
        assert baseFeatures[23] == sentence.getTags()[0];
        assert baseFeatures[46] == maps.labelNullIndex;

        parser.rightArc(configuration.state, maps.dep2Int("neg"));
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == sentence.getWords()[2];
        assert baseFeatures[0] == maps.word2Int("n't");
        assert baseFeatures[2] == sentence.getWords()[0];
        assert baseFeatures[2] == maps.word2Int("Terms");
        assert baseFeatures[14] == maps.word2Int("were");
        assert baseFeatures[14] == sentence.getWords()[1];
        assert baseFeatures[4] == maps.word2Int("disclosed");
        assert baseFeatures[4] == sentence.getWords()[3];
        assert baseFeatures[3] == IndexMaps.NullIndex;
        assert baseFeatures[22] == sentence.getTags()[2];
        assert baseFeatures[22] == maps.pos2Int("ADV");
        assert baseFeatures[23] == sentence.getTags()[1];
        assert baseFeatures[44] == maps.dep2Int("neg");
        assert baseFeatures[47] == maps.labelNullIndex;

        parser.reduce(configuration.state);
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == sentence.getWords()[1];
        assert baseFeatures[0] == maps.word2Int("were");
        assert baseFeatures[1] == sentence.getWords()[0];
        assert baseFeatures[1] == maps.word2Int("Terms");
        assert baseFeatures[12] == sentence.getWords()[2];
        assert baseFeatures[12] == maps.word2Int("n't");
        assert baseFeatures[34] == sentence.getTags()[2];
        assert baseFeatures[34] == maps.pos2Int("ADV");
        assert baseFeatures[47] == maps.dep2Int("neg");
        assert baseFeatures[44] == maps.labelNullIndex;
        assert baseFeatures[6] == IndexMaps.RootIndex;
        assert baseFeatures[6] == maps.word2Int("ROOT");

        parser.leftArc(configuration.state, maps.dep2Int("auxpass"));
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == sentence.getWords()[0];
        assert baseFeatures[0] == maps.word2Int("Terms");
        assert baseFeatures[1] == IndexMaps.NullIndex;
        assert baseFeatures[8] == sentence.getWords()[1];
        assert baseFeatures[8] == maps.word2Int("were");
        assert baseFeatures[16] == IndexMaps.NullIndex;
        assert baseFeatures[30] == sentence.getTags()[1];
        assert baseFeatures[30] == maps.pos2Int("VERB");
        assert baseFeatures[50] == maps.dep2Int("auxpass");
        assert baseFeatures[51] == maps.labelNullIndex;

        parser.leftArc(configuration.state, maps.dep2Int("nsubjpass"));
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == IndexMaps.NullIndex;
        assert baseFeatures[1] == IndexMaps.NullIndex;
        assert baseFeatures[8] == sentence.getWords()[0];
        assert baseFeatures[8] == maps.word2Int("Terms");
        assert baseFeatures[9] == sentence.getWords()[1];
        assert baseFeatures[9] == maps.word2Int("were");
        assert baseFeatures[16] == IndexMaps.NullIndex;
        assert baseFeatures[30] == sentence.getTags()[0];
        assert baseFeatures[30] == maps.pos2Int("NOUN");
        assert baseFeatures[31] == sentence.getTags()[1];
        assert baseFeatures[31] == maps.pos2Int("VERB");
        assert baseFeatures[50] == maps.dep2Int("nsubjpass");
        assert baseFeatures[51] == maps.dep2Int("auxpass");
        assert baseFeatures[12] == IndexMaps.NullIndex;


        parser.shift(configuration.state);
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == sentence.getWords()[3];
        assert baseFeatures[0] == maps.word2Int("disclosed");
        assert baseFeatures[1] == IndexMaps.NullIndex;
        assert baseFeatures[10] == sentence.getWords()[0];
        assert baseFeatures[10] == maps.word2Int("Terms");
        assert baseFeatures[11] == sentence.getWords()[1];
        assert baseFeatures[11] == maps.word2Int("were");
        assert baseFeatures[32] == sentence.getTags()[0];
        assert baseFeatures[32] == maps.pos2Int("NOUN");
        assert baseFeatures[33] == sentence.getTags()[1];
        assert baseFeatures[33] == maps.pos2Int("VERB");
        assert baseFeatures[5] == sentence.getWords()[5];
        assert baseFeatures[5] == maps.word2Int("ROOT");
        assert baseFeatures[5] == IndexMaps.RootIndex;
        assert baseFeatures[27] == IndexMaps.RootIndex;
        assert baseFeatures[27] == maps.pos2Int("ROOT");
        assert baseFeatures[27] == sentence.getTags()[5];

        parser.rightArc(configuration.state, maps.dep2Int("p"));
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[14] == sentence.getWords()[3];
        assert baseFeatures[14] == maps.word2Int("disclosed");
        assert baseFeatures[47] == maps.labelNullIndex;
        assert baseFeatures[44] == maps.dep2Int("p");

        parser.reduce(configuration.state);
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[12] == sentence.getWords()[4];
        assert baseFeatures[12] == maps.word2Int(".");
        assert baseFeatures[14] == IndexMaps.NullIndex;
        assert baseFeatures[34] == sentence.getTags()[4];
        assert baseFeatures[34] == maps.pos2Int(".");

        parser.leftArc(configuration.state, maps.dep2Int("ROOT"));
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[16] == sentence.getWords()[0];
        assert baseFeatures[16] == maps.word2Int("Terms");
    }

    @Test
    public void testOOVFeatureConsistency() throws Exception {
        ShiftReduceParser parser = new ArcEager();
        writeConllFile(shortConllOOVText);

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", 1, false);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, false, false, maps);
        Sentence sentence = dataSet.get(0).getSentence();
        Configuration configuration = new Configuration(sentence, options.generalProperties.rootFirst);

        double[] baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[1] == IndexMaps.NullIndex;
        assert baseFeatures[23] == IndexMaps.NullIndex;
        assert baseFeatures[46] == maps.labelNullIndex;

        parser.shift(configuration.state);
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == sentence.getWords()[0];
        assert baseFeatures[0] == maps.word2Int("X");
        assert baseFeatures[1] == IndexMaps.NullIndex;
        assert baseFeatures[22] == sentence.getTags()[0];
        assert baseFeatures[22] == maps.pos2Int("NOUN");
        assert baseFeatures[23] == IndexMaps.NullIndex;
        assert baseFeatures[46] == maps.labelNullIndex;

        parser.shift(configuration.state);
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == sentence.getWords()[1];
        assert baseFeatures[0] == maps.word2Int("Z");
        assert baseFeatures[1] == sentence.getWords()[0];
        assert baseFeatures[1] == maps.word2Int("X");
        assert baseFeatures[2] == IndexMaps.NullIndex;
        assert baseFeatures[22] == sentence.getTags()[1];
        assert baseFeatures[22] == maps.pos2Int("VERB");
        assert baseFeatures[23] == sentence.getTags()[0];
        assert baseFeatures[46] == maps.labelNullIndex;

        parser.rightArc(configuration.state, maps.dep2Int("neg"));
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == sentence.getWords()[2];
        assert baseFeatures[0] == IndexMaps.UnknownIndex;
        assert baseFeatures[2] == sentence.getWords()[0];
        assert baseFeatures[2] == maps.word2Int("X");
        assert baseFeatures[14] == maps.word2Int("Z");
        assert baseFeatures[14] == sentence.getWords()[1];
        assert baseFeatures[4] == maps.word2Int("Z");
        assert baseFeatures[4] == sentence.getWords()[3];
        assert baseFeatures[3] == IndexMaps.NullIndex;
        assert baseFeatures[22] == sentence.getTags()[2];
        assert baseFeatures[22] == maps.pos2Int("ADV");
        assert baseFeatures[23] == sentence.getTags()[1];
        assert baseFeatures[44] == maps.dep2Int("neg");
        assert baseFeatures[47] == maps.labelNullIndex;

        parser.reduce(configuration.state);
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == sentence.getWords()[1];
        assert baseFeatures[0] == maps.word2Int("Z");
        assert baseFeatures[1] == sentence.getWords()[0];
        assert baseFeatures[1] == maps.word2Int("X");
        assert baseFeatures[12] == sentence.getWords()[2];
        assert baseFeatures[12] == IndexMaps.UnknownIndex;
        assert baseFeatures[34] == sentence.getTags()[2];
        assert baseFeatures[34] == maps.pos2Int("ADV");
        assert baseFeatures[47] == maps.dep2Int("neg");
        assert baseFeatures[41] == maps.pos2Int("ADV");
        assert baseFeatures[6] == IndexMaps.RootIndex;
        assert baseFeatures[6] == maps.word2Int("ROOT");

        parser.leftArc(configuration.state, maps.dep2Int("auxpass"));
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == sentence.getWords()[0];
        assert baseFeatures[0] == maps.word2Int("X");
        assert baseFeatures[1] == IndexMaps.NullIndex;
        assert baseFeatures[8] == sentence.getWords()[1];
        assert baseFeatures[8] == maps.word2Int("Z");
        assert baseFeatures[16] == IndexMaps.NullIndex;
        assert baseFeatures[30] == sentence.getTags()[1];
        assert baseFeatures[30] == maps.pos2Int("VERB");
        assert baseFeatures[50] == maps.dep2Int("auxpass");
        assert baseFeatures[51] == maps.labelNullIndex;

        parser.leftArc(configuration.state, maps.dep2Int("nsubjpass"));
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == IndexMaps.NullIndex;
        assert baseFeatures[1] == IndexMaps.NullIndex;
        assert baseFeatures[8] == sentence.getWords()[0];
        assert baseFeatures[8] == maps.word2Int("X");
        assert baseFeatures[9] == sentence.getWords()[1];
        assert baseFeatures[9] == maps.word2Int("Z");
        assert baseFeatures[16] == IndexMaps.NullIndex;
        assert baseFeatures[30] == sentence.getTags()[0];
        assert baseFeatures[30] == maps.pos2Int("NOUN");
        assert baseFeatures[31] == sentence.getTags()[1];
        assert baseFeatures[31] == maps.pos2Int("VERB");
        assert baseFeatures[50] == maps.dep2Int("nsubjpass");
        assert baseFeatures[51] == maps.dep2Int("auxpass");
        assert baseFeatures[12] == IndexMaps.NullIndex;


        parser.shift(configuration.state);
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[0] == sentence.getWords()[3];
        assert baseFeatures[0] == maps.word2Int("Z");
        assert baseFeatures[1] == IndexMaps.NullIndex;
        assert baseFeatures[10] == sentence.getWords()[0];
        assert baseFeatures[10] == maps.word2Int("X");
        assert baseFeatures[11] == sentence.getWords()[1];
        assert baseFeatures[11] == maps.word2Int("Z");
        assert baseFeatures[32] == sentence.getTags()[0];
        assert baseFeatures[32] == maps.pos2Int("NOUN");
        assert baseFeatures[33] == sentence.getTags()[1];
        assert baseFeatures[33] == maps.pos2Int("VERB");
        assert baseFeatures[5] == sentence.getWords()[5];
        assert baseFeatures[5] == maps.word2Int("ROOT");
        assert baseFeatures[5] == IndexMaps.RootIndex;
        assert baseFeatures[27] == IndexMaps.RootIndex;
        assert baseFeatures[27] == maps.pos2Int("ROOT");
        assert baseFeatures[27] == sentence.getTags()[5];

        parser.rightArc(configuration.state, maps.dep2Int("p"));
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[14] == sentence.getWords()[3];
        assert baseFeatures[14] == maps.word2Int("Z");
        assert baseFeatures[47] == maps.labelNullIndex;
        assert baseFeatures[44] == maps.dep2Int("p");

        parser.reduce(configuration.state);
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[12] == sentence.getWords()[4];
        assert baseFeatures[12] == maps.word2Int("X");
        assert baseFeatures[14] == IndexMaps.NullIndex;
        assert baseFeatures[34] == sentence.getTags()[4];
        assert baseFeatures[34] == maps.pos2Int(".");

        parser.leftArc(configuration.state, maps.dep2Int("ROOT"));
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert baseFeatures[16] == sentence.getWords()[0];
        assert baseFeatures[16] == maps.word2Int("X");
    }

    private void writeConllFile(String txt) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(tmpPath));
        writer.write(txt);
        writer.close();
    }
}
