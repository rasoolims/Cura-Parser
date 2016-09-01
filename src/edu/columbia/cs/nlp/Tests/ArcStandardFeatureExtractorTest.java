package edu.columbia.cs.nlp.Tests;

import edu.columbia.cs.nlp.CuraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.Structures.Sentence;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Features.FeatureExtractor;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.Actions;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcStandard;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/15/16
 * Time: 1:04 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */
public class ArcStandardFeatureExtractorTest {
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
    public void testFeatureExtractionWithRootAtLast() throws Exception {
        ShiftReduceParser parser = new ArcStandard();
        writeConllFile(conllText);

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1, false);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        options.generalProperties.rootFirst = false;
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, options.generalProperties.rootFirst, false, maps);
        Sentence sentence = dataSet.get(0).getSentence();
        Configuration configuration = new Configuration(sentence, options.generalProperties.rootFirst);


        /**
         * actions = shift, shift, left-arc, left-arc, shift, right-arc, reduce, left-arc
         */
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("compmod"));
        parser.leftArc(configuration.state, maps.dep2Int("amod"));
        parser.leftArc(configuration.state, maps.dep2Int("det"));
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("compmod"));
        parser.leftArc(configuration.state, maps.dep2Int("compmod"));
        parser.leftArc(configuration.state, maps.dep2Int("det"));
        parser.rightArc(configuration.state, maps.dep2Int("adpobj"));  // 5->9
        parser.rightArc(configuration.state, maps.dep2Int("adpmod"));   // 4->5
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("num"));   // 13<-14
        parser.rightArc(configuration.state, maps.dep2Int("num"));   // 12->14
        parser.rightArc(configuration.state, maps.dep2Int("dobj"));   // 11->12
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("advmod"));   // 20<-21
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("amod"));   // 21<-22
        parser.leftArc(configuration.state, maps.dep2Int("det"));   // 19<-22
        parser.rightArc(configuration.state, maps.dep2Int("adpobj"));   // 18->22
        parser.rightArc(configuration.state, maps.dep2Int("adpmod"));   // 17->18
        parser.rightArc(configuration.state, maps.dep2Int("vmod"));   // 16->17
        parser.rightArc(configuration.state, maps.dep2Int("adpobj"));   // 15->16
        parser.rightArc(configuration.state, maps.dep2Int("adpmod"));   // 11->15

        double[] baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert configuration.state.stackSize() == 3;
        int fIndex = 0;
        assert baseFeatures[fIndex++] == maps.word2Int("raising");
        assert baseFeatures[fIndex++] == maps.word2Int("includes");
        assert baseFeatures[fIndex++] == maps.word2Int("plan");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert configuration.state.bufferSize() == 2;
        assert baseFeatures[fIndex++] == maps.word2Int(".");
        assert baseFeatures[fIndex++] == IndexMaps.RootIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("from");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("$");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("debt");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;

        assert baseFeatures[fIndex++] == maps.pos2Int("VERB");
        assert baseFeatures[fIndex++] == maps.pos2Int("VERB");
        assert baseFeatures[fIndex++] == maps.pos2Int("NOUN");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int(".");
        assert baseFeatures[fIndex++] == IndexMaps.RootIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int("ADP");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int(".");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int("NOUN");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;

        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("adpmod");
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("dobj");
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("adpobj");
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        parser.rightArc(configuration.state, maps.dep2Int("xcomp"));   // 10->11

        fIndex = 0;
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert configuration.state.stackSize() == 2;
        assert baseFeatures[fIndex++] == maps.word2Int("includes");
        assert baseFeatures[fIndex++] == maps.word2Int("plan");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert configuration.state.bufferSize() == 2;
        assert baseFeatures[fIndex++] == maps.word2Int(".");
        assert baseFeatures[fIndex++] == IndexMaps.RootIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("raising");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("The");
        assert baseFeatures[fIndex++] == maps.word2Int("in");
        assert baseFeatures[fIndex++] == maps.word2Int("complex");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("from");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("law");

        assert baseFeatures[fIndex++] == maps.pos2Int("VERB");
        assert baseFeatures[fIndex++] == maps.pos2Int("NOUN");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int(".");
        assert baseFeatures[fIndex++] == IndexMaps.RootIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int("VERB");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int("DET");
        assert baseFeatures[fIndex++] == maps.pos2Int("ADP");
        assert baseFeatures[fIndex++] == maps.pos2Int("ADJ");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int("ADP");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int("NOUN");

        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("xcomp");
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("det");
        assert baseFeatures[fIndex++] == maps.dep2Int("adpmod");
        assert baseFeatures[fIndex++] == maps.dep2Int("amod");
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("adpmod");
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("adpobj");
        parser.leftArc(configuration.state, maps.dep2Int("nsubj"));   // 10->4

        parser.shift(configuration.state);
        parser.rightArc(configuration.state, maps.dep2Int("p"));   // 10->23
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("ROOT"));
        assert configuration.state.isTerminalState();
    }

    @Test
    public void testFeatureExtractionWithRootAtBegining() throws Exception {
        ShiftReduceParser parser = new ArcStandard();
        writeConllFile(conllText);

        IndexMaps maps = CoNLLReader.createIndices(tmpPath, true, false, "", -1, false);
        CoNLLReader reader = new CoNLLReader(tmpPath);
        Options options = new Options();
        options.generalProperties.rootFirst = true;
        ArrayList<GoldConfiguration> dataSet = reader.readData(Integer.MAX_VALUE, false, true, options.generalProperties.rootFirst, false, maps);
        Sentence sentence = dataSet.get(0).getSentence();
        Configuration configuration = new Configuration(sentence, options.generalProperties.rootFirst);


        /**
         * actions = shift, shift, left-arc, left-arc, shift, right-arc, reduce, left-arc
         */
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("compmod"));
        parser.leftArc(configuration.state, maps.dep2Int("amod"));
        parser.leftArc(configuration.state, maps.dep2Int("det"));
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("compmod"));
        parser.leftArc(configuration.state, maps.dep2Int("compmod"));
        parser.leftArc(configuration.state, maps.dep2Int("det"));
        parser.rightArc(configuration.state, maps.dep2Int("adpobj"));  // 5->9
        parser.rightArc(configuration.state, maps.dep2Int("adpmod"));   // 4->5
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("num"));   // 13<-14
        parser.rightArc(configuration.state, maps.dep2Int("num"));   // 12->14
        parser.rightArc(configuration.state, maps.dep2Int("dobj"));   // 11->12
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("advmod"));   // 20<-21
        parser.shift(configuration.state);
        parser.leftArc(configuration.state, maps.dep2Int("amod"));   // 21<-22
        parser.leftArc(configuration.state, maps.dep2Int("det"));   // 19<-22
        parser.rightArc(configuration.state, maps.dep2Int("adpobj"));   // 18->22
        parser.rightArc(configuration.state, maps.dep2Int("adpmod"));   // 17->18
        parser.rightArc(configuration.state, maps.dep2Int("vmod"));   // 16->17
        parser.rightArc(configuration.state, maps.dep2Int("adpobj"));   // 15->16
        parser.rightArc(configuration.state, maps.dep2Int("adpmod"));   // 11->15

        double[] baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert configuration.state.stackSize() == 4;
        int fIndex = 0;
        assert baseFeatures[fIndex++] == maps.word2Int("raising");
        assert baseFeatures[fIndex++] == maps.word2Int("includes");
        assert baseFeatures[fIndex++] == maps.word2Int("plan");
        assert baseFeatures[fIndex++] == IndexMaps.RootIndex;
        assert configuration.state.bufferSize() == 1;
        assert baseFeatures[fIndex++] == maps.word2Int(".");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("from");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("$");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("debt");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;

        assert baseFeatures[fIndex++] == maps.pos2Int("VERB");
        assert baseFeatures[fIndex++] == maps.pos2Int("VERB");
        assert baseFeatures[fIndex++] == maps.pos2Int("NOUN");
        assert baseFeatures[fIndex++] == IndexMaps.RootIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int(".");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int("ADP");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int(".");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int("NOUN");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;

        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("adpmod");
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("dobj");
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("adpobj");
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        parser.rightArc(configuration.state, maps.dep2Int("xcomp"));   // 10->11

        fIndex = 0;
        baseFeatures = FeatureExtractor.extractFeatures(configuration, maps.labelNullIndex, parser);
        assert configuration.state.stackSize() == 3;
        assert baseFeatures[fIndex++] == maps.word2Int("includes");
        assert baseFeatures[fIndex++] == maps.word2Int("plan");
        assert baseFeatures[fIndex++] == IndexMaps.RootIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert configuration.state.bufferSize() == 1;
        assert baseFeatures[fIndex++] == maps.word2Int(".");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("raising");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("The");
        assert baseFeatures[fIndex++] == maps.word2Int("in");
        assert baseFeatures[fIndex++] == maps.word2Int("complex");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("from");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.word2Int("law");

        assert baseFeatures[fIndex++] == maps.pos2Int("VERB");
        assert baseFeatures[fIndex++] == maps.pos2Int("NOUN");
        assert baseFeatures[fIndex++] == IndexMaps.RootIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int(".");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int("VERB");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int("DET");
        assert baseFeatures[fIndex++] == maps.pos2Int("ADP");
        assert baseFeatures[fIndex++] == maps.pos2Int("ADJ");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int("ADP");
        assert baseFeatures[fIndex++] == IndexMaps.NullIndex;
        assert baseFeatures[fIndex++] == maps.pos2Int("NOUN");

        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("xcomp");
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("det");
        assert baseFeatures[fIndex++] == maps.dep2Int("adpmod");
        assert baseFeatures[fIndex++] == maps.dep2Int("amod");
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("adpmod");
        assert baseFeatures[fIndex++] == maps.labelNullIndex;
        assert baseFeatures[fIndex++] == maps.dep2Int("adpobj");
        parser.leftArc(configuration.state, maps.dep2Int("nsubj"));   // 10->4

        parser.shift(configuration.state);
        parser.rightArc(configuration.state, maps.dep2Int("p"));   // 10->23
        assert !parser.canDo(Actions.LeftArc, configuration.state);
        parser.rightArc(configuration.state, maps.dep2Int("ROOT"));
        assert configuration.state.isTerminalState();
    }


    private void writeConllFile(String txt) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(tmpPath));
        writer.write(txt);
        writer.close();
    }
}

