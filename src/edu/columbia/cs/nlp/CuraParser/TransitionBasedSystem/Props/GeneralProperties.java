package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Props;

import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.ParserType;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.HashSet;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/17/16
 * Time: 12:00 PM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class GeneralProperties implements Serializable {
    public boolean showHelp;
    public boolean evaluate;
    public boolean train;
    public boolean parseTaggedFile;
    public boolean parseConllFile;
    public boolean parsePartialConll;
    public String modelFile;
    public int beamWidth;
    public boolean rootFirst;
    public boolean labeled;
    public boolean lowercase;
    public String inputFile;
    public String outputFile;
    public int numOfThreads;
    public HashSet<String> punctuations;
    public ParserType parserType;


    public GeneralProperties() {
        showHelp = false;
        train = false;
        parseConllFile = false;
        parseTaggedFile = false;
        beamWidth = 1;
        rootFirst = false;
        modelFile = "";
        outputFile = "";
        inputFile = "";
        labeled = true;
        lowercase = false;
        evaluate = false;
        numOfThreads = 8;
        parsePartialConll = false;
        parserType = ParserType.ArcEager;
        initializePunctuations();
    }

    private GeneralProperties(boolean showHelp, boolean evaluate, boolean train, boolean parseTaggedFile, boolean parseConllFile, boolean
            parsePartialConll, String modelFile, int beamWidth, boolean rootFirst, boolean labeled, boolean lowercase, String inputFile, String
                                      outputFile, int numOfThreads, HashSet<String> punctuations, ParserType parserType) {
        this.showHelp = showHelp;
        this.evaluate = evaluate;
        this.train = train;
        this.parseTaggedFile = parseTaggedFile;
        this.parseConllFile = parseConllFile;
        this.parsePartialConll = parsePartialConll;
        this.modelFile = modelFile;
        this.beamWidth = beamWidth;
        this.rootFirst = rootFirst;
        this.labeled = labeled;
        this.lowercase = lowercase;
        this.inputFile = inputFile;
        this.outputFile = outputFile;
        this.numOfThreads = numOfThreads;
        this.punctuations = punctuations;
        this.parserType = parserType;
    }

    private void initializePunctuations() {
        punctuations = new HashSet<>();
        punctuations.add("#");
        punctuations.add("''");
        punctuations.add("(");
        punctuations.add(")");
        punctuations.add("[");
        punctuations.add("]");
        punctuations.add("{");
        punctuations.add("}");
        punctuations.add("\"");
        punctuations.add(",");
        punctuations.add(".");
        punctuations.add(":");
        punctuations.add("``");
        punctuations.add("-LRB-");
        punctuations.add("-RRB-");
        punctuations.add("-LSB-");
        punctuations.add("-RSB-");
        punctuations.add("-LCB-");
        punctuations.add("-RCB-");
        punctuations.add("!");
        punctuations.add(".");
        punctuations.add("#");
        punctuations.add("$");
        punctuations.add("''");
        punctuations.add("(");
        punctuations.add(")");
        punctuations.add(",");
        punctuations.add("-LRB-");
        punctuations.add("-RRB-");
        punctuations.add(":");
        punctuations.add("?");
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        if (train) {
            builder.append("beam width: " + beamWidth + "\n");
            builder.append("rootFirst: " + rootFirst + "\n");
            builder.append("labeled: " + labeled + "\n");
            builder.append("lower-case: " + lowercase + "\n");
            builder.append("number of threads: " + numOfThreads + "\n");
            builder.append("parser type: " + parserType + "\n");
        } else if (parseConllFile) {
            builder.append("parse conll" + "\n");
            builder.append("input file: " + inputFile + "\n");
            builder.append("output file: " + outputFile + "\n");
            builder.append("model file: " + modelFile + "\n");
            builder.append("number of threads: " + numOfThreads + "\n");
        } else if (parseConllFile) {
            builder.append("parse partial conll" + "\n");
            builder.append("input file: " + inputFile + "\n");
            builder.append("output file: " + outputFile + "\n");
            builder.append("model file: " + modelFile + "\n");
            builder.append("labeled: " + labeled + "\n");
            builder.append("number of threads: " + numOfThreads + "\n");
        } else if (evaluate) {
            builder.append("input file: " + inputFile + "\n");
            builder.append("parsed file: " + outputFile + "\n");
        }
        return builder.toString();
    }

    public void changePunc(String puncPath) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(puncPath));

        punctuations = new HashSet<>();
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.length() > 0)
                punctuations.add(line.split(" ")[0].trim());
        }
    }

    /**
     * Punctuation is not deep copy though.
     *
     * @return
     */
    @Override
    public GeneralProperties clone() {
        return new GeneralProperties(showHelp, evaluate, train, parseTaggedFile, parseConllFile, parsePartialConll, modelFile, beamWidth, rootFirst,
                labeled, lowercase, inputFile, outputFile, numOfThreads, punctuations, parserType);
    }
}
