/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.YaraParser.Accessories;

import edu.columbia.cs.nlp.YaraParser.Learning.Activation.Enums.ActivationType;
import edu.columbia.cs.nlp.YaraParser.Learning.Props.NetworkProperties;
import edu.columbia.cs.nlp.YaraParser.Learning.Updater.Enums.AveragingOption;
import edu.columbia.cs.nlp.YaraParser.Learning.Updater.Enums.SGDType;
import edu.columbia.cs.nlp.YaraParser.Learning.Updater.Enums.UpdaterType;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Enums.ParserType;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.HashSet;


public class Options implements Serializable {
    // General options
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
    public String devPath;
    public int numOfThreads;
    public HashSet<String> punctuations;

    // Training options
    public int trainingIter;
    public String clusterFile;
    public String wordEmbeddingFile;
    public boolean useMaxViol;
    public boolean useDynamicOracle;
    public boolean useRandomOracleSelection;
    public int UASEvalPerStep;
    public double decayStep;
    public AveragingOption averagingOption;
    public int partialTrainingStartingIteration;
    public int minFreq;

    // Parsing options
    public String scorePath;
    public String separator;
    public ParserType parserType;

    // Network options
    public NetworkProperties networkProperties;

    // Updater options
    public double momentum;
    public SGDType sgdType;
    public double learningRate;
    public UpdaterType updaterType;

    public Options() {
        showHelp = false;
        networkProperties = new NetworkProperties();
        updaterType = UpdaterType.ADAM;
        sgdType = SGDType.NESTEROV;
        train = false;
        parseConllFile = false;
        parseTaggedFile = false;
        beamWidth = 1;

        // good for ADAM.
        learningRate = 0.001;

        decayStep = 0.2;
        momentum = 0.9;
        rootFirst = false;
        modelFile = "";
        outputFile = "";
        inputFile = "";
        devPath = "";
        scorePath = "";
        minFreq = 1;
        averagingOption = AveragingOption.BOTH;
        separator = "_";
        clusterFile = "";
        wordEmbeddingFile = "";
        labeled = true;
        lowercase = false;
        useMaxViol = true;
        useDynamicOracle = true;
        useRandomOracleSelection = false;
        trainingIter = 20000;
        evaluate = false;
        numOfThreads = 8;
        parsePartialConll = false;
        UASEvalPerStep = 100;
        partialTrainingStartingIteration = 3;
        parserType = ParserType.ArcEager;

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

    public static void showHelp() {
        StringBuilder output = new StringBuilder();
        output.append("© Yara YaraParser.Parser \n");
        output.append("\u00a9 Copyright 2014, Yahoo! Inc.\n");
        output.append("\u00a9 Licensed under the terms of the Apache License 2.0. See LICENSE file at the project " +
                "root for terms.");
        output.append("http://www.apache.org/licenses/LICENSE-2.0\n");
        output.append("\n");

        output.append("Usage:\n");

        output.append("* Train a parser:\n");
        output.append("\tjava -jar YaraParser.jar train -train-file [train-file] -dev [dev-file] -model [model-file] " +
                "-punc [punc-file]\n");
        output.append("\t** The model for each iteration is with the pattern [model-file]_iter[iter#]; e.g. " +
                "mode_iter2\n");
        output.append("\t** [punc-file]: File contains list of pos tags for punctuations in the treebank, each in one" +
                " line\n");
        output.append("\t** Other options\n");
        output.append("\t \t -cluster [cluster-file] Brown cluster file: at most 4096 clusters are supported by the " +
                "parser (default: empty)\n\t\t\t the format should be the same as https://github" +
                ".com/percyliang/brown-cluster/blob/master/output.txt \n");
        output.append("\t \t -e [embedding-file] \n");
        output.append("\t \t -avg [both,no,only] (default: both)\n");
        output.append("\t \t -h1 [hidden-layer-size-1] \n");
        output.append("\t \t -h2 [hidden-layer-size-2] \n");
        output.append("\t \t -lr [learning-rate] \n");
        output.append("\t \t -ds [decay-step] \n");
        output.append("\t \t -parser [ae(arc-eager:default), as(arc-standard)] \n");
        output.append("\t \t -a [activation (relu,cubic) -- default:relu] \n");
        output.append("\t \t -u [updater-type: sgd(default),adam,adagrad] \n");
        output.append("\t \t -sgd [sgd-type (if using sgd): nesterov(default),momentum, vanilla] \n");
        output.append("\t \t -batch [batch-size] \n");
        output.append("\t \t -d [dropout-prob (default:0)] \n");
        output.append("\t \t -bias [true/false (use output bias term in softmax layer: default false)] \n");
        output.append("\t \t -momentum [momentum (default:0.9)] \n");
        output.append("\t \t -reg [regularization with L2] \n");
        output.append("\t \t -min [min freq (default 1)] \n");
        output.append("\t \t -wdim [word dim (default 64)] \n");
        output.append("\t \t -posdim [pos dim (default 32)] \n");
        output.append("\t \t -depdim [dep dim (default 32)]  \n");
        output.append("\t \t -eval [uas eval per step (default 100)] \n");
        output.append("\t \t -reg_all [true/false regularize all layers (default=true)] \n");
        output.append("\t \t drop [put if want dropout] \n");
        output.append("\t \t beam:[beam-width] (default:64)\n");
        output.append("\t \t iter:[training-iterations] (default:20)\n");
        output.append("\t \t unlabeled (default: labeled parsing, unless explicitly put `unlabeled')\n");
        output.append("\t \t lowercase (default: case-sensitive words, unless explicitly put 'lowercase')\n");
        output.append("\t \t basic (default: use extended feature set, unless explicitly put 'basic')\n");
        output.append("\t \t early (default: use max violation update, unless explicitly put `early' for early " +
                "update)\n");
        output.append("\t \t static (default: use dynamic oracles, unless explicitly put `static' for static oracles)" +
                "\n");
        output.append("\t \t random (default: choose maximum scoring oracle, unless explicitly put `random' for " +
                "randomly choosing an oracle)\n");
        output.append("\t \t nt:[#_of_threads] (default:8)\n");
        output.append("\t \t pt:[#partail_training_starting_iteration] (default:3; shows the starting iteration for " +
                "considering partial trees)\n");
        output.append("\t \t root_first (default: put ROOT in the last position, unless explicitly put 'root_first')" +
                "\n\n");

        output.append("* Parse a CoNLL'2006 file:\n");
        output.append("\tjava -jar YaraParser.jar parse_conll -input [test-file] -out [output-file] -model " +
                "[model-file] nt:[#_of_threads (optional -- default:8)] \n");
        output.append("\t** The test file should have the conll 2006 format\n");
        output.append("\t** Optional: -score [score file] averaged score of each output parse tree in a file\n\n");

        output.append("* Parse a tagged file:\n");
        output.append("\tjava -jar YaraParser.jar parse_tagged -input [test-file] -out [output-file]  -model " +
                "[model-file] nt:[#_of_threads (optional -- default:8)] \n");
        output.append("\t** The test file should have each sentence in line and word_tag pairs are space-delimited\n");
        output.append("\t** Optional:  -delim [delim] (default is _)\n");
        output.append("\t \t Example: He_PRP is_VBZ nice_AJ ._.\n\n");

        output.append("* Parse a CoNLL'2006 file with partial gold trees:\n");
        output.append("\tjava -jar YaraParser.jar parse_partial -input [test-file] -out [output-file] -model " +
                "[model-file] nt:[#_of_threads (optional -- default:8)] \n");
        output.append("\t** The test file should have the conll 2006 format; each word that does not have a parent, " +
                "should have a -1 parent-index");
        output.append("\t** Optional: -score [score file] averaged score of each output parse tree in a file\n\n");

        output.append("* Evaluate a Conll file:\n");
        output.append("\tjava -jar YaraParser.jar eval -input [gold-file] -out [parsed-file]  -punc [punc-file]\n");
        output.append("\t** [punc-file]: File contains list of pos tags for punctuations in the treebank, each in one" +
                " line\n");
        output.append("\t** Both files should have conll 2006 format\n");
        System.out.println(output.toString());
    }

    public static Options processArgs(String[] args) throws Exception {
        Options options = new Options();

        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--help") || args[i].equals("-h") || args[i].equals("-help"))
                options.showHelp = true;
            else if (args[i].equals("train"))
                options.train = true;
            else if (args[i].equals("parse_conll"))
                options.parseConllFile = true;
            else if (args[i].equals("parse_partial"))
                options.parsePartialConll = true;
            else if (args[i].equals("eval"))
                options.evaluate = true;
            else if (args[i].equals("parse_tagged"))
                options.parseTaggedFile = true;
            else if (args[i].equals("-train-file") || args[i].equals("-input"))
                options.inputFile = args[i + 1];
            else if (args[i].equals("-punc"))
                options.changePunc(args[i + 1]);
            else if (args[i].equals("-model"))
                options.modelFile = args[i + 1];
            else if (args[i].equals("-dev"))
                options.devPath = args[i + 1];
            else if (args[i].equals("-e"))
                options.wordEmbeddingFile = args[i + 1];
            else if (args[i].equals("-bias") && args[i + 1].equals("true"))
                options.networkProperties.outputBiasTerm = true;
            else if (args[i].equals("-reg_all") && args[i + 1].equals("false"))
                options.networkProperties.regualarizeAllLayers = false;
            else if (args[i].equals("-a")) {
                if (args[i + 1].equals("relu"))
                    options.networkProperties.activationType = ActivationType.RELU;
                else if (args[i + 1].equals("cubic"))
                    options.networkProperties.activationType = ActivationType.CUBIC;
                else
                    throw new Exception("updater not supported");
            } else if (args[i].equals("-parser")) {
                if (args[i + 1].equals("ae"))
                    options.parserType = ParserType.ArcEager;
                else if (args[i + 1].equals("as"))
                    options.parserType = ParserType.ArcStandard;
                else
                    throw new Exception("parser not supported");
            } else if (args[i].equals("-sgd")) {
                if (args[i + 1].equals("nesterov"))
                    options.sgdType = SGDType.NESTEROV;
                else if (args[i + 1].equals("momentum"))
                    options.sgdType = SGDType.MOMENTUM;
                else if (args[i + 1].equals("vanilla"))
                    options.sgdType = SGDType.VANILLA;
                else
                    throw new Exception("sgd not supported");
            } else if (args[i].startsWith("-u")) {
                if (args[i + 1].equals("sgd"))
                    options.updaterType = UpdaterType.SGD;
                else if (args[i + 1].equals("adam"))
                    options.updaterType = UpdaterType.ADAM;
                else if (args[i + 1].equals("adagrad"))
                    options.updaterType = UpdaterType.ADAGRAD;
                else if (args[i + 1].equals("adamax"))
                    options.updaterType = UpdaterType.ADAMAX;
                else
                    throw new Exception("updater not supported");
            } else if (args[i].equals("-avg")) {
                if (args[i + 1].equals("both"))
                    options.averagingOption = AveragingOption.BOTH;
                else if (args[i + 1].equals("no"))
                    options.averagingOption = AveragingOption.NO;
                else if (args[i + 1].equals("only"))
                    options.averagingOption = AveragingOption.ONLY;
                else
                    throw new Exception("updater not supported");
            } else if (args[i].equals("-eval"))
                options.UASEvalPerStep = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-h1"))
                options.networkProperties.hiddenLayer1Size = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-h2"))
                options.networkProperties.hiddenLayer2Size = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-batch"))
                options.networkProperties.batchSize = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-posdim"))
                options.networkProperties.posDim = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-depdim"))
                options.networkProperties.depDim = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-wdim"))
                options.networkProperties.wDim = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-min"))
                options.minFreq = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-lr"))
                options.learningRate = Double.parseDouble(args[i + 1]);
            else if (args[i].equals("-ds"))
                options.decayStep = Double.parseDouble(args[i + 1]);
            else if (args[i].equals("-d"))
                options.networkProperties.dropoutProbForHiddenLayer = Double.parseDouble(args[i + 1]);
            else if (args[i].equals("-momentum"))
                options.momentum = Double.parseDouble(args[i + 1]);
            else if (args[i].equals("-reg"))
                options.networkProperties.regularization = Double.parseDouble(args[i + 1]);
            else if (args[i].equals("-cluster"))
                options.clusterFile = args[i + 1];
            else if (args[i].equals("-out"))
                options.outputFile = args[i + 1];
            else if (args[i].equals("-delim"))
                options.separator = args[i + 1];
            else if (args[i].startsWith("beam:"))
                options.beamWidth = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
            else if (args[i].startsWith("nt:"))
                options.numOfThreads = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
            else if (args[i].startsWith("pt:"))
                options.partialTrainingStartingIteration = Integer.parseInt(args[i].substring(args[i].lastIndexOf
                        (":") + 1));
            else if (args[i].equals("unlabeled"))
                options.labeled = Boolean.parseBoolean(args[i]);
            else if (args[i].equals("lowercase"))
                options.lowercase = Boolean.parseBoolean(args[i]);
            else if (args[i].startsWith("-score"))
                options.scorePath = args[i + 1];
            else if (args[i].equals("early"))
                options.useMaxViol = false;
            else if (args[i].equals("static"))
                options.useDynamicOracle = false;
            else if (args[i].equals("random"))
                options.useRandomOracleSelection = true;
            else if (args[i].equals("root_first"))
                options.rootFirst = true;
            else if (args[i].startsWith("iter:"))
                options.trainingIter = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
        }

        if (options.train || options.parseTaggedFile || options.parseConllFile)
            options.showHelp = false;

        return options;
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

    public String toString() {
        if (train) {
            StringBuilder builder = new StringBuilder();
            builder.append("train file: " + inputFile + "\n");
            builder.append("dev file: " + devPath + "\n");
            builder.append("cluster file: " + clusterFile + "\n");
            builder.append("beam width: " + beamWidth + "\n");
            builder.append("rootFirst: " + rootFirst + "\n");
            builder.append("labeled: " + labeled + "\n");
            builder.append("lower-case: " + lowercase + "\n");
            builder.append("updateModel: " + (useMaxViol ? "max violation" : "early") + "\n");
            builder.append("oracle: " + (useDynamicOracle ? "dynamic" : "static") + "\n");
            if (useDynamicOracle)
                builder.append("oracle selection: " + (!useRandomOracleSelection ? "latent max" : "random") + "\n");

            builder.append("training-iterations: " + trainingIter + "\n");
            builder.append("number of threads: " + numOfThreads + "\n");
            builder.append("partial training starting iteration: " + partialTrainingStartingIteration + "\n");
            builder.append(networkProperties.toString());
            builder.append("updater: " + updaterType + "\n");
            builder.append("learning rate: " + learningRate + "\n");
            builder.append("decay step: " + decayStep + "\n");
            builder.append("parser type: " + parserType + "\n");
            return builder.toString();
        } else if (parseConllFile) {
            StringBuilder builder = new StringBuilder();
            builder.append("parse conll" + "\n");
            builder.append("input file: " + inputFile + "\n");
            builder.append("output file: " + outputFile + "\n");
            builder.append("model file: " + modelFile + "\n");
            builder.append("score file: " + scorePath + "\n");
            builder.append("number of threads: " + numOfThreads + "\n");
            return builder.toString();
        } else if (parseTaggedFile) {
            StringBuilder builder = new StringBuilder();
            builder.append("parse  tag file" + "\n");
            builder.append("input file: " + inputFile + "\n");
            builder.append("output file: " + outputFile + "\n");
            builder.append("model file: " + modelFile + "\n");
            builder.append("score file: " + scorePath + "\n");
            builder.append("number of threads: " + numOfThreads + "\n");
            return builder.toString();
        } else if (parsePartialConll) {
            StringBuilder builder = new StringBuilder();
            builder.append("parse partial conll" + "\n");
            builder.append("input file: " + inputFile + "\n");
            builder.append("output file: " + outputFile + "\n");
            builder.append("score file: " + scorePath + "\n");
            builder.append("model file: " + modelFile + "\n");
            builder.append("labeled: " + labeled + "\n");
            builder.append("number of threads: " + numOfThreads + "\n");
            return builder.toString();
        } else if (evaluate) {
            StringBuilder builder = new StringBuilder();
            builder.append("Evaluate" + "\n");
            builder.append("input file: " + inputFile + "\n");
            builder.append("parsed file: " + outputFile + "\n");
            return builder.toString();
        }
        return "";
    }
}
