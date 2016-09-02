/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.CuraParser.Accessories;

import edu.columbia.cs.nlp.CuraParser.Learning.Activation.Enums.ActivationType;
import edu.columbia.cs.nlp.CuraParser.Learning.Props.NetworkProperties;
import edu.columbia.cs.nlp.CuraParser.Learning.Props.UpdaterProperties;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.AveragingOption;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.SGDType;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.UpdaterType;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.ParserType;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Props.GeneralProperties;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Props.TrainingOptions;

import java.io.Serializable;

public class Options implements Serializable {
    // General options
    public GeneralProperties generalProperties;

    // Training options
    public TrainingOptions trainingOptions;

    // Parsing options
    public String scorePath;
    public String separator;

    // Network options
    public NetworkProperties networkProperties;

    // Updater options
    public UpdaterProperties updaterProperties;

    public Options() {
        generalProperties = new GeneralProperties();
        networkProperties = new NetworkProperties();
        updaterProperties = new UpdaterProperties();
        trainingOptions = new TrainingOptions();
        scorePath = "";
        separator = "_";
    }

    private Options(GeneralProperties generalProperties, TrainingOptions trainingOptions, String scorePath, String separator, NetworkProperties
            networkProperties, UpdaterProperties updaterProperties) {
        this.generalProperties = generalProperties;
        this.trainingOptions = trainingOptions;
        this.scorePath = scorePath;
        this.separator = separator;
        this.networkProperties = networkProperties;
        this.updaterProperties = updaterProperties;
    }

    public static void showHelp() {
        StringBuilder output = new StringBuilder();
        output.append("© Cura Parser \n");
        output.append("\u00a9 Licensed under the terms of the Apache License 2.0. See LICENSE file at the project " +
                "root for terms.");
        output.append("http://www.apache.org/licenses/LICENSE-2.0\n");
        output.append("\n");

        output.append("Usage:\n");

        output.append("* Train a parser:\n");
        output.append("\tjava -jar CuraParser.jar train -train-file [train-file] -dev [dev-file] -model [model-file]\n");
        output.append("\t** The model for each iteration is with the pattern [model-file]_iter[iter#]; e.g. " +
                "mode_iter2\n");
        output.append("\t** [punc-file]: File contains list of pos tags for punctuations in the treebank, each in one" +
                " line\n");
        output.append("\t** Other options\n");
        output.append("\t \t -punc [punc-file]\n");
        output.append("\t \t -cluster [cluster-file]\n");
        output.append("\t \t -e [embedding-file] \n");
        output.append("\t \t -avg [both,no,only] (default: only)\n");
        output.append("\t \t -h1 [hidden-layer-size-1 (default 256)] \n");
        output.append("\t \t -h2 [hidden-layer-size-2 (default 256)] \n");
        output.append("\t \t -lr [learning-rate: default 0.0005 (good for ADAM)] \n");
        output.append("\t \t -ds [decay-step (default 4400)] \n");
        output.append("\t \t -parser [ae(arc-eager), as(arc-standard:default)] \n");
        output.append("\t \t -pretrained [pre-trained greedy model path (for beam learning)] \n");
        output.append("\t \t -pos_c [true, false] (default: true; replacing pos for unknown words)\n");
        output.append("\t \t -a [activation (relu,cubic,lrelu,rrelu) -- default:relu] \n");
        output.append("\t \t -u [updater-type: sgd,adam(default),adamax,adagrad] \n");
        output.append("\t \t -sgd [sgd-type (if using sgd): nesterov(default),momentum, vanilla] \n");
        output.append("\t \t -batch [batch-size; default 1000] \n");
        output.append("\t \t -beam_batch [beam-batch-size -- num of sentences in a batch (default:8)] \n");
        output.append("\t \t -d [dropout-prob (default:0)] \n");
        output.append("\t \t -bias [true/false (use output bias term in softmax layer: default true)] \n");
        output.append("\t \t -reg [regularization with L2] \n");
        output.append("\t \t -momentum [momentum for sgd; default 0.9] \n");
        output.append("\t \t -min [min freq for not regarding as unknown(default 5)] \n");
        output.append("\t \t -wdim [word dim (default 64)] \n");
        output.append("\t \t -posdim [pos dim (default 32)] \n");
        output.append("\t \t -depdim [dep dim (default 32)]  \n");
        output.append("\t \t -eval [uas eval per step (default 500)] \n");
        output.append("\t \t -reg_all [true/false regularize all layers (default=false)] \n");
        output.append("\t \t drop [put if want dropout] \n");
        output.append("\t \t beam:[beam-width] (default:8)\n");
        output.append("\t \t pre_iter:[pre-training-iterations for first layer in multi-layer] (default:20000)\n");
        output.append("\t \t iter:[training-iterations] (default:30000)\n");
        output.append("\t \t beam_iter:[beam-training-iterations] (default:20000)\n");
        output.append("\t \t consider_all (put want to consider all, even infeasible actions)\n");
        output.append("\t \t unlabeled (default: labeled parsing, unless explicitly put `unlabeled')\n");
        output.append("\t \t -layer_pretrain (true/false default:true)\n");
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
        output.append("\tjava -jar CuraParser.jar parse_conll -input [test-file] -out [output-file] -model " +
                "[model-file] nt:[#_of_threads (optional -- default:8)] \n");
        output.append("\t** The test file should have the conll 2006 format\n");
        output.append("\t** Optional: -score [score file] averaged score of each output parse tree in a file\n\n");

        output.append("* Parse a tagged file:\n");
        output.append("\tjava -jar CuraParser.jar parse_tagged -input [test-file] -out [output-file]  -model " +
                "[model-file] nt:[#_of_threads (optional -- default:8)] \n");
        output.append("\t** The test file should have each sentence in line and word_tag pairs are space-delimited\n");
        output.append("\t** Optional:  -delim [delim] (default is _)\n");
        output.append("\t \t Example: He_PRP is_VBZ nice_AJ ._.\n\n");

        output.append("* Parse a CoNLL'2006 file with partial gold trees:\n");
        output.append("\tjava -jar CuraParser.jar parse_partial -input [test-file] -out [output-file] -model " +
                "[model-file] nt:[#_of_threads (optional -- default:8)] \n");
        output.append("\t** The test file should have the conll 2006 format; each word that does not have a parent, " +
                "should have a -1 parent-index");
        output.append("\t** Optional: -score [score file] averaged score of each output parse tree in a file\n\n");

        output.append("* Evaluate a Conll file:\n");
        output.append("\tjava -jar CuraParser.jar eval -input [gold-file] -out [parsed-file] \n");
        output.append("\t** optional -punc [punc-file]: File contains list of pos tags for punctuations in the treebank, each in one" +
                " line\n");
        output.append("\t** Both files should have conll 2006 format\n");

        output.append("* Output csv for train instances:\n");
        output.append("\tjava -jar CuraParser.jar output -input [gold-file] -out [output file]\n");
        System.out.println(output.toString());
    }

    public static Options processArgs(String[] args) throws Exception {
        Options options = new Options();

        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--help") || args[i].equals("-h") || args[i].equals("-help"))
                options.generalProperties.showHelp = true;
            else if (args[i].equals("train"))
                options.generalProperties.train = true;
            else if (args[i].equals("output"))
                options.generalProperties.output = true;
            else if (args[i].equals("parse_conll"))
                options.generalProperties.parseConllFile = true;
            else if (args[i].equals("parse_partial"))
                options.generalProperties.parsePartialConll = true;
            else if (args[i].equals("eval"))
                options.generalProperties.evaluate = true;
            else if (args[i].equals("parse_tagged"))
                options.generalProperties.parseTaggedFile = true;
            else if (args[i].equals("-train-file"))
                options.trainingOptions.trainFile = args[i + 1];
            else if (args[i].equals("-input"))
                options.generalProperties.inputFile = args[i + 1];
            else if (args[i].equals("-punc"))
                options.changePunc(args[i + 1]);
            else if (args[i].equals("-model"))
                options.generalProperties.modelFile = args[i + 1];
            else if (args[i].equals("-dev"))
                options.trainingOptions.devPath = args[i + 1];
            else if (args[i].equals("-e"))
                options.trainingOptions.wordEmbeddingFile = args[i + 1];
            else if (args[i].equals("-bias") && args[i + 1].equals("false"))
                options.networkProperties.outputBiasTerm = false;
            else if (args[i].equals("-layer_pretrain") && args[i + 1].equals("false"))
                options.trainingOptions.pretrainLayers = false;
            else if (args[i].equals("-reg_all") && args[i + 1].equals("true"))
                options.networkProperties.regualarizeAllLayers = true;
            else if (args[i].equals("-pos_c") && args[i + 1].equals("false"))
                options.generalProperties.includePosAsUnknown = false;
            else if (args[i].equals("-a")) {
                if (args[i + 1].equals("relu"))
                    options.networkProperties.activationType = ActivationType.RELU;
                else if (args[i + 1].equals("rrelu"))
                    options.networkProperties.activationType = ActivationType.RandomRelu;
                else if (args[i + 1].equals("lrelu"))
                    options.networkProperties.activationType = ActivationType.LeakyRELU;
                else if (args[i + 1].equals("cubic"))
                    options.networkProperties.activationType = ActivationType.CUBIC;
                else
                    throw new Exception("updater not supported");
            } else if (args[i].equals("-parser")) {
                if (args[i + 1].equals("ae"))
                    options.generalProperties.parserType = ParserType.ArcEager;
                else if (args[i + 1].equals("as"))
                    options.generalProperties.parserType = ParserType.ArcStandard;
                else
                    throw new Exception("parser not supported");
            } else if (args[i].equals("-sgd")) {
                if (args[i + 1].equals("nesterov"))
                    options.updaterProperties.sgdType = SGDType.NESTEROV;
                else if (args[i + 1].equals("momentum"))
                    options.updaterProperties.sgdType = SGDType.MOMENTUM;
                else if (args[i + 1].equals("vanilla"))
                    options.updaterProperties.sgdType = SGDType.VANILLA;
                else
                    throw new Exception("sgd not supported");
            } else if (args[i].startsWith("-u")) {
                if (args[i + 1].equals("sgd"))
                    options.updaterProperties.updaterType = UpdaterType.SGD;
                else if (args[i + 1].equals("adam"))
                    options.updaterProperties.updaterType = UpdaterType.ADAM;
                else if (args[i + 1].equals("adagrad"))
                    options.updaterProperties.updaterType = UpdaterType.ADAGRAD;
                else if (args[i + 1].equals("adamax"))
                    options.updaterProperties.updaterType = UpdaterType.ADAMAX;
                else
                    throw new Exception("updater not supported");
            } else if (args[i].equals("-avg")) {
                if (args[i + 1].equals("both"))
                    options.trainingOptions.averagingOption = AveragingOption.BOTH;
                else if (args[i + 1].equals("no"))
                    options.trainingOptions.averagingOption = AveragingOption.NO;
                else if (args[i + 1].equals("only"))
                    options.trainingOptions.averagingOption = AveragingOption.ONLY;
                else
                    throw new Exception("updater not supported");
            } else if (args[i].equals("-eval"))
                options.trainingOptions.UASEvalPerStep = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-h1"))
                options.networkProperties.hiddenLayer1Size = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-h2"))
                options.networkProperties.hiddenLayer2Size = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-batch"))
                options.networkProperties.batchSize = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-beam_batch"))
                options.networkProperties.beamBatchSize = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-posdim"))
                options.networkProperties.posDim = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-depdim"))
                options.networkProperties.depDim = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-wdim"))
                options.networkProperties.wDim = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-min"))
                options.trainingOptions.minFreq = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-lr"))
                options.updaterProperties.learningRate = Double.parseDouble(args[i + 1]);
            else if (args[i].equals("-ds"))
                options.trainingOptions.decayStep = Integer.parseInt(args[i + 1]);
            else if (args[i].equals("-d"))
                options.networkProperties.dropoutProbForHiddenLayer = Double.parseDouble(args[i + 1]);
            else if (args[i].equals("-momentum"))
                options.updaterProperties.momentum = Double.parseDouble(args[i + 1]);
            else if (args[i].equals("-pretrained"))
                options.trainingOptions.preTrainedModelPath = args[i + 1];
            else if (args[i].equals("-reg"))
                options.networkProperties.regularization = Double.parseDouble(args[i + 1]);
            else if (args[i].equals("-cluster"))
                options.trainingOptions.clusterFile = args[i + 1];
            else if (args[i].equals("-out"))
                options.generalProperties.outputFile = args[i + 1];
            else if (args[i].equals("-delim"))
                options.separator = args[i + 1];
            else if (args[i].startsWith("beam:"))
                options.generalProperties.beamWidth = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
            else if (args[i].startsWith("beam_iter:"))
                options.trainingOptions.beamTrainingIter = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
            else if (args[i].startsWith("pre_iter:"))
                options.trainingOptions.preTrainingIter = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
            else if (args[i].startsWith("nt:"))
                options.generalProperties.numOfThreads = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
            else if (args[i].startsWith("pt:"))
                options.trainingOptions.partialTrainingStartingIteration = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
            else if (args[i].equals("unlabeled"))
                options.generalProperties.labeled = Boolean.parseBoolean(args[i]);
            else if (args[i].equals("lowercase"))
                options.generalProperties.lowercase = Boolean.parseBoolean(args[i]);
            else if (args[i].startsWith("-score"))
                options.scorePath = args[i + 1];
            else if (args[i].equals("early"))
                options.trainingOptions.useMaxViol = false;
            else if (args[i].equals("static"))
                options.trainingOptions.useDynamicOracle = false;
            else if (args[i].equals("consider_all"))
                options.trainingOptions.considerAllActions = true;
            else if (args[i].equals("random"))
                options.trainingOptions.useRandomOracleSelection = true;
            else if (args[i].equals("root_first"))
                options.generalProperties.rootFirst = true;
            else if (args[i].startsWith("iter:"))
                options.trainingOptions.trainingIter = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
        }

        if (options.generalProperties.train || options.generalProperties.parseTaggedFile || options.generalProperties.parseConllFile)
            options.generalProperties.showHelp = false;

        return options;
    }

    public void changePunc(String puncPath) throws Exception {
        generalProperties.changePunc(puncPath);
    }

    public String toString() {
        if (generalProperties.train) {
            StringBuilder builder = new StringBuilder();
            builder.append(generalProperties.toString());
            builder.append(trainingOptions.toString());
            builder.append(networkProperties.toString());
            builder.append(updaterProperties.toString());
            return builder.toString();
        } else if (generalProperties.parseConllFile) {
            StringBuilder builder = new StringBuilder();
            builder.append(generalProperties.toString());
            builder.append("score file: ").append(scorePath).append("\n");
            return builder.toString();
        } else if (generalProperties.parseTaggedFile) {
            StringBuilder builder = new StringBuilder();
            builder.append(generalProperties.toString());
            builder.append("score file: ").append(scorePath).append("\n");
            return builder.toString();
        } else if (generalProperties.parsePartialConll) {
            StringBuilder builder = new StringBuilder();
            builder.append(generalProperties.toString());
            builder.append("score file: ").append(scorePath).append("\n");
            return builder.toString();
        } else if (generalProperties.evaluate) {
            StringBuilder builder = new StringBuilder();
            builder.append("Evaluate" + "\n");
            builder.append(generalProperties.toString());
            return builder.toString();
        }
        return "";
    }

    @Override
    public Options clone() {
        return new Options(generalProperties.clone(), trainingOptions.clone(), scorePath, separator, networkProperties.clone(),
                updaterProperties.clone());
    }
}
