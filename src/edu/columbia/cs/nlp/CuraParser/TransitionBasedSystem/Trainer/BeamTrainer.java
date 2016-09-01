package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Trainer;

import edu.columbia.cs.nlp.CuraParser.Accessories.CoNLLReader;
import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPTrainer;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.AveragingOption;
import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.UpdaterType;
import edu.columbia.cs.nlp.CuraParser.Structures.Pair;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.BeamElement;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Threading.BeamScorerThread;

import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.TreeSet;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.zip.GZIPInputStream;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/24/16
 * Time: 10:39 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class BeamTrainer extends GreedyTrainer {

    public BeamTrainer(Options options, ArrayList<Integer> dependencyRelations, int labelNullIndex, HashSet<Integer> rareWords) throws Exception {
        super(options, dependencyRelations, labelNullIndex, rareWords);
    }

    public static void trainWithNN(Options options) throws Exception {
        if (options.trainingOptions.trainFile.equals("") || options.generalProperties.modelFile.equals("")) {
            Options.showHelp();
        } else {
            MLPNetwork network = getGreedyModel(options);

            BeamTrainer trainer = new BeamTrainer(options, network.getDepLabels(),
                    network.maps.labelNullIndex, network.maps.rareWords);
            trainer.trainGlobal(options, network);
        }
    }

    private static MLPNetwork getGreedyModel(Options options) throws Exception {
        if (options.trainingOptions.preTrainedModelPath.equals("")) {
            System.out.println("Pre-training with Greedy model!");
            GreedyTrainer.trainWithNN(options);
        }

        System.out.println("Loading the trained Greedy model!");
        String m = options.trainingOptions.preTrainedModelPath.equals("") ? options.generalProperties.modelFile : options.trainingOptions
                .preTrainedModelPath;
        FileInputStream fos = new FileInputStream(m);
        GZIPInputStream gz = new GZIPInputStream(fos);
        ObjectInput reader = new ObjectInputStream(gz);
        MLPNetwork mlpNetwork = (MLPNetwork) reader.readObject();
        reader.close();
        return mlpNetwork;
    }

    public void trainGlobal(Options options, MLPNetwork network) throws Exception {
        CoNLLReader reader = new CoNLLReader(options.trainingOptions.trainFile);
        ArrayList<GoldConfiguration> dataSet =
                reader.readData(Integer.MAX_VALUE, false, options.generalProperties.labeled, options.generalProperties.rootFirst,
                        options.generalProperties.lowercase, network.maps);
        System.out.println("CoNLL data reading done!");

        MLPTrainer neuralTrainer = new MLPTrainer(network, options);
        MLPNetwork avgMlpNetwork = new MLPNetwork(network.maps, options, network.getDepLabels(), network.getwDim(), options.networkProperties.posDim,
                options.networkProperties.depDim, options.generalProperties.parserType);

        double bestModelUAS = evaluate(options, network, Double.NEGATIVE_INFINITY);

        ExecutorService executor = Executors.newFixedThreadPool(options.generalProperties.numOfThreads);
        CompletionService<ArrayList<BeamElement>> pool = new ExecutorCompletionService<>(executor);

        ArrayList<Pair<Configuration, ArrayList<Configuration>>> currentBatch = new ArrayList<>(options.networkProperties.beamBatchSize);
        int step = 0;
        for (int iter = 0; iter < options.trainingOptions.beamTrainingIter; iter++) {
            Collections.shuffle(dataSet);
            for (GoldConfiguration goldConfiguration : dataSet) {
                Pair<Configuration, ArrayList<Configuration>> goldAndBeam = getGoldAndBeamElements(goldConfiguration, network, pool);
                currentBatch.add(goldAndBeam);

                if (currentBatch.size() >= options.networkProperties.beamBatchSize) {
                    neuralTrainer.fit(currentBatch, step++, step % (Math.max(1, options.trainingOptions.UASEvalPerStep / 10)) == 0);
                    currentBatch.clear();

                    if (options.updaterProperties.updaterType == UpdaterType.SGD) {
                        if ((step + 1) % options.trainingOptions.decayStep == 0) {
                            neuralTrainer.setLearningRate(0.96 * neuralTrainer.getLearningRate());
                            System.out.println("The new learning rate: " + neuralTrainer.getLearningRate());
                        }
                    }

                    if (options.trainingOptions.averagingOption != AveragingOption.NO) {
                        // averaging
                        double ratio = Math.min(0.9999, (double) step / (9 + step));
                        network.averageNetworks(avgMlpNetwork, 1 - ratio, step == 1 ? 0 : ratio);
                    }

                    if (step % options.trainingOptions.UASEvalPerStep == 0) {
                        if (options.trainingOptions.averagingOption != AveragingOption.ONLY) {
                            bestModelUAS = evaluate(options, network, bestModelUAS);
                        }
                        if (options.trainingOptions.averagingOption != AveragingOption.NO) {
                            avgMlpNetwork.preCompute();
                            bestModelUAS = evaluate(options, avgMlpNetwork, bestModelUAS);
                        }
                    }
                }
            }

        }
    }

    private Pair<Configuration, ArrayList<Configuration>> getGoldAndBeamElements(GoldConfiguration goldConfiguration, MLPNetwork network,
                                                                                 CompletionService<ArrayList<BeamElement>> pool) throws Exception {
        Configuration initialConfiguration = new Configuration(goldConfiguration.getSentence(), options.generalProperties.rootFirst);
        Configuration firstOracle = initialConfiguration.clone();
        ArrayList<Configuration> beam = new ArrayList<>(options.generalProperties.beamWidth);
        beam.add(initialConfiguration);

        Configuration oracle = firstOracle;
        boolean oracleInBeam;
        while (!parser.isTerminal(beam) && beam.size() > 0) {
            // todo think about making it dynamic.
            oracle = parser.staticOracle(goldConfiguration, oracle, dependencyRelations.size());

            TreeSet<BeamElement> beamPreserver = new TreeSet<>();
            for (int b = 0; b < beam.size(); b++) {
                pool.submit(new BeamScorerThread(false, network, beam.get(b),
                        dependencyRelations, b, options.generalProperties.rootFirst, labelNullIndex, parser));
            }
            for (int b = 0; b < beam.size(); b++) {
                for (BeamElement element : pool.take().get()) {
                    beamPreserver.add(element);
                    if (beamPreserver.size() > options.generalProperties.beamWidth)
                        beamPreserver.pollFirst();
                }
            }

            if (beamPreserver.size() == 0 || beam.size() == 0) {
                break;
            } else {
                oracleInBeam = false;

                ArrayList<Configuration> repBeam = new ArrayList<>(options.generalProperties.beamWidth);
                for (BeamElement beamElement : beamPreserver.descendingSet()) {
                    if (repBeam.size() >= options.generalProperties.beamWidth)
                        break;
                    int b = beamElement.number;
                    int action = beamElement.action;
                    int label = beamElement.label;
                    double sc = beamElement.score;

                    Configuration newConfig = beam.get(b).clone();

                    if (action == 0) {
                        parser.shift(newConfig.state);
                        newConfig.addAction(0);
                    } else if (action == 1) {
                        parser.reduce(newConfig.state);
                        newConfig.addAction(1);
                    } else if (action == 2) {
                        parser.rightArc(newConfig.state, label);
                        newConfig.addAction(3 + label);
                    } else if (action == 3) {
                        parser.leftArc(newConfig.state, label);
                        newConfig.addAction(3 + dependencyRelations.size() + label);
                    } else if (action == 4) {
                        parser.unShift(newConfig.state);
                        newConfig.addAction(2);
                    }
                    newConfig.setScore(sc);
                    repBeam.add(newConfig);

                    if (newConfig.equals(oracle))
                        oracleInBeam = true;
                }
                beam = repBeam;

                if (beam.size() > 0) {
                    // do early update
                    if (!oracleInBeam) {
                        // include this because we need it for the gradient matching.
                        beam.add(oracle);
                        break;
                    }
                } else
                    break;
            }
        }

        return new Pair<>(oracle, beam);
    }
}
