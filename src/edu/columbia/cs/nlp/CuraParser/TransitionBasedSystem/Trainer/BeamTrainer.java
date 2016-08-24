package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Trainer;

import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Structures.Pair;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.BeamElement;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Threading.BeamScorerThread;

import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeSet;
import java.util.concurrent.CompletionService;
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
        MLPNetwork greedyModel = getGreedyModel(options);
    }

    public static void trainWithNN(Options options) throws Exception {
        // todo
    }


    private Pair<Configuration, ArrayList<Configuration>> getGoldAndBeamElements(GoldConfiguration goldConfiguration, MLPNetwork network,
                                                                                 CompletionService<ArrayList<BeamElement>> pool) throws Exception {
        Configuration initialConfiguration = new Configuration(goldConfiguration.getSentence(), options.generalProperties.rootFirst);
        Configuration firstOracle = initialConfiguration.clone();
        ArrayList<Configuration> beam = new ArrayList<>(options.generalProperties.beamWidth);
        beam.add(initialConfiguration);
        HashMap<Configuration, Double> oracles = new HashMap<>();
        oracles.put(firstOracle, 0.0);

        Configuration bestScoringOracle = null;
        boolean oracleInBeam;
        while (!parser.isTerminal(beam) && beam.size() > 0) {
            HashMap<Configuration, Double> newOracles = new HashMap<>();
            // todo think about making it dynamic.
            bestScoringOracle = parser.staticOracle(goldConfiguration, oracles, newOracles, dependencyRelations.size());
            oracles = newOracles;

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

                    if (oracles.containsKey(newConfig))
                        oracleInBeam = true;
                }
                beam = repBeam;

                if (beam.size() > 0 && oracles.size() > 0) {
                    Configuration bestConfig = beam.get(0);
                    if (oracles.containsKey(bestConfig)) {
                        oracles = new HashMap<>();
                        oracles.put(bestConfig, 0.0);
                    } else {
                        oracles.put(bestScoringOracle, 0.0);
                    }

                    // do early update
                    if (!oracleInBeam)
                        break;
                } else
                    break;
            }
        }

        return new Pair<>(bestScoringOracle, beam);
    }


    private MLPNetwork getGreedyModel(Options options) throws Exception {
        GreedyTrainer.trainWithNN(options);

        FileInputStream fos = new FileInputStream(options.generalProperties.modelFile);
        GZIPInputStream gz = new GZIPInputStream(fos);
        ObjectInput reader = new ObjectInputStream(gz);
        MLPNetwork mlpNetwork = (MLPNetwork) reader.readObject();
        reader.close();
        return mlpNetwork;
    }
}
