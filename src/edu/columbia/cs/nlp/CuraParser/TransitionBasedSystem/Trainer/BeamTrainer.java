package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Trainer;

import edu.columbia.cs.nlp.CuraParser.Accessories.Options;
import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;

import java.io.FileInputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.zip.GZIPInputStream;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/24/16
 * Time: 10:39 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class BeamTrainer extends GreedyTrainer{

    public BeamTrainer(Options options, ArrayList<Integer> dependencyRelations, int labelNullIndex, HashSet<Integer> rareWords) throws Exception {
        super(options, dependencyRelations, labelNullIndex, rareWords);
        MLPNetwork greedyModel = getGreedyModel(options);
    }

    public static void trainWithNN(Options options) throws Exception {
      // todo
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
