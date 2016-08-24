package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Props;

import edu.columbia.cs.nlp.CuraParser.Learning.Updater.Enums.AveragingOption;

import java.io.Serializable;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/17/16
 * Time: 11:49 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class TrainingOptions implements Serializable {
    public int trainingIter;
    public int beamTrainingIter;
    public String clusterFile;
    public String wordEmbeddingFile;
    public boolean useMaxViol;
    public boolean useDynamicOracle;
    public boolean useRandomOracleSelection;
    public int UASEvalPerStep;
    public int decayStep;
    public AveragingOption averagingOption;
    public int partialTrainingStartingIteration;
    public int minFreq;
    public String devPath;
    public String trainFile;
    public boolean considerAllActions;

    public TrainingOptions() {
        decayStep = 4400;
        minFreq = 1;
        averagingOption = AveragingOption.ONLY;
        clusterFile = "";
        wordEmbeddingFile = "";
        useMaxViol = true;
        useDynamicOracle = true;
        useRandomOracleSelection = false;
        trainingIter = 5000;
        beamTrainingIter = 0;
        UASEvalPerStep = 100;
        partialTrainingStartingIteration = 3;
        devPath = "";
        trainFile = "";
        considerAllActions = false;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("train file: " + trainFile + "\n");
        builder.append("dev file: " + devPath + "\n");
        builder.append("cluster file: " + clusterFile + "\n");
        if (useDynamicOracle)
            builder.append("oracle selection: " + (!useRandomOracleSelection ? "latent max" : "random") + "\n");
        builder.append("updateModel: " + (useMaxViol ? "max violation" : "early") + "\n");
        builder.append("oracle: " + (useDynamicOracle ? "dynamic" : "static") + "\n");
        builder.append("training-iterations: " + trainingIter + "\n");
        builder.append("beam-training iterations: " + beamTrainingIter + "\n");
        builder.append("partial training starting iteration: " + partialTrainingStartingIteration + "\n");
        builder.append("decay step: " + decayStep + "\n");
        builder.append("consider all actions: " + considerAllActions + "\n");
        return builder.toString();
    }


}
