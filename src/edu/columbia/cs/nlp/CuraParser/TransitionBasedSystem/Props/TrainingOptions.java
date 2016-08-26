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
    public int preTrainingIter;
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
    public String preTrainedModelPath;
    public boolean pretrainLayers;

    public TrainingOptions() {
        decayStep = 4400;
        minFreq = 4;
        pretrainLayers = true;
        averagingOption = AveragingOption.ONLY;
        clusterFile = "";
        wordEmbeddingFile = "";
        useMaxViol = true;
        useDynamicOracle = true;
        useRandomOracleSelection = false;
        trainingIter = 20000;
        preTrainingIter = 5000;
        beamTrainingIter = 30000;
        UASEvalPerStep = 500;
        partialTrainingStartingIteration = 3;
        devPath = "";
        trainFile = "";
        considerAllActions = false;
        preTrainedModelPath = "";
    }

    private TrainingOptions(int trainingIter, int beamTrainingIter, String clusterFile, String wordEmbeddingFile, boolean useMaxViol, boolean
            useDynamicOracle, boolean useRandomOracleSelection, int UASEvalPerStep, int decayStep, AveragingOption averagingOption, int
                                    partialTrainingStartingIteration, int minFreq, String devPath, String trainFile, boolean considerAllActions,
                            String preTrainedModelPath, boolean pretrainLayers, int preTrainingIter) {
        this.trainingIter = trainingIter;
        this.beamTrainingIter = beamTrainingIter;
        this.clusterFile = clusterFile;
        this.wordEmbeddingFile = wordEmbeddingFile;
        this.useMaxViol = useMaxViol;
        this.useDynamicOracle = useDynamicOracle;
        this.useRandomOracleSelection = useRandomOracleSelection;
        this.UASEvalPerStep = UASEvalPerStep;
        this.decayStep = decayStep;
        this.averagingOption = averagingOption;
        this.partialTrainingStartingIteration = partialTrainingStartingIteration;
        this.minFreq = minFreq;
        this.devPath = devPath;
        this.trainFile = trainFile;
        this.considerAllActions = considerAllActions;
        this.preTrainedModelPath = preTrainedModelPath;
        this.pretrainLayers = pretrainLayers;
        this.preTrainingIter = preTrainingIter;
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
        builder.append("pre-trained model path: " + preTrainedModelPath + "\n");
        return builder.toString();
    }

    @Override
    public TrainingOptions clone() {
        return new TrainingOptions(trainingIter, beamTrainingIter, clusterFile, wordEmbeddingFile, useMaxViol, useDynamicOracle,
                useRandomOracleSelection, UASEvalPerStep, decayStep, averagingOption, partialTrainingStartingIteration, minFreq, devPath, trainFile,
                considerAllActions, preTrainedModelPath, pretrainLayers, preTrainingIter);
    }

}
