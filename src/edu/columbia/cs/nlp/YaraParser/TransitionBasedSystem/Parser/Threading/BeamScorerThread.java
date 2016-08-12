/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Threading;

import edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.BeamElement;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.State;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Features.FeatureExtractor;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.ArcEager.Actions;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.ArcEager.ArcEager;

import java.util.ArrayList;
import java.util.concurrent.Callable;


public class BeamScorerThread implements Callable<ArrayList<BeamElement>> {

    boolean isDecode;
    MLPNetwork network;
    Configuration configuration;
    ArrayList<Integer> dependencyRelations;
    int b;
    boolean rootFirst;
    int labelNullIndex;

    public BeamScorerThread(boolean isDecode, MLPNetwork network, Configuration configuration,
                            ArrayList<Integer> dependencyRelations, int b, boolean rootFirst, int labelNullIndex) {
        this.isDecode = isDecode;
        this.network = network;
        this.configuration = configuration;
        this.dependencyRelations = dependencyRelations;
        this.b = b;
        this.rootFirst = rootFirst;
        this.labelNullIndex = labelNullIndex;
    }


    public ArrayList<BeamElement> call() throws Exception {
        ArrayList<BeamElement> elements = new ArrayList<BeamElement>(dependencyRelations.size() * 2 + 3);

        State currentState = configuration.state;
        double prevScore = configuration.score;

        boolean canShift = ArcEager.canDo(Actions.Shift, currentState);
        boolean canReduce = ArcEager.canDo(Actions.Reduce, currentState);
        boolean canRightArc = ArcEager.canDo(Actions.RightArc, currentState);
        boolean canLeftArc = ArcEager.canDo(Actions.LeftArc, currentState);
        int[] labels = new int[network.getSoftmaxLayerDim()];
        if (!canShift) labels[0] = -1;
        if (!canReduce) labels[1] = -1;
        if (!canRightArc)
            for (int i = 0; i < dependencyRelations.size(); i++)
                labels[2 + i] = -1;
        if (!canLeftArc)
            for (int i = 0; i < dependencyRelations.size(); i++)
                labels[dependencyRelations.size() + 2 + i] = -1;
        int[] features = FeatureExtractor.extractBaseFeatures(configuration, labelNullIndex);
        double[] scores = network.output(features, labels);


        if (canShift) {
            double score = scores[0];
            double addedScore = score + prevScore;
            elements.add(new BeamElement(addedScore, b, 0, -1));
        }
        if (canReduce) {
            double score = scores[1];
            double addedScore = score + prevScore;
            elements.add(new BeamElement(addedScore, b, 1, -1));

        }

        if (canRightArc) {
            for (int dependency : dependencyRelations) {
                double score = scores[2 + dependency];
                double addedScore = score + prevScore;
                elements.add(new BeamElement(addedScore, b, 2, dependency));
            }
        }
        if (canLeftArc) {
            for (int dependency : dependencyRelations) {
                double score = scores[2 + dependencyRelations.size() + dependency];
                double addedScore = score + prevScore;
                elements.add(new BeamElement(addedScore, b, 3, dependency));
            }
        }
        return elements;
    }
}