/**
 * Copyright 2014, Yahoo! Inc. and Mohammad Sadegh Rasooli
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package YaraParser.TransitionBasedSystem.Parser.Threading;

import YaraParser.Learning.NeuralNetwork.MLPNetwork;
import YaraParser.TransitionBasedSystem.Configuration.BeamElement;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Configuration.State;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;
import YaraParser.TransitionBasedSystem.Parser.ArcEager.Actions;
import YaraParser.TransitionBasedSystem.Parser.ArcEager.ArcEager;

import java.util.ArrayList;
import java.util.concurrent.Callable;


public class PartialTreeBeamScorerThread implements Callable<ArrayList<BeamElement>> {

    boolean isDecode;
    MLPNetwork network;
    Configuration configuration;
    GoldConfiguration goldConfiguration;
    ArrayList<Integer> dependencyRelations;
    int b;

    public PartialTreeBeamScorerThread(boolean isDecode, MLPNetwork network, GoldConfiguration
            goldConfiguration, Configuration configuration, ArrayList<Integer> dependencyRelations, int b) {
        this.isDecode = isDecode;
        this.network = network;
        this.configuration = configuration;
        this.goldConfiguration = goldConfiguration;
        this.dependencyRelations = dependencyRelations;
        this.b = b;
    }


    public ArrayList<BeamElement> call() throws Exception {
        ArrayList<BeamElement> elements = new ArrayList<BeamElement>(dependencyRelations.size() * 2 + 3);

        boolean isNonProjective = false;
        if (goldConfiguration.isNonprojective()) {
            isNonProjective = true;
        }

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
        int[] features = FeatureExtractor.extractBaseFeatures(configuration);
        double[] scores = network.output(features, labels);

        if (canShift) {
            if (isNonProjective || goldConfiguration.actionCost(Actions.Shift, -1, currentState) == 0) {
                double score = scores[0];
                double addedScore = score + prevScore;
                elements.add(new BeamElement(addedScore, b, 0, -1));
            }
        }
        if (canReduce) {
            if (isNonProjective || goldConfiguration.actionCost(Actions.Reduce, -1, currentState) == 0) {
                double score = scores[1];
                double addedScore = score + prevScore;
                elements.add(new BeamElement(addedScore, b, 1, -1));
            }

        }

        if (canRightArc) {
            for (int dependency : dependencyRelations) {
                if (isNonProjective || goldConfiguration.actionCost(Actions.RightArc, dependency, currentState) == 0) {
                    double score = scores[2 + dependency];
                    double addedScore = score + prevScore;
                    elements.add(new BeamElement(addedScore, b, 2, dependency));
                }
            }
        }
        if (canLeftArc) {
            for (int dependency : dependencyRelations) {
                if (isNonProjective || goldConfiguration.actionCost(Actions.LeftArc, dependency, currentState) == 0) {
                    double score = scores[2 + dependencyRelations.size() + dependency];
                    double addedScore = score + prevScore;
                    elements.add(new BeamElement(addedScore, b, 3, dependency));

                }
            }
        }

        if (elements.size() == 0) {
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
        }

        return elements;
    }
}