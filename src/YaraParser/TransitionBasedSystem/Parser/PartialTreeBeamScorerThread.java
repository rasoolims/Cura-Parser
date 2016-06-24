/**
 * Copyright 2014, Yahoo! Inc. and Mohammad Sadegh Rasooli
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package YaraParser.TransitionBasedSystem.Parser;

import YaraParser.Learning.AveragedPerceptron;
import YaraParser.TransitionBasedSystem.Configuration.BeamElement;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Configuration.State;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;

import java.util.ArrayList;
import java.util.concurrent.Callable;


public class PartialTreeBeamScorerThread implements Callable<ArrayList<BeamElement>> {

    boolean isDecode;
    AveragedPerceptron classifier;
    Configuration configuration;
    GoldConfiguration goldConfiguration;
    ArrayList<Integer> dependencyRelations;
    int featureLength;
    int b;

    public PartialTreeBeamScorerThread(boolean isDecode, AveragedPerceptron classifier, GoldConfiguration
            goldConfiguration, Configuration configuration, ArrayList<Integer> dependencyRelations, int
                                               featureLength, int b) {
        this.isDecode = isDecode;
        this.classifier = classifier;
        this.configuration = configuration;
        this.goldConfiguration = goldConfiguration;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
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
        Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);

        if (canShift) {
            if (isNonProjective || goldConfiguration.actionCost(Actions.Shift, -1, currentState) == 0) {
                double score = classifier.shiftScore(features, isDecode);
                double addedScore = score + prevScore;
                elements.add(new BeamElement(addedScore, b, 0, -1));
            }
        }
        if (canReduce) {
            if (isNonProjective || goldConfiguration.actionCost(Actions.Reduce, -1, currentState) == 0) {
                double score = classifier.reduceScore(features, isDecode);
                double addedScore = score + prevScore;
                elements.add(new BeamElement(addedScore, b, 1, -1));
            }

        }

        if (canRightArc) {
            double[] rightArcScores = classifier.rightArcScores(features, isDecode);
            for (int dependency : dependencyRelations) {
                if (isNonProjective || goldConfiguration.actionCost(Actions.RightArc, dependency, currentState) == 0) {
                    double score = rightArcScores[dependency];
                    double addedScore = score + prevScore;
                    elements.add(new BeamElement(addedScore, b, 2, dependency));
                }
            }
        }
        if (canLeftArc) {
            double[] leftArcScores = classifier.leftArcScores(features, isDecode);
            for (int dependency : dependencyRelations) {
                if (isNonProjective || goldConfiguration.actionCost(Actions.LeftArc, dependency, currentState) == 0) {
                    double score = leftArcScores[dependency];
                    double addedScore = score + prevScore;
                    elements.add(new BeamElement(addedScore, b, 3, dependency));

                }
            }
        }

        if (elements.size() == 0) {
            if (canShift) {
                double score = classifier.shiftScore(features, isDecode);
                double addedScore = score + prevScore;
                elements.add(new BeamElement(addedScore, b, 0, -1));
            }
            if (canReduce) {
                double score = classifier.reduceScore(features, isDecode);
                double addedScore = score + prevScore;
                elements.add(new BeamElement(addedScore, b, 1, -1));
            }

            if (canRightArc) {
                double[] rightArcScores = classifier.rightArcScores(features, isDecode);
                for (int dependency : dependencyRelations) {
                    double score = rightArcScores[dependency];
                    double addedScore = score + prevScore;
                    elements.add(new BeamElement(addedScore, b, 2, dependency));
                }
            }
            if (canLeftArc) {
                double[] leftArcScores = classifier.leftArcScores(features, isDecode);
                for (int dependency : dependencyRelations) {
                    double score = leftArcScores[dependency];
                    double addedScore = score + prevScore;
                    elements.add(new BeamElement(addedScore, b, 3, dependency));
                }
            }
        }

        return elements;
    }
}