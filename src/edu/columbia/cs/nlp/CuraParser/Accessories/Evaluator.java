/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.CuraParser.Accessories;

import edu.columbia.cs.nlp.CuraParser.Structures.Pair;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.CompactTree;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class Evaluator {
    public static Pair<Double, Double> evaluate(String testPath, String predictedPath, HashSet<String> puncTags) throws
            Exception {
        CoNLLReader goldReader = new CoNLLReader(testPath);
        CoNLLReader predictedReader = new CoNLLReader(predictedPath);

        ArrayList<CompactTree> goldConfiguration = goldReader.readStringData();
        ArrayList<CompactTree> predConfiguration = predictedReader.readStringData();

        double unlabMatch = 0;
        double labMatch = 0;
        int all = 0;

        double fullULabMatch = 0;
        double fullLabMatch = 0;
        int numTree = 0;

        for (int i = 0; i < predConfiguration.size(); i++) {
            HashMap<Integer, Pair<Integer, String>> goldDeps = goldConfiguration.get(i).goldDependencies;
            HashMap<Integer, Pair<Integer, String>> predDeps = predConfiguration.get(i).goldDependencies;

            ArrayList<String> goldTags = goldConfiguration.get(i).posTags;

            numTree++;
            boolean fullMatch = true;
            boolean fullUnlabMatch = true;
            for (int dep : goldDeps.keySet()) {
                if (!puncTags.contains(goldTags.get(dep - 1).trim())) {
                    all++;
                    int gh = goldDeps.get(dep).first;
                    int ph = predDeps.get(dep).first;
                    String gl = goldDeps.get(dep).second.trim();
                    String pl = predDeps.get(dep).second.trim();

                    if (ph == gh) {
                        unlabMatch++;

                        if (pl.equals(gl))
                            labMatch++;
                        else {
                            fullMatch = false;
                        }
                    } else {
                        fullMatch = false;
                        fullUnlabMatch = false;
                    }
                }
            }

            if (fullMatch)
                fullLabMatch++;
            if (fullUnlabMatch)
                fullULabMatch++;
        }

        DecimalFormat format = new DecimalFormat("##.00");
        double labeledAccuracy = 100.0 * labMatch / all;
        double unlabaledAccuracy = 100.0 * unlabMatch / all;
        System.out.println("las: " + format.format(labeledAccuracy));
        System.out.println("uas: " + format.format(unlabaledAccuracy));
        double labExact = 100.0 * fullLabMatch / numTree;
        double ulabExact = 100.0 * fullULabMatch / numTree;
        System.out.println("Labeled exact match:  " + format.format(labExact));
        System.out.println("Unlabeled exact match:  " + format.format(ulabExact) + " \n");

        return new Pair<>(unlabaledAccuracy, labeledAccuracy);
    }
}
