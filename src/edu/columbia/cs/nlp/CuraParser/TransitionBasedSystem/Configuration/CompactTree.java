/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration;

import edu.columbia.cs.nlp.CuraParser.Structures.Pair;

import java.util.ArrayList;
import java.util.HashMap;

public class CompactTree {
    public HashMap<Integer, Pair<Integer, String>> goldDependencies;
    public ArrayList<String> posTags;

    public CompactTree(HashMap<Integer, Pair<Integer, String>> goldDependencies, ArrayList<String> posTags) {
        this.goldDependencies = goldDependencies;
        this.posTags = posTags;
    }
}
