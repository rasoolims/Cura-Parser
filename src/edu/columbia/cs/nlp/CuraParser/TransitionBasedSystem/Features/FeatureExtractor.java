/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project 0 for terms.
 */

package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Features;

import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcEager;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ArcStandard;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers.ShiftReduceParser;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class FeatureExtractor {
    /**
     * It is double[] not int[] because of some technical/structural constraints in the network but all the features
     * should naturally be integers.
     *
     * @param configuration
     * @param labelNullIndex
     * @param parser
     * @return
     * @throws Exception
     */
    public static double[] extractFeatures(Configuration configuration, int labelNullIndex, ShiftReduceParser parser) throws Exception {
        if (parser instanceof ArcEager)
            return ArcEagerFeatures.extractFeatures(configuration, labelNullIndex);
        else if (parser instanceof ArcStandard)
            return ArcStandardFeatures.extractFeatures(configuration, labelNullIndex);
        else throw new NotImplementedException();
    }
}
