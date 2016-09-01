/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration;

import edu.columbia.cs.nlp.CuraParser.Structures.Sentence;

import java.io.Serializable;
import java.util.ArrayList;

public class Configuration implements Comparable, Cloneable, Serializable {
    public Sentence sentence;
    public State state;
    public ArrayList<Integer> actionHistory;
    public double score;

    public Configuration(Sentence sentence, boolean rootFirst) {
        this.sentence = sentence;
        state = new State(sentence.size(), rootFirst);
        score = 0.0;
        actionHistory = new ArrayList<>(2 * (sentence.size() + 1));
    }

    public Configuration(Sentence sentence) {
        this.sentence = sentence;
        state = new State(sentence.size());
        score = 0.0;
        actionHistory = new ArrayList<>(2 * (sentence.size() + 1));
    }

    /**
     * Returns the current score of the configuration
     *
     * @param normalized if true, the score will be normalized by the number of done actions
     * @return
     */
    public double getScore(boolean normalized) {
        if (normalized && actionHistory.size() > 0)
            return score / actionHistory.size();
        return score;
    }

    public void addScore(double score) {
        this.score += score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    public void addAction(int action) {
        actionHistory.add(action);
    }

    @Override
    public int compareTo(Object o) {
        if (!(o instanceof Configuration))
            return hashCode() - o.hashCode();

        // may be unsafe
        Configuration configuration = (Configuration) o;
        double diff = getScore(true) - configuration.getScore(true);

        if (diff > 0)
            return (int) Math.ceil(diff);
        else if (diff < 0)
            return (int) Math.floor(diff);
        else
            return 0;
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof Configuration) {
            Configuration configuration = (Configuration) o;
            if (configuration.actionHistory.size() != actionHistory.size())
                return false;
            for (int i = 0; i < actionHistory.size(); i++)
                if (!actionHistory.get(i).equals(configuration.actionHistory.get(i)))
                    return false;
            return true;
        }
        return false;
    }

    @Override
    public Configuration clone() {
        Configuration configuration = new Configuration(sentence);
        ArrayList<Integer> history = new ArrayList<>(actionHistory.size());
        for (int i = 0; i < actionHistory.size(); i++)
            history.add(actionHistory.get(i));
        configuration.actionHistory = history;
        configuration.score = score;
        configuration.state = state.clone();
        return configuration;
    }

    @Override
    public int hashCode() {
        int hashCode = 0;
        int i = 0;
        for (int action : actionHistory)
            hashCode += action << i++;
        hashCode += score;
        return hashCode;
    }
}
