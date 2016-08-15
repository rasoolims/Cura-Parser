/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms.
 */

package edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.ArcEager;

import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.State;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Enums.Actions;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.ShiftReduceParser;

import java.util.ArrayList;
import java.util.HashMap;

public class ArcEager extends ShiftReduceParser {
    public void leftArc(State state, int dependency) throws Exception {
        state.addArc(state.pop(), state.bufferHead(), dependency);
    }

    public void rightArc(State state, int dependency) throws Exception {
        state.addArc(state.bufferHead(), state.peek(), dependency);
        state.push(state.bufferHead());
        state.incrementBufferHead();
        if (!state.isEmptyFlag() && state.bufferEmpty())
            state.setEmptyFlag(true);
    }

    public boolean canDo(Actions action, State state) {
        if (action == Actions.Shift) { //shift
            return !(!state.bufferEmpty() && state.bufferHead() == state.rootIndex && !state.stackEmpty()) && !state
                    .bufferEmpty() && !state.isEmptyFlag();
        } else if (action == Actions.RightArc) { //right arc
            if (state.stackEmpty())
                return false;
            return !(!state.bufferEmpty() && state.bufferHead() == state.rootIndex) && !state.bufferEmpty() && !state
                    .stackEmpty();

        } else if (action == Actions.LeftArc) { //left arc
            if (state.stackEmpty() || state.bufferEmpty())
                return false;

            if (!state.stackEmpty() && state.peek() == state.rootIndex)
                return false;

            return state.peek() != state.rootIndex && !state.hasHead(state.peek()) && !state.stackEmpty();
        } else if (action == Actions.Reduce) { //reduce
            return !state.stackEmpty() && state.hasHead(state.peek()) || !state.stackEmpty() && state.stackSize() ==
                    1 && state.bufferSize() == 0 && state.peek() == state.rootIndex;
        } else if (action == Actions.Unshift) { //unshift
            return !state.stackEmpty() && !state.hasHead(state.peek()) && state.isEmptyFlag();
        }
        return false;
    }

    /**
     * Shows true if all of the configurations in the beam are in the terminal state
     *
     * @param beam the current beam
     * @return true if all of the configurations in the beam are in the terminal state
     */
    public boolean isTerminal(ArrayList<Configuration> beam) {
        for (Configuration configuration : beam)
            if (!configuration.state.isTerminalState())
                return false;
        return true;
    }

    public boolean isTerminal(HashMap<Configuration, Double> oracles) {
        for (Configuration configuration : oracles.keySet())
            if (!configuration.state.isTerminalState())
                return false;
        return true;
    }
}
