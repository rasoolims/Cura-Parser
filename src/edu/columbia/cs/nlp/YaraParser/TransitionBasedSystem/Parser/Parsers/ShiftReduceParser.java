package edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Parsers;

import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.State;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Enums.Actions;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/15/16
 * Time: 10:16 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public abstract class ShiftReduceParser {
    public void shift(State state) throws Exception {
        state.push(state.bufferHead());
        state.incrementBufferHead();

        // changing the constraint
        if (state.bufferEmpty())
            state.setEmptyFlag(true);
    }

    public void unShift(State state) throws Exception {
        if (!state.stackEmpty())
            state.setBufferH(state.pop());
        // to make sure
        state.setEmptyFlag(true);
        state.setMaxSentenceSize(state.bufferHead());
    }

    public void reduce(State state) throws Exception {
        state.pop();
        if (state.stackEmpty() && state.bufferEmpty())
            state.setEmptyFlag(true);
    }

    public abstract void leftArc(State state, int dependency) throws Exception;

    public abstract void rightArc(State state, int dependency) throws Exception;

    public abstract boolean canDo(Actions action, State state);

    public abstract boolean isTerminal(ArrayList<Configuration> beam);

    public abstract boolean isTerminal(HashMap<Configuration, Double> oracles);
}
