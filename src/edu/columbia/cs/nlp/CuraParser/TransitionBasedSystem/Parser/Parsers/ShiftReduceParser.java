package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers;

import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.State;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.Actions;

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
    public void shift(State state) {
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

    public abstract boolean canDo(Actions action, State state) throws Exception;

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

    public abstract Configuration staticOracle(GoldConfiguration goldConfiguration, Configuration configuration, int depSize) throws Exception;

    public abstract Configuration zeroCostDynamicOracle(GoldConfiguration goldConfiguration, HashMap<Configuration, Double>
            oracles, HashMap<Configuration, Double> newOracles, MLPNetwork network, int labelNullIndex, ArrayList<Integer> dependencyRelations)
            throws Exception;

    public void advance(Configuration configuration, int action, int depSize) throws Exception {
        if (action == 0) {
            shift(configuration.state);
        } else if (action == 1) {
            reduce(configuration.state);
        } else if (action == 2) {
            unShift(configuration.state);
        } else if (action >= (3 + depSize)) {
            int dependency = action - (3 + depSize);
            leftArc(configuration.state, dependency);
        } else if (action >= 3) {
            int dependency = action - 3;
            rightArc(configuration.state, dependency);
        }
    }
}
