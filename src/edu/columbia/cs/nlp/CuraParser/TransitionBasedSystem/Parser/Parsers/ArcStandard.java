package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Parsers;

import edu.columbia.cs.nlp.CuraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.CuraParser.Structures.Pair;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.State;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Parser.Enums.Actions;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 8/15/16
 * Time: 10:59 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public class ArcStandard extends ShiftReduceParser {
    public void leftArc(State state, int dependency) throws Exception {
        int first = state.pop();
        int second = state.pop();
        state.addArc(second, first, dependency);
        state.push(first);
    }

    public void rightArc(State state, int dependency) throws Exception {
        int first = state.pop();
        state.addArc(first, state.peek(), dependency);
    }

    public void reduce(State state) throws Exception {
        throw new NotImplementedException();
    }

    public void unShift(State state) throws Exception {
        throw new NotImplementedException();
    }

    public boolean canDo(Actions action, State state) throws Exception {
        if (action == Actions.Shift) { //shift
            return !state.bufferEmpty();
        } else if (action == Actions.RightArc) { //right arc
            if (state.stackSize() < 2)
                return false;
            return state.peek() != state.rootIndex;
        } else if (action == Actions.LeftArc) { //left arc
            if (state.stackSize() < 2)
                return false;
            int first = state.pop();
            boolean canDo = true;
            if (state.peek() == state.rootIndex)
                canDo = false;
            state.push(first);
            return canDo;
        } else if (action == Actions.Reduce || action == Actions.Unshift) { // reduce  or unshift
            return false;
        }
        return false;
    }

    public Configuration staticOracle(GoldConfiguration goldConfiguration, Configuration configuration, int depSize) throws Exception {
        HashMap<Integer, Pair<Integer, Integer>> goldDependencies = goldConfiguration.getGoldDependencies();
        HashMap<Integer, HashSet<Integer>> reversedDependencies = goldConfiguration.getReversedDependencies();
        State state = configuration.state;
        if (!configuration.state.isTerminalState()) {
            Configuration newConfig = configuration.clone();

            if (state.stackSize() < 2) {
                shift(newConfig.state);
                newConfig.addAction(0);
                newConfig.addScore(0);
            } else {
                int first = state.pop();
                int second = state.peek();
                state.push(first);

                if (goldDependencies.containsKey(second) && goldDependencies.get(second).first == first) { // always prefers left-arc
                    int dependency = goldDependencies.get(second).second;
                    leftArc(newConfig.state, dependency);
                    newConfig.addAction(3 + depSize + dependency);
                    newConfig.addScore(0);
                } else if (goldDependencies.containsKey(first) && goldDependencies.get(first).first == second) {
                    boolean gotAllDeps = true;

                    if (reversedDependencies.containsKey(first))
                        for (int dep : reversedDependencies.get(first)) {
                            if (!state.hasHead(dep)) {
                                gotAllDeps = false;
                                break;
                            }
                        }

                    if (gotAllDeps) {
                        int dependency = goldDependencies.get(first).second;
                        rightArc(newConfig.state, dependency);
                        newConfig.addAction(3 + dependency);
                        newConfig.addScore(0);
                    } else {
                        shift(newConfig.state);
                        newConfig.addAction(0);
                        newConfig.addScore(0);
                    }
                } else {
                    shift(newConfig.state);
                    newConfig.addAction(0);
                    newConfig.addScore(0);
                }
            }
            return newConfig;
        }
        return configuration;
    }

    public Configuration zeroCostDynamicOracle(GoldConfiguration goldConfiguration, HashMap<Configuration, Double> oracles, HashMap<Configuration,
            Double> newOracles, MLPNetwork network, int labelNullIndex, ArrayList<Integer> dependencyRelations) throws Exception {
        throw new NotImplementedException();
    }
}