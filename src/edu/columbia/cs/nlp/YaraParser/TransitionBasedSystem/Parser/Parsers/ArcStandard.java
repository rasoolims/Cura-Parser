package edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Parsers;

import edu.columbia.cs.nlp.YaraParser.Learning.NeuralNetwork.MLPNetwork;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Configuration.State;
import edu.columbia.cs.nlp.YaraParser.TransitionBasedSystem.Parser.Enums.Actions;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.ArrayList;
import java.util.HashMap;

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
            if (state.bufferEmpty())
                return false;
            return true;
        } else if (action == Actions.RightArc) { //right arc
            if (state.stackSize() < 2)
                return false;
            if (state.peek() == state.rootIndex)
                return false;

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

    @Override
    public Configuration staticOracle(GoldConfiguration goldConfiguration, HashMap<Configuration, Double> oracles,
                                      HashMap<Configuration, Double> newOracles, int depSize) throws Exception {
        return null;
    }

    @Override
    public Configuration zeroCostDynamicOracle(GoldConfiguration goldConfiguration, HashMap<Configuration, Double> oracles, HashMap<Configuration,
            Double> newOracles, MLPNetwork network, int labelNullIndex, ArrayList<Integer> dependencyRelations) throws Exception {
        throw new NotImplementedException();
    }


}

