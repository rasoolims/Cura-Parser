/**
 * Copyright 2014-2016, Mohammad Sadegh Rasooli
 * Parts of this code is extracted from the Yara parser.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project 0 for terms.
 */

package edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Features;

import edu.columbia.cs.nlp.CuraParser.Structures.IndexMaps;
import edu.columbia.cs.nlp.CuraParser.Structures.Sentence;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.Configuration;
import edu.columbia.cs.nlp.CuraParser.TransitionBasedSystem.Configuration.State;
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
    public static double[] extractBaseFeatures(Configuration configuration, int labelNullIndex, ShiftReduceParser parser) throws Exception {
        if (parser instanceof ArcEager)
            return extractArcEagerFeatures(configuration, labelNullIndex);
        else if (parser instanceof ArcStandard)
            return extractArcStandardFeatures(configuration, labelNullIndex);
        else throw new NotImplementedException();
    }

    private static double[] extractArcEagerFeatures(Configuration configuration, int labelNullIndex) throws Exception {
        State state = configuration.state;
        Sentence sentence = configuration.sentence;

        int b0w = IndexMaps.NullIndex;
        int b1w = IndexMaps.NullIndex;
        int b2w = IndexMaps.NullIndex;
        int b3w = IndexMaps.NullIndex;
        int s0w = IndexMaps.NullIndex;
        int s1w = IndexMaps.NullIndex;
        int s2w = IndexMaps.NullIndex;
        int s3w = IndexMaps.NullIndex;
        int b0l1w = IndexMaps.NullIndex;
        int b0l2w = IndexMaps.NullIndex;
        int b0llw = IndexMaps.NullIndex;
        int sr1w = IndexMaps.NullIndex;
        int s0rrw = IndexMaps.NullIndex;
        int sh0w = IndexMaps.NullIndex;
        int s0l1w = IndexMaps.NullIndex;
        int s0llw = IndexMaps.NullIndex;
        int s0r2w = IndexMaps.NullIndex;
        int sh1w = IndexMaps.NullIndex;
        int s0l2w = IndexMaps.NullIndex;
        int b0bw = IndexMaps.NullIndex;
        int s0bw = IndexMaps.NullIndex;
        int s0aw = IndexMaps.NullIndex;

        int b0p = IndexMaps.NullIndex;
        int b1p = IndexMaps.NullIndex;
        int b2p = IndexMaps.NullIndex;
        int b3p = IndexMaps.NullIndex;
        int s0p = IndexMaps.NullIndex;
        int s1p = IndexMaps.NullIndex;
        int s2p = IndexMaps.NullIndex;
        int s3p = IndexMaps.NullIndex;
        int b0l1p = IndexMaps.NullIndex;
        int b0l2p = IndexMaps.NullIndex;
        int b0llp = IndexMaps.NullIndex;
        int sr1p = IndexMaps.NullIndex;
        int s0rrp = IndexMaps.NullIndex;
        int sh0p = IndexMaps.NullIndex;
        int s0l1p = IndexMaps.NullIndex;
        int s0llp = IndexMaps.NullIndex;
        int s0r2p = IndexMaps.NullIndex;
        int sh1p = IndexMaps.NullIndex;
        int s0l2p = IndexMaps.NullIndex;
        int b0bp = IndexMaps.NullIndex;
        int s0bp = IndexMaps.NullIndex;
        int s0ap = IndexMaps.NullIndex;

        int s0l = labelNullIndex;
        int b0l1l = labelNullIndex;
        int b0l2l = labelNullIndex;
        int b0lll = labelNullIndex;
        int sr1l = labelNullIndex;
        int s0rrl = labelNullIndex;
        int sh0l = labelNullIndex;
        int s0l1l = labelNullIndex;
        int s0lll = labelNullIndex;
        int s0r2l = labelNullIndex;
        int s0l2l = labelNullIndex;

        int[] words = sentence.getWords();
        int[] tags = sentence.getTags();

        if (0 < state.bufferSize()) {
            int b0Position = state.bufferHead();
            b0w = (words[b0Position - 1]);
            b0p = (tags[b0Position - 1]);

            if (b0Position > 1) {
                b0bw = words[b0Position - 2];
                b0bp = tags[b0Position - 2];
            }

            int leftMost = state.leftMostModifier(b0Position);
            if (leftMost >= 0) {
                b0l1w = (words[leftMost - 1]);
                b0l1p = (tags[leftMost - 1]);
                b0l1l = (state.getDependency(leftMost, labelNullIndex));

                int l2 = state.secondLeftMostModifier(b0Position);
                if (l2 >= 0) {
                    b0l2w = (words[l2 - 1]);
                    b0l2p = (tags[l2 - 1]);
                    b0l2l = (state.getDependency(l2, labelNullIndex));
                }

                int secondLeftMost = state.leftMostModifier(leftMost);
                if (secondLeftMost >= 0) {
                    b0llw = (words[secondLeftMost - 1]);
                    b0llp = (tags[secondLeftMost - 1]);
                    b0lll = (state.getDependency(secondLeftMost, labelNullIndex));
                }
            }

            if (1 < state.bufferSize()) {
                int b1Position = state.getBufferItem(1);
                b1w = (words[b1Position - 1]);
                b1p = (tags[b1Position - 1]);
                if (2 < state.bufferSize()) {
                    int b2Position = state.getBufferItem(2);
                    b2w = (words[b2Position - 1]);
                    b2p = (tags[b2Position - 1]);
                    if (3 < state.bufferSize()) {
                        int b3Position = state.getBufferItem(3);
                        b3w = (words[b3Position - 1]);
                        b3p = (tags[b3Position - 1]);
                    }
                }
            }
        }

        if (0 < state.stackSize()) {
            int s0Position = state.peek();
            s0w = s0Position > 0 ? words[s0Position - 1] : IndexMaps.RootIndex;
            s0p = s0Position > 0 ? tags[s0Position - 1] : IndexMaps.RootIndex;
            s0l = (state.getDependency(s0Position, labelNullIndex));

            if (s0Position > 1) {
                s0bw = words[s0Position - 2];
                s0bp = tags[s0Position - 2];
            }

            if (s0Position < words.length) {
                s0aw = words[s0Position];
                s0ap = tags[s0Position];
            }

            if (1 < state.stackSize()) {
                int top1 = state.pop();
                int s1Position = state.peek();
                s1w = s1Position > 0 ? words[s1Position - 1] : IndexMaps.RootIndex;
                s1p = s1Position > 0 ? tags[s1Position - 1] : IndexMaps.RootIndex;

                if (1 < state.stackSize()) {
                    int top2 = state.pop();
                    int s2Position = state.peek();
                    s2w = s2Position > 0 ? words[s2Position - 1] : IndexMaps.RootIndex;
                    s2p = s2Position > 0 ? tags[s2Position - 1] : IndexMaps.RootIndex;

                    if (1 < state.stackSize()) {
                        int top3 = state.pop();
                        int s3Position = state.peek();
                        s3w = s3Position > 0 ? words[s3Position - 1] : IndexMaps.RootIndex;
                        s3p = s3Position > 0 ? tags[s3Position - 1] : IndexMaps.RootIndex;
                        state.push(top3);
                    }
                    state.push(top2);
                }
                state.push(top1);
            }

            int leftMost = state.leftMostModifier(s0Position);
            if (leftMost >= 0) {
                s0l1p = (tags[leftMost - 1]);
                s0l1w = (words[leftMost - 1]);
                s0l1l = (state.getDependency(leftMost, labelNullIndex));

                int secondLeftMost = state.leftMostModifier(leftMost);
                if (secondLeftMost >= 0) {
                    s0llp = (tags[secondLeftMost - 1]);
                    s0llw = (words[secondLeftMost - 1]);
                    s0lll = (state.getDependency(secondLeftMost, labelNullIndex));
                }
            }

            int rightMost = state.rightMostModifier(s0Position);
            if (rightMost >= 0) {
                sr1p = (tags[rightMost - 1]);
                sr1w = (words[rightMost - 1]);
                sr1l = (state.getDependency(rightMost, labelNullIndex));

                int secondRightMost = state.rightMostModifier(rightMost);
                if (secondRightMost >= 0) {
                    s0rrp = (tags[secondRightMost - 1]);
                    s0rrw = (words[secondRightMost - 1]);
                    s0rrl = (state.getDependency(secondRightMost, labelNullIndex));
                }
            }

            int headIndex = state.getHead(s0Position);
            if (headIndex >= 0) {
                sh0w = (words[headIndex - 1]);
                sh0p = (tags[headIndex - 1]);
                sh0l = (state.getDependency(headIndex, labelNullIndex));
            }

            if (leftMost >= 0) {
                int l2 = state.secondLeftMostModifier(s0Position);
                if (l2 >= 0) {
                    s0l2w = (words[l2 - 1]);
                    s0l2p = (tags[l2 - 1]);
                    s0l2l = (state.getDependency(l2, labelNullIndex));
                }
            }
            if (headIndex >= 0) {
                if (state.hasHead(headIndex)) {
                    int h2 = state.getHead(headIndex);
                    sh1w = (words[h2 - 1]);
                    sh1p = (tags[h2 - 1]);
                }
            }
            if (rightMost >= 0) {
                int r2 = state.secondRightMostModifier(s0Position);
                if (r2 >= 0) {
                    s0r2w = (words[r2 - 1]);
                    s0r2p = (tags[r2 - 1]);
                    s0r2l = (state.getDependency(r2, labelNullIndex));
                }
            }
        }
        double[] baseFeatureIds = new double[55];

        int index = 0;
        baseFeatureIds[index++] = s0w;
        baseFeatureIds[index++] = s1w;
        baseFeatureIds[index++] = s2w;
        baseFeatureIds[index++] = s3w;
        baseFeatureIds[index++] = b0w;
        baseFeatureIds[index++] = b1w;
        baseFeatureIds[index++] = b2w;
        baseFeatureIds[index++] = b3w;
        baseFeatureIds[index++] = b0l1w;
        baseFeatureIds[index++] = b0l2w;
        baseFeatureIds[index++] = s0l1w;
        baseFeatureIds[index++] = s0l2w;
        baseFeatureIds[index++] = sr1w;
        baseFeatureIds[index++] = s0r2w;
        baseFeatureIds[index++] = sh0w;
        baseFeatureIds[index++] = sh1w;
        baseFeatureIds[index++] = b0llw;
        baseFeatureIds[index++] = s0llw;
        baseFeatureIds[index++] = s0rrw;
        baseFeatureIds[index++] = b0bw;
        baseFeatureIds[index++] = s0bw;
        baseFeatureIds[index++] = s0aw;

        baseFeatureIds[index++] = s0p;
        baseFeatureIds[index++] = s1p;
        baseFeatureIds[index++] = s2p;
        baseFeatureIds[index++] = s3p;
        baseFeatureIds[index++] = b0p;
        baseFeatureIds[index++] = b1p;
        baseFeatureIds[index++] = b2p;
        baseFeatureIds[index++] = b3p;
        baseFeatureIds[index++] = b0l1p;
        baseFeatureIds[index++] = b0l2p;
        baseFeatureIds[index++] = s0l1p;
        baseFeatureIds[index++] = s0l2p;
        baseFeatureIds[index++] = sr1p;
        baseFeatureIds[index++] = s0r2p;
        baseFeatureIds[index++] = sh0p;
        baseFeatureIds[index++] = sh1p;
        baseFeatureIds[index++] = b0llp;
        baseFeatureIds[index++] = s0llp;
        baseFeatureIds[index++] = s0rrp;
        baseFeatureIds[index++] = b0bp;
        baseFeatureIds[index++] = s0bp;
        baseFeatureIds[index++] = s0ap;

        baseFeatureIds[index++] = s0l;
        baseFeatureIds[index++] = sh0l;
        baseFeatureIds[index++] = s0l1l;
        baseFeatureIds[index++] = sr1l;
        baseFeatureIds[index++] = s0l2l;
        baseFeatureIds[index++] = s0r2l;
        baseFeatureIds[index++] = b0l1l;
        baseFeatureIds[index++] = b0l2l;
        baseFeatureIds[index++] = b0lll;
        baseFeatureIds[index++] = s0lll;
        baseFeatureIds[index++] = s0rrl;

        return baseFeatureIds;
    }

    private static double[] extractArcStandardFeatures(Configuration configuration, int labelNullIndex) throws Exception {
        State state = configuration.state;
        Sentence sentence = configuration.sentence;

        int b0w = IndexMaps.NullIndex;
        int b1w = IndexMaps.NullIndex;
        int b2w = IndexMaps.NullIndex;
        int b3w = IndexMaps.NullIndex;
        int s0w = IndexMaps.NullIndex;
        int s1w = IndexMaps.NullIndex;
        int s2w = IndexMaps.NullIndex;
        int s3w = IndexMaps.NullIndex;
        int s0lc1w = IndexMaps.NullIndex;
        int s0rc1w = IndexMaps.NullIndex;
        int s0lc2w = IndexMaps.NullIndex;
        int s0rc2w = IndexMaps.NullIndex;
        int s1lc1w = IndexMaps.NullIndex;
        int s1rc1w = IndexMaps.NullIndex;
        int s1lc2w = IndexMaps.NullIndex;
        int s1rc2w = IndexMaps.NullIndex;
        int s0lc11w = IndexMaps.NullIndex;
        int s0rc11w = IndexMaps.NullIndex;
        int s1lc11w = IndexMaps.NullIndex;
        int s1rc11w = IndexMaps.NullIndex;

        int b0p = IndexMaps.NullIndex;
        int b1p = IndexMaps.NullIndex;
        int b2p = IndexMaps.NullIndex;
        int b3p = IndexMaps.NullIndex;
        int s0p = IndexMaps.NullIndex;
        int s1p = IndexMaps.NullIndex;
        int s2p = IndexMaps.NullIndex;
        int s3p = IndexMaps.NullIndex;
        int s0lc1p = IndexMaps.NullIndex;
        int s0rc1p = IndexMaps.NullIndex;
        int s0lc2p = IndexMaps.NullIndex;
        int s0rc2p = IndexMaps.NullIndex;
        int s1lc1p = IndexMaps.NullIndex;
        int s1rc1p = IndexMaps.NullIndex;
        int s1lc2p = IndexMaps.NullIndex;
        int s1rc2p = IndexMaps.NullIndex;
        int s0lc11p = IndexMaps.NullIndex;
        int s0rc11p = IndexMaps.NullIndex;
        int s1lc11p = IndexMaps.NullIndex;
        int s1rc11p = IndexMaps.NullIndex;

        int s0lc1l = labelNullIndex;
        int s0rc1l = labelNullIndex;
        int s0lc2l = labelNullIndex;
        int s0rc2l = labelNullIndex;
        int s1lc1l = labelNullIndex;
        int s1rc1l = labelNullIndex;
        int s1lc2l = labelNullIndex;
        int s1rc2l = labelNullIndex;
        int s0lc11l = labelNullIndex;
        int s0rc11l = labelNullIndex;
        int s1lc11l = labelNullIndex;
        int s1rc11l = labelNullIndex;

        int[] words = sentence.getWords();
        int[] tags = sentence.getTags();

        if (0 < state.bufferSize()) {
            int b0Position = state.bufferHead();
            b0w = words[b0Position - 1];
            b0p = tags[b0Position - 1];

            if (1 < state.bufferSize()) {
                int b1Position = state.getBufferItem(1);
                b1w = words[b1Position - 1];
                b1p = tags[b1Position - 1];
                if (2 < state.bufferSize()) {
                    int b2Position = state.getBufferItem(2);
                    b2w = words[b2Position - 1];
                    b2p = tags[b2Position - 1];
                    if (3 < state.bufferSize()) {
                        int b3Position = state.getBufferItem(3);
                        b3w = words[b3Position - 1];
                        b3p = tags[b3Position - 1];
                    }
                }
            }
        }

        if (0 < state.stackSize()) {
            if (1 < state.stackSize()) {
                int top1 = state.pop();
                int s1Position = state.peek();
                s1w = s1Position > 0 ? words[s1Position - 1] : IndexMaps.RootIndex;
                s1p = s1Position > 0 ? tags[s1Position - 1] : IndexMaps.RootIndex;

                int leftMost = state.leftMostModifier(s1Position);
                if (leftMost >= 0) {
                    s1lc1p = tags[leftMost - 1];
                    s1lc1w = words[leftMost - 1];
                    s1lc1l = state.getDependency(leftMost, labelNullIndex);

                    int l2 = state.leftMostModifier(leftMost);
                    if (l2 >= 0) {
                        s1lc11p = tags[l2 - 1];
                        s1lc11w = words[l2 - 1];
                        s1lc11l = state.getDependency(l2, labelNullIndex);
                    }

                    int sml = state.secondLeftMostModifier(s1Position);
                    if (sml >= 0) {
                        s1lc2p = tags[sml - 1];
                        s1lc2w = words[sml - 1];
                        s1lc2l = state.getDependency(sml, labelNullIndex);
                    }
                }

                int rightMost = state.rightMostModifier(s1Position);
                if (rightMost >= 0) {
                    s1rc1p = tags[rightMost - 1];
                    s1rc1w = words[rightMost - 1];
                    s1rc1l = state.getDependency(rightMost, labelNullIndex);

                    int r2 = state.rightMostModifier(rightMost);
                    if (r2 >= 0) {
                        s1rc11p = tags[r2 - 1];
                        s1rc11w = words[r2 - 1];
                        s1rc11l = state.getDependency(r2, labelNullIndex);
                    }

                    int smr = state.secondRightMostModifier(s1Position);
                    if (smr >= 0) {
                        s1rc2p = tags[smr - 1];
                        s1rc2w = words[smr - 1];
                        s1rc2l = state.getDependency(smr, labelNullIndex);
                    }
                }


                if (1 < state.stackSize()) {
                    int top2 = state.pop();
                    int s2Position = state.peek();
                    s2w = s2Position > 0 ? words[s2Position - 1] : IndexMaps.RootIndex;
                    s2p = s2Position > 0 ? tags[s2Position - 1] : IndexMaps.RootIndex;

                    if (1 < state.stackSize()) {
                        int top3 = state.pop();
                        int s3Position = state.peek();
                        s3w = s3Position > 0 ? words[s3Position - 1] : IndexMaps.RootIndex;
                        s3p = s3Position > 0 ? tags[s3Position - 1] : IndexMaps.RootIndex;
                        state.push(top3);
                    }
                    state.push(top2);
                }
                state.push(top1);
            }

            int s0Position = state.peek();
            s0w = s0Position > 0 ? words[s0Position - 1] : IndexMaps.RootIndex;
            s0p = s0Position > 0 ? tags[s0Position - 1] : IndexMaps.RootIndex;
            int leftMost = state.leftMostModifier(s0Position);
            if (leftMost >= 0) {
                s0lc1p = tags[leftMost - 1];
                s0lc1w = words[leftMost - 1];
                s0lc1l = state.getDependency(leftMost, labelNullIndex);

                int l2 = state.leftMostModifier(leftMost);
                if (l2 >= 0) {
                    s0lc11p = tags[l2 - 1];
                    s0lc11w = words[l2 - 1];
                    s0lc11l = state.getDependency(l2, labelNullIndex);
                }

                int sml = state.secondLeftMostModifier(s0Position);
                if (sml >= 0) {
                    s0lc2p = tags[sml - 1];
                    s0lc2w = words[sml - 1];
                    s0lc2l = state.getDependency(sml, labelNullIndex);
                }
            }

            int rightMost = state.rightMostModifier(s0Position);
            if (rightMost >= 0) {
                s0rc1p = tags[rightMost - 1];
                s0rc1w = words[rightMost - 1];
                s0rc1l = state.getDependency(rightMost, labelNullIndex);

                int r2 = state.rightMostModifier(rightMost);
                if (r2 >= 0) {
                    s0rc11p = tags[r2 - 1];
                    s0rc11w = words[r2 - 1];
                    s0rc11l = state.getDependency(r2, labelNullIndex);
                }

                int smr = state.secondRightMostModifier(s0Position);
                if (smr >= 0) {
                    s0rc2p = tags[smr - 1];
                    s0rc2w = words[smr - 1];
                    s0rc2l = state.getDependency(smr, labelNullIndex);
                }
            }


        }
        double[] baseFeatureIds = new double[52];

        int index = 0;
        baseFeatureIds[index++] = s0w;
        baseFeatureIds[index++] = s1w;
        baseFeatureIds[index++] = s2w;
        baseFeatureIds[index++] = s3w;
        baseFeatureIds[index++] = b0w;
        baseFeatureIds[index++] = b1w;
        baseFeatureIds[index++] = b2w;
        baseFeatureIds[index++] = b3w;
        baseFeatureIds[index++] = s0lc1w;
        baseFeatureIds[index++] = s0rc1w;
        baseFeatureIds[index++] = s0lc2w;
        baseFeatureIds[index++] = s0rc2w;
        baseFeatureIds[index++] = s1lc1w;
        baseFeatureIds[index++] = s1rc1w;
        baseFeatureIds[index++] = s1lc2w;
        baseFeatureIds[index++] = s1rc2w;
        baseFeatureIds[index++] = s0lc11w;
        baseFeatureIds[index++] = s0rc11w;
        baseFeatureIds[index++] = s1lc11w;
        baseFeatureIds[index++] = s1rc11w;

        baseFeatureIds[index++] = s0p;
        baseFeatureIds[index++] = s1p;
        baseFeatureIds[index++] = s2p;
        baseFeatureIds[index++] = s3p;
        baseFeatureIds[index++] = b0p;
        baseFeatureIds[index++] = b1p;
        baseFeatureIds[index++] = b2p;
        baseFeatureIds[index++] = b3p;
        baseFeatureIds[index++] = s0lc1p;
        baseFeatureIds[index++] = s0rc1p;
        baseFeatureIds[index++] = s0lc2p;
        baseFeatureIds[index++] = s0rc2p;
        baseFeatureIds[index++] = s1lc1p;
        baseFeatureIds[index++] = s1rc1p;
        baseFeatureIds[index++] = s1lc2p;
        baseFeatureIds[index++] = s1rc2p;
        baseFeatureIds[index++] = s0lc11p;
        baseFeatureIds[index++] = s0rc11p;
        baseFeatureIds[index++] = s1lc11p;
        baseFeatureIds[index++] = s1rc11p;

        baseFeatureIds[index++] = s0lc1l;
        baseFeatureIds[index++] = s0rc1l;
        baseFeatureIds[index++] = s0lc2l;
        baseFeatureIds[index++] = s0rc2l;
        baseFeatureIds[index++] = s1lc1l;
        baseFeatureIds[index++] = s1rc1l;
        baseFeatureIds[index++] = s1lc2l;
        baseFeatureIds[index++] = s1rc2l;
        baseFeatureIds[index++] = s0lc11l;
        baseFeatureIds[index++] = s0rc11l;
        baseFeatureIds[index++] = s1lc11l;
        baseFeatureIds[index++] = s1rc11l;

        return baseFeatureIds;
    }
}
