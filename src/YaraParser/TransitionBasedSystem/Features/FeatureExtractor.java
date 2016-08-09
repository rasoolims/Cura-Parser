/**
 * Copyright 2014, Yahoo! Inc.
 * Licensed under the terms of the Apache License 2.0. See LICENSE file at the project 0 for terms.
 */

package YaraParser.TransitionBasedSystem.Features;

import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.Sentence;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.State;

public class FeatureExtractor {
    /**
     * Given a list of templates, extracts all features for the given state
     *
     * @param configuration
     * @return
     * @throws Exception
     */
    public static Object[] extractAllParseFeatures(Configuration configuration, int length) {
        if (length == 26)
            return extractBasicFeatures(configuration, length);
        else
            return extractExtendedFeatures(configuration, length);
    }

    /**
     * Given a list of templates, extracts all features for the given state
     *
     * @param configuration
     * @return
     * @throws Exception
     */
    private static Object[] extractExtendedFeatures(Configuration configuration, int length) {
        Object[] featureMap = new Object[length];

        State state = configuration.state;
        Sentence sentence = configuration.sentence;

        int b0Position = 0;
        int b1Position = 0;
        int b2Position = 0;
        int s0Position = 0;

        long svr = 0; // stack right valency
        long svl = 0; // stack left valency
        long bvl = 0; // buffer left valency

        long b0w = 0;
        long b0p = 0;

        long b1w = 0;
        long b1p = 0;

        long b2w = 0;
        long b2p = 0;

        long s0w = 0;
        long s0p = 0;
        long s0l = 0;

        long b0l1p = 0;
        long b0l1w = 0;
        long b0l1l = 0;

        long b0l2w = 0;
        long b0l2p = 0;
        long b0l2l = 0;

        long sr1p = 0;
        long sr1w = 0;
        long sr1l = 0;

        long sh0w = 0;
        long sh0p = 0;
        long sh0l = 0;

        long s0l1p = 0;
        long s0l1w = 0;
        long s0l1l = 0;

        long s0r2w = 0;
        long s0r2p = 0;
        long s0r2l = 0;

        long sh1w = 0;
        long sh1p = 0;

        long s0l2w = 0;
        long s0l2p = 0;
        long s0l2l = 0;

        long sdl = 0;
        long sdr = 0;
        long bdl = 0;

        int[] words = sentence.getWords();
        int[] tags = sentence.getTags();

        if (0 < state.bufferSize()) {
            b0Position = state.bufferHead();
            b0w = b0Position == 0 ? 0 : words[b0Position - 1];
            b0w += 2;
            b0p = b0Position == 0 ? 0 : tags[b0Position - 1];
            b0p += 2;
            bvl = state.leftValency(b0Position);

            int leftMost = state.leftMostModifier(b0Position);
            if (leftMost >= 0) {
                b0l1p = leftMost == 0 ? 0 : tags[leftMost - 1];
                b0l1p += 2;
                b0l1w = leftMost == 0 ? 0 : words[leftMost - 1];
                b0l1w += 2;
                b0l1l = state.getDependency(leftMost);
                b0l1l += 2;

                int l2 = state.secondLeftMostModifier(b0Position);
                if (l2 >= 0) {
                    b0l2w = l2 == 0 ? 0 : words[l2 - 1];
                    b0l2w += 2;
                    b0l2p = l2 == 0 ? 0 : tags[l2 - 1];
                    b0l2p += 2;
                    b0l2l = state.getDependency(l2);
                    b0l2l += 2;
                }
            }

            if (1 < state.bufferSize()) {
                b1Position = state.getBufferItem(1);
                b1w = b1Position == 0 ? 0 : words[b1Position - 1];
                b1w += 2;
                b1p = b1Position == 0 ? 0 : tags[b1Position - 1];
                b1p += 2;

                if (2 < state.bufferSize()) {
                    b2Position = state.getBufferItem(2);

                    b2w = b2Position == 0 ? 0 : words[b2Position - 1];
                    b2w += 2;
                    b2p = b2Position == 0 ? 0 : tags[b2Position - 1];
                    b2p += 2;
                }
            }
        }

        if (0 < state.stackSize()) {
            s0Position = state.peek();
            s0w = s0Position == 0 ? 0 : words[s0Position - 1];
            s0w += 2;
            s0p = s0Position == 0 ? 0 : tags[s0Position - 1];
            s0p += 2;
            s0l = state.getDependency(s0Position);
            s0l += 2;

            svl = state.leftValency(s0Position);
            svr = state.rightValency(s0Position);

            int leftMost = state.leftMostModifier(s0Position);
            if (leftMost >= 0) {
                s0l1p = leftMost == 0 ? 0 : tags[leftMost - 1];
                s0l1p += 2;
                s0l1w = leftMost == 0 ? 0 : words[leftMost - 1];
                s0l1w += 2;
                s0l1l = state.getDependency(leftMost);
                s0l1l += 2;
            }

            int rightMost = state.rightMostModifier(s0Position);
            if (rightMost >= 0) {
                sr1p = rightMost == 0 ? 0 : tags[rightMost - 1];
                sr1p += 2;
                sr1w = rightMost == 0 ? 0 : words[rightMost - 1];
                sr1w += 2;
                sr1l = state.getDependency(rightMost);
                sr1l += 2;
            }

            int headIndex = state.getHead(s0Position);
            if (headIndex >= 0) {
                sh0w = headIndex == 0 ? 0 : words[headIndex - 1];
                sh0w += 2;
                sh0p = headIndex == 0 ? 0 : tags[headIndex - 1];
                sh0p += 2;
                sh0l = state.getDependency(headIndex);
                sh0l += 2;
            }

            if (leftMost >= 0) {
                int l2 = state.secondLeftMostModifier(s0Position);
                if (l2 >= 0) {
                    s0l2w = l2 == 0 ? 0 : words[l2 - 1];
                    s0l2w += 2;
                    s0l2p = l2 == 0 ? 0 : tags[l2 - 1];
                    s0l2p += 2;
                    s0l2l = state.getDependency(l2);
                    s0l2l += 2;
                }
            }
            if (headIndex >= 0) {
                if (state.hasHead(headIndex)) {
                    int h2 = state.getHead(headIndex);
                    sh1w = h2 == 0 ? 0 : words[h2 - 1];
                    sh1w += 2;
                    sh1p = h2 == 0 ? 0 : tags[h2 - 1];
                    sh1p += 2;
                }
            }
            if (rightMost >= 0) {
                int r2 = state.secondRightMostModifier(s0Position);
                if (r2 >= 0) {
                    s0r2w = r2 == 0 ? 0 : words[r2 - 1];
                    s0r2w += 2;
                    s0r2p = r2 == 0 ? 0 : tags[r2 - 1];
                    s0r2p += 2;
                    s0r2l = state.getDependency(r2);
                    s0r2l += 2;
                }
            }
        }
        int index = 0;

        long b0wp = b0p;
        b0wp |= (b0w << 8);
        long b1wp = b1p;
        b1wp |= (b1w << 8);
        long s0wp = s0p;
        s0wp |= (s0w << 8);
        long b2wp = b2p;
        b2wp |= (b2w << 8);

        /**
         * From single words
         */
        if (s0w != 1) {
            featureMap[index++] = s0wp;
            featureMap[index++] = s0w;
        } else {
            featureMap[index++] = null;
            featureMap[index++] = null;
        }
        featureMap[index++] = s0p;

        if (b0w != 1) {
            featureMap[index++] = b0wp;
            featureMap[index++] = b0w;
        } else {
            featureMap[index++] = null;
            featureMap[index++] = null;
        }
        featureMap[index++] = b0p;

        if (b1w != 1) {
            featureMap[index++] = b1wp;
            featureMap[index++] = b1w;
        } else {
            featureMap[index++] = null;
            featureMap[index++] = null;
        }
        featureMap[index++] = b1p;

        if (b2w != 1) {
            featureMap[index++] = b2wp;
            featureMap[index++] = b2w;
        } else {
            featureMap[index++] = null;
            featureMap[index++] = null;
        }
        featureMap[index++] = b2p;

        /**
         * from word pairs
         */
        if (s0w != 1 && b0w != 1) {
            featureMap[index++] = (s0wp << 28) | b0wp;
            featureMap[index++] = (s0wp << 20) | b0w;
            featureMap[index++] = (s0w << 28) | b0wp;
        } else {
            featureMap[index++] = null;
            featureMap[index++] = null;
            featureMap[index++] = null;
        }

        if (s0w != 1) {
            featureMap[index++] = (s0wp << 8) | b0p;
        } else {
            featureMap[index++] = null;
        }

        if (b0w != 1) {
            featureMap[index++] = (s0p << 28) | b0wp;
        } else {
            featureMap[index++] = null;
        }

        if (s0w != 1 && b0w != 1) {
            featureMap[index++] = (s0w << 20) | b0w;
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = (s0p << 8) | b0p;
        featureMap[index++] = (b0p << 8) | b1p;

        /**
         * from three words
         */
        featureMap[index++] = (b0p << 16) | (b1p << 8) | b2p;
        featureMap[index++] = (s0p << 16) | (b0p << 8) | b1p;
        featureMap[index++] = (sh0p << 16) | (s0p << 8) | b0p;
        featureMap[index++] = (s0p << 16) | (s0l1p << 8) | b0p;
        featureMap[index++] = (s0p << 16) | (sr1p << 8) | b0p;
        featureMap[index++] = (s0p << 16) | (b0p << 8) | b0l1p;

        /**
         * distance
         */
        long distance = 0;
        if (s0Position > 0 && b0Position > 0)
            distance = Math.abs(b0Position - s0Position);
        if (s0w != 1) {
            featureMap[index++] = s0w | (distance << 20);
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = s0p | (distance << 8);
        if (b0w != 1) {
            featureMap[index++] = b0w | (distance << 20);
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = b0p | (distance << 8);
        if (s0w != 1 && b0w != 1) {
            featureMap[index++] = s0w | (b0w << 20) | (distance << 40);
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = s0p | (b0p << 8) | (distance << 28);

        /**
         * Valency information
         */
        if (s0w != 1) {
            featureMap[index++] = s0w | (svr << 20);
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = s0p | (svr << 8);
        if (s0w != 1) {
            featureMap[index++] = s0w | (svl << 20);
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = s0p | (svl << 8);
        if (b0w != 1) {
            featureMap[index++] = b0w | (bvl << 20);
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = b0p | (bvl << 8);

        /**
         * Unigrams
         */
        if (sh0w != 1) {
            featureMap[index++] = sh0w;
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = sh0p;
        featureMap[index++] = s0l;
        if (s0l1w != 1) {
            featureMap[index++] = s0l1w;
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = s0l1p;
        featureMap[index++] = s0l1l;
        if (sr1w != 1) {
            featureMap[index++] = sr1w;
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = sr1p;
        featureMap[index++] = sr1l;
        if (b0l1w != 1) {
            featureMap[index++] = b0l1w;
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = b0l1p;
        featureMap[index++] = b0l1l;

        /**
         * From third order features
         */
        if (sh1w != 1) {
            featureMap[index++] = sh1w;
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = sh1p;
        featureMap[index++] = sh0l;
        if (s0l2w != 1) {
            featureMap[index++] = s0l2w;
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = s0l2p;
        featureMap[index++] = s0l2l;
        if (s0r2w != 1) {
            featureMap[index++] = s0r2w;
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = s0r2p;
        featureMap[index++] = s0r2l;
        if (b0l2w != 1) {
            featureMap[index++] = b0l2w;
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = b0l2p;
        featureMap[index++] = b0l2l;
        featureMap[index++] = s0p | (s0l1p << 8) | (s0l2p << 16);
        featureMap[index++] = s0p | (sr1p << 8) | (s0r2p << 16);
        featureMap[index++] = s0p | (sh0p << 8) | (sh1p << 16);
        featureMap[index++] = b0p | (b0l1p << 8) | (b0l2p << 16);

        /**
         * label set
         */
        if (s0Position >= 0) {
            sdl = state.leftDependentLabels(s0Position);
            sdr = state.rightDependentLabels(s0Position);
        }

        if (b0Position >= 0) {
            bdl = state.leftDependentLabels(b0Position);
        }

        if (s0w != 1) {
            featureMap[index++] = (s0w + "|" + sdr);
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = (s0p + "|" + sdr);
        if (s0w != 1) {
            featureMap[index++] = s0w + "|" + sdl;
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = (s0p + "|" + sdl);
        if (b0w != 1) {
            featureMap[index++] = (b0w + "|" + bdl);
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = (b0p + "|" + bdl);
        return featureMap;
    }

    /**
     * Given a list of templates, extracts all features for the given state
     *
     * @param configuration
     * @return
     * @throws Exception
     */
    private static Long[] extractBasicFeatures(Configuration configuration, int length) {
        Long[] featureMap = new Long[length];

        State state = configuration.state;
        Sentence sentence = configuration.sentence;

        int b0Position = 0;
        int b1Position = 0;
        int b2Position = 0;
        int s0Position = 0;

        long b0w = 0;
        long b0p = 0;

        long b1w = 0;
        long b1p = 0;

        long b2w = 0;
        long b2p = 0;

        long s0w = 0;
        long s0p = 0;
        long bl0p = 0;
        long sr0p = 0;
        long sh0p = 0;

        long sl0p = 0;

        int[] words = sentence.getWords();
        int[] tags = sentence.getTags();

        if (0 < state.bufferSize()) {
            b0Position = state.bufferHead();
            b0w = b0Position == 0 ? 0 : words[b0Position - 1];
            b0w += 2;
            b0p = b0Position == 0 ? 0 : tags[b0Position - 1];
            b0p += 2;

            int leftMost = state.leftMostModifier(state.getBufferItem(0));
            if (leftMost >= 0) {
                bl0p = leftMost == 0 ? 0 : tags[leftMost - 1];
                bl0p += 2;
            }

            if (1 < state.bufferSize()) {
                b1Position = state.getBufferItem(1);
                b1w = b1Position == 0 ? 0 : words[b1Position - 1];
                b1w += 2;
                b1p = b1Position == 0 ? 0 : tags[b1Position - 1];
                b1p += 2;

                if (2 < state.bufferSize()) {
                    b2Position = state.getBufferItem(2);

                    b2w = b2Position == 0 ? 0 : words[b2Position - 1];
                    b2w += 2;
                    b2p = b2Position == 0 ? 0 : tags[b2Position - 1];
                    b2p += 2;
                }
            }
        }


        if (0 < state.stackSize()) {
            s0Position = state.peek();
            s0w = s0Position == 0 ? 0 : words[s0Position - 1];
            s0w += 2;
            s0p = s0Position == 0 ? 0 : tags[s0Position - 1];
            s0p += 2;

            int leftMost = state.leftMostModifier(s0Position);
            if (leftMost >= 0) {
                sl0p = leftMost == 0 ? 0 : tags[leftMost - 1];
                sl0p += 2;
            }

            int rightMost = state.rightMostModifier(s0Position);
            if (rightMost >= 0) {
                sr0p = rightMost == 0 ? 0 : tags[rightMost - 1];
                sr0p += 2;
            }

            int headIndex = state.getHead(s0Position);
            if (headIndex >= 0) {
                sh0p = headIndex == 0 ? 0 : tags[headIndex - 1];
                sh0p += 2;
            }

        }
        int index = 0;

        long b0wp = b0p;
        b0wp |= (b0w << 8);
        long b1wp = b1p;
        b1wp |= (b1w << 8);
        long s0wp = s0p;
        s0wp |= (s0w << 8);
        long b2wp = b2p;
        b2wp |= (b2w << 8);

        /**
         * From single words
         */
        if (s0w != 1) {
            featureMap[index++] = s0wp;
            featureMap[index++] = s0w;
        } else {
            featureMap[index++] = null;
            featureMap[index++] = null;
        }
        featureMap[index++] = s0p;

        if (b0w != 1) {
            featureMap[index++] = b0wp;
            featureMap[index++] = b0w;
        } else {
            featureMap[index++] = null;
            featureMap[index++] = null;
        }
        featureMap[index++] = b0p;

        if (b1w != 1) {
            featureMap[index++] = b1wp;
            featureMap[index++] = b1w;
        } else {
            featureMap[index++] = null;
            featureMap[index++] = null;
        }
        featureMap[index++] = b1p;

        if (b2w != 1) {
            featureMap[index++] = b2wp;
            featureMap[index++] = b2w;
        } else {
            featureMap[index++] = null;
            featureMap[index++] = null;
        }
        featureMap[index++] = b2p;

        /**
         * from word pairs
         */
        if (s0w != 1 && b0w != 1) {
            featureMap[index++] = (s0wp << 28) | b0wp;
            featureMap[index++] = (s0wp << 20) | b0w;
            featureMap[index++] = (s0w << 28) | b0wp;
        } else {
            featureMap[index++] = null;
            featureMap[index++] = null;
            featureMap[index++] = null;
        }

        if (s0w != 1) {
            featureMap[index++] = (s0wp << 8) | b0p;
        } else {
            featureMap[index++] = null;
        }

        if (b0w != 1) {
            featureMap[index++] = (s0p << 28) | b0wp;
        } else {
            featureMap[index++] = null;
        }

        if (s0w != 1 && b0w != 1) {
            featureMap[index++] = (s0w << 20) | b0w;
        } else {
            featureMap[index++] = null;
        }
        featureMap[index++] = (s0p << 8) | b0p;
        featureMap[index++] = (b0p << 8) | b1p;

        /**
         * from three words
         */
        featureMap[index++] = (b0p << 16) | (b1p << 8) | b2p;
        featureMap[index++] = (s0p << 16) | (b0p << 8) | b1p;
        featureMap[index++] = (sh0p << 16) | (s0p << 8) | b0p;
        featureMap[index++] = (s0p << 16) | (sl0p << 8) | b0p;
        featureMap[index++] = (s0p << 16) | (sr0p << 8) | b0p;
        featureMap[index++] = (s0p << 16) | (b0p << 8) | bl0p;
        return featureMap;
    }


    public static int[] extractBaseFeatures(Configuration configuration, IndexMaps maps) throws Exception {

        State state = configuration.state;
        Sentence sentence = configuration.sentence;

        int b0w = IndexMaps.NullIndex;
        int b0p = IndexMaps.NullIndex;

        int b1w = IndexMaps.NullIndex;
        int b1p = IndexMaps.NullIndex;

        int b2w = IndexMaps.NullIndex;
        int b2p = IndexMaps.NullIndex;

        int b3w = IndexMaps.NullIndex;
        int b3p = IndexMaps.NullIndex;

        int s0w = IndexMaps.NullIndex;
        int s0p = IndexMaps.NullIndex;
        int s0l = IndexMaps.LabelNullIndex;

        int s1w = IndexMaps.NullIndex;
        int s1p = IndexMaps.NullIndex;

        int s2w = IndexMaps.NullIndex;
        int s2p = IndexMaps.NullIndex;

        int s3w = IndexMaps.NullIndex;
        int s3p = IndexMaps.NullIndex;

        int b0l1w = IndexMaps.NullIndex;
        int b0l1p = IndexMaps.NullIndex;
        int b0l1l = IndexMaps.LabelNullIndex;

        int b0l2w = IndexMaps.NullIndex;
        int b0l2p = IndexMaps.NullIndex;
        int b0l2l = IndexMaps.LabelNullIndex;

        int b0llw = IndexMaps.NullIndex;
        int b0llp = IndexMaps.NullIndex;
        int b0lll = IndexMaps.LabelNullIndex;

        int sr1p = IndexMaps.NullIndex;
        int sr1w = IndexMaps.NullIndex;
        int sr1l = IndexMaps.LabelNullIndex;

        int s0rrp = IndexMaps.NullIndex;
        int s0rrw = IndexMaps.NullIndex;
        int s0rrl = IndexMaps.LabelNullIndex;

        int sh0w = IndexMaps.NullIndex;
        int sh0p = IndexMaps.NullIndex;
        int sh0l = IndexMaps.LabelNullIndex;

        int s0l1p = IndexMaps.NullIndex;
        int s0l1w = IndexMaps.NullIndex;
        int s0l1l = IndexMaps.LabelNullIndex;

        int s0llp = IndexMaps.NullIndex;
        int s0llw = IndexMaps.NullIndex;
        int s0lll = IndexMaps.LabelNullIndex;

        int s0r2w = IndexMaps.NullIndex;
        int s0r2p = IndexMaps.NullIndex;
        int s0r2l = IndexMaps.LabelNullIndex;

        int sh1w = IndexMaps.NullIndex;
        int sh1p = IndexMaps.NullIndex;

        int s0l2w = IndexMaps.NullIndex;
        int s0l2p = IndexMaps.NullIndex;
        int s0l2l = IndexMaps.LabelNullIndex;

        int[] words = sentence.getWords();
        int[] tags = sentence.getTags();

        if (0 < state.bufferSize()) {
            int b0Position = state.bufferHead();
            b0w = (words[b0Position - 1]);
            b0p = (tags[b0Position - 1]);

            int leftMost = state.leftMostModifier(b0Position);
            if (leftMost >= 0) {
                b0l1w = (words[leftMost - 1]);
                b0l1p = (tags[leftMost - 1]);
                b0l1l = (state.getDependency(leftMost));

                int l2 = state.secondLeftMostModifier(b0Position);
                if (l2 >= 0) {
                    b0l2w = (words[l2 - 1]);
                    b0l2p = (tags[l2 - 1]);
                    b0l2l = (state.getDependency(l2));
                }

                int secondLeftMost = state.leftMostModifier(leftMost);
                if (secondLeftMost >= 0) {
                    b0llw = (words[secondLeftMost - 1]);
                    b0llp = (tags[secondLeftMost - 1]);
                    b0lll = (state.getDependency(secondLeftMost));
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
            s0w = (words[s0Position - 1]);
            s0p = (tags[s0Position - 1]);
            s0l = (state.getDependency(s0Position));

            if (1 < state.stackSize()) {
                int top1 = state.pop();
                int s1Position = state.peek();
                s1w = (words[s1Position - 1]);
                s1p = (tags[s1Position - 1]);

                if (1 < state.stackSize()) {
                    int top2 = state.pop();
                    int s2Position = state.peek();
                    s2w = (words[s2Position - 1]);
                    s2p = (tags[s2Position - 1]);

                    if (1 < state.stackSize()) {
                        int top3 = state.pop();
                        int s3Position = state.peek();
                        s3w = (words[s3Position - 1]);
                        s3p = (tags[s3Position - 1]);
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
                s0l1l = (state.getDependency(leftMost));

                int secondLeftMost = state.leftMostModifier(leftMost);
                if (secondLeftMost >= 0) {
                    s0llp = (tags[secondLeftMost - 1]);
                    s0llw = (words[secondLeftMost - 1]);
                    s0lll = (state.getDependency(secondLeftMost));
                }
            }

            int rightMost = state.rightMostModifier(s0Position);
            if (rightMost >= 0) {
                sr1p = (tags[rightMost - 1]);
                sr1w = (words[rightMost - 1]);
                sr1l = (state.getDependency(rightMost));

                int secondRightMost = state.rightMostModifier(rightMost);
                if (secondRightMost >= 0) {
                    s0rrp = (tags[secondRightMost - 1]);
                    s0rrw = (words[secondRightMost - 1]);
                    s0rrl = (state.getDependency(secondRightMost));
                }
            }

            int headIndex = state.getHead(s0Position);
            if (headIndex >= 0) {
                sh0w = (words[headIndex - 1]);
                sh0p = (tags[headIndex - 1]);
                sh0l = (state.getDependency(headIndex));
            }

            if (leftMost >= 0) {
                int l2 = state.secondLeftMostModifier(s0Position);
                if (l2 >= 0) {
                    s0l2w = (words[l2 - 1]);
                    s0l2p = (tags[l2 - 1]);
                    s0l2l = (state.getDependency(l2));
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
                    s0r2l = (state.getDependency(r2));
                }
            }
        }
        int[] baseFeatureIds = new int[49];

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
}
