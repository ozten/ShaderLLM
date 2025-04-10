# DATA

I am creating various glsl shaders in `data/shaders`. Each in it's own directory
and then using [comb-synth-gen](https://github.com/ozten/comb-synth-gen) to expand
each shader against a bunch of permutations.

The template source is named `NNN.frag`. These generated variants are numbered 000.frag, 001.frag, etc.

In the data preperation stage, all of the numeric files are collected, randomized, and copied into `data/train` and `data/val`.