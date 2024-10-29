#!/bin/sh
# AUTHOR=Gustaf Gren
# NOTES=Based on the OpenNMT tutorial:
# https://github.com/ymoslem/OpenNMT-Tutorial/blob/main/2-NMT-Training.ipynb
# onmt_build_vocab -config ../current_config.yaml -n_sample -1 -num_threads 12
onmt_train -config ../current_config.yaml -num_threads 12
