#!/bin/bash

##################################################################################

# Acquire data
wget http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz

gunzip loc-gowalla_totalCheckins.txt.gz

#  Get some rows
head -n 100000 loc-gowalla_totalCheckins.txt  > test.tab

# Call matlab function to plot maps
matlab < test_map_function.m


