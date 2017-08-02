#!/bin/bash
datasynth -walkers $1 -tmax $2 -voxels $3 -p $4 -schemefile $5 -initial uniform -substrate cylinder -packing hex -cylinderrad $6 -cylindersep $7 > $8