#!/bin/bash

find . -name "A.csv" -o -name "S.csv" -o -name "component_stats.csv" | tar cvzf outputs.tar.gz -T -

