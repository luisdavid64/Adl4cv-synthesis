#!/bin/bash

# Creates a dictionary of all furniture from ATISS base file

cat ./base.py | awk NF | awk '!/[{}]/' | awk -F '[,:\n]' '
    BEGIN {print "THREED_FUTURE_FURNITURE = {"}
    !seen[$0]++ {printf "%s: %s,\n", $1, $2}
    END {print "}"}
' > ./../data/threed_future_labels.py