#!/bin/bash

set -e

infile=$1
outfile=$2
shift
shift

if [ "$infile" != "$outfile" ]; then
  cp $infile $outfile
else
  cp $infile ${infile}_old
  infile=${infile}_old
fi

while [[ $# -gt 0 ]]
do
  if grep -q "\s$1\s*=" $outfile
  then
    sed  -i -r -e "s/\s$1\s*=.*/ $1 = $2/g" $outfile
  else
    echo "WARNING: parameter $1 not found in namelist file!"
  fi
  shift
  shift
done
echo "Difference between files:"
diff $infile $outfile | cat
