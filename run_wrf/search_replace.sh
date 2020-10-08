#!/bin/bash

set -e

infile=$1
outfile=$2
verbose=$3
shift
shift
shift

if [ "$infile" != "$outfile" ]; then
  cp $infile $outfile
else
  cp $infile ${infile}_old
  infile=${infile}_old
fi


raise=false
missing_params=""
while [[ $# -gt 0 ]]
do
  if grep -x -q "\s*$1\s*=.*" ${outfile}
  then
    line=$(grep "\s$1\s*=" ${outfile})
    value_change=$(python ${code_dir}/check_namelist_value.py "$line" "$2")
    if [ -z "$value_change" ]
    then
      sed  -i -r -e "s/\s$1\s*=.*/ $1 = $2/g" $outfile
    fi

  else
    missing_params="$1, $missing_params"
    raise=true
  fi
  shift
  shift
done

if ( $raise )
then
  >&2 echo "ERROR: parameters not found in namelist file: $missing_params"
  exit 1
fi

if [ $verbose == 1 ]
then
  echo "Difference between files:"
  diff $infile $outfile | cat
fi
