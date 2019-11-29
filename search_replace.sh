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
while [[ $# -gt 0 ]]
do
  if grep -x -q "\s*$1\s*=.*" ${outfile}
  then
    line=$(grep "\s$1\s*=" ${outfile})
    echo "${code_dir}/check_namelist_value.py '$line' '$2'"
    value_change=$(python ${code_dir}/check_namelist_value.py "$line" "$2")
    if [ -z "$value_change" ]
    then
      sed  -i -r -e "s/\s$1\s*=.*/ $1 = $2/g" $outfile
    fi

  else
    >&2 echo "ERROR: parameter $1 not found in namelist file!"
    raise=true
  fi
  shift
  shift
done

if ( $raise )
then
  exit 1
fi

if [ verbose == 1 ]
then
  echo "Difference between files:"
  diff $infile $outfile | cat
fi
