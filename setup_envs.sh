#!/bin/bash

NOTEBOOKS=notebooks

for f in $(ls -1 $NOTEBOOKS); do
  current=$NOTEBOOKS/$f

  echo
  echo "Setting up environment for $current"
  echo

  virtualenv "$current/spreadr_env"
  . $current/spreadr_env/bin/activate
  pip install -r $current/spreadr/requirements.txt
  deactivate

done

echo
echo "All done."
