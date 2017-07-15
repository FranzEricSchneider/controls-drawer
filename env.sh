# Make the basic folders that will be built off later
export CNC_DRAWER_BASE=~/projects/controls-drawer
export CNC_DRAWER_SOFTWARE=${CNC_DRAWER_BASE}/software
export CNC_DRAWER_RESULTS=${CNC_DRAWER_BASE}/results
echo "CNC_DRAWER_BASE = ${CNC_DRAWER_BASE}"
echo "CNC_DRAWER_SOFTWARE = ${CNC_DRAWER_SOFTWARE}"
echo "CNC_DRAWER_RESULTS = ${CNC_DRAWER_RESULTS}"

# Let the Python files find each other
export PYTHONPATH=${CNC_DRAWER_SOFTWARE}:${PYTHONPATH}
export PYTHONPATH=${CNC_DRAWER_SOFTWARE}/messages:${PYTHONPATH}
echo "PYTHONPATH = ${PYTHONPATH}"