# Make the basic folders that will be built off later
export CNC_DRAWER_BASE=/home/eon-alone/projects/controls-drawer
export CNC_DRAWER_SOFTWARE=${CNC_DRAWER_BASE}/software
echo "CNC_DRAWER_BASE = ${CNC_DRAWER_BASE}"
echo "CNC_DRAWER_SOFTWARE = ${CNC_DRAWER_SOFTWARE}"

# Let the Python files find each other
export PYTHONPATH=${CNC_DRAWER_SOFTWARE}:${PYTHONPATH}
export PYTHONPATH=${CNC_DRAWER_SOFTWARE}/messages:${PYTHONPATH}