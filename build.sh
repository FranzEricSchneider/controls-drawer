# Must source env.sh first
if [ -z ${CNC_DRAWER_SOFTWARE+x} ]; then
	echo "MUST SOURCE ENV.SH BEFORE RUNNING THIS";
else
	# Build the LCM messages
	cd ${CNC_DRAWER_SOFTWARE}/messages
	lcm-gen -pj lcmtypes_image_t.lcm
	lcm-gen -pj lcmtypes_image_points_2d_t.lcm
	lcm-gen -pj lcmtypes_image_request_t.lcm
	lcm-gen -pj lcmtypes_relative_position_t.lcm
	lcm-gen -pj lcmtypes_tool_state_t.lcm
	lcm-gen -pj lcmtypes_table_state_t.lcm
	lcm-gen -pj lcmtypes_velocity_t.lcm
	javac -cp lcm.jar lcmtypes/*.java  # lcm.jar can be linked from /usr/share/local/java/
	jar cf controls-drawer.jar lcmtypes/*.class
	echo "Built LCM messages in $(pwd)"
	cd ${CNC_DRAWER_BASE}
fi
