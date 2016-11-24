source env.sh

cd ${CNC_DRAWER_SOFTWARE}/messages/
lcm-gen -p lcmtypes_image_t.lcm
lcm-gen -p lcmtypes_image_request_t.lcm
lcm-gen -p lcmtypes_velocity_t.lcm
echo "Built LCM messages in $(pwd)"
cd ${CNC_DRAWER_BASE}/
