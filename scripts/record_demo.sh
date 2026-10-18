#!/bin/bash
# Record a screen capture of rviz while our particle filter runs live on a bag.
# Produces <name>.mp4 (8 fps) and <name>.gif (4 fps, smaller, for the README) in docs/media/.
#
# usage: scripts/record_demo.sh <name> <bag dir> <map yaml> [rate] [initial x y yaw_deg]
#   rate:              bag playback speed (default 2, so a 90 s bag becomes a 45 s video)
#   initial x y yaw:   optional initial guess sent on /initialpose, like a 2D Pose Estimate click;
#                      without it the filter starts from the robot's odometry pose
#
# Needs an X11 display (works under XWayland) and GStreamer's ximagesrc/x264enc/pngenc plugins.
# job control: background processes get their own process group and, unlike in a plain
# non-interactive script, don't ignore SIGINT -- so the stop signals below actually work
set -m

stop() {  # stop() <pid>: Ctrl+C a process, and force-kill it if it hasn't exited after 15 s
  kill -INT "$1" 2>/dev/null || return 0
  for _ in $(seq 30); do kill -0 "$1" 2>/dev/null || return 0; sleep 0.5; done
  kill -9 "$1" 2>/dev/null; true
}

NAME=$1; BAG=$(realpath "$2"); MAP=$(realpath "$3"); RATE=${4:-2}
PKG=$(cd "$(dirname "$0")/.." && pwd)
OUT=$PKG/docs/media; FRAMES=$(mktemp -d); mkdir -p "$OUT"

# GUI tools launched from VS Code's snap terminal pick up the snap's libraries; strip that, and conda
for v in GTK_PATH GTK_EXE_PREFIX LOCPATH GIO_MODULE_DIR GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR \
         GTK_IM_MODULE_FILE GSETTINGS_SCHEMA_DIR XDG_DATA_HOME CONDA_PREFIX; do unset $v; done
[ -n "$XDG_DATA_DIRS_VSCODE_SNAP_ORIG" ] && export XDG_DATA_DIRS="$XDG_DATA_DIRS_VSCODE_SNAP_ORIG"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | paste -sd:)
export QT_QPA_PLATFORM=xcb DISPLAY=${DISPLAY:-:0}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
source /opt/ros/jazzy/setup.bash
source "$PKG/../../install/setup.bash"

ros2 launch robot_localization test_pf.py map_yaml:="$MAP" > /tmp/record_demo_pf.log 2>&1 &
PF=$!
rviz2 -d "$PKG/rviz/pf_demo.rviz" --ros-args -p use_sim_time:=true > /tmp/record_demo_rviz.log 2>&1 &
RVIZ=$!
sleep 10
XID=$(xwininfo -root -tree | grep '("rviz2" "rviz2")' | grep -v '"rviz2": ' | head -1 | awk '{print $1}')

if [ -n "$5" ]; then
  python3 - "$5" "$6" "$7" <<'EOF'
import math, sys, rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
x, y, yaw = float(sys.argv[1]), float(sys.argv[2]), math.radians(float(sys.argv[3]))
rclpy.init(); node = rclpy.create_node('demo_initial_pose')
pub = node.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
msg = PoseWithCovarianceStamped()
msg.header.frame_id = 'map'
msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
while pub.get_subscription_count() == 0:
    rclpy.spin_once(node, timeout_sec=0.2)
pub.publish(msg); rclpy.spin_once(node, timeout_sec=0.5)
EOF
fi

# capture the rviz window as 8 fps PNG frames, cropped to the 3D view (rviz's tool and status bars
# removed, to an even size since x264 needs one); the mp4 and gif are built from these frames afterwards
# (encoding to mp4 live, through a tee, stalls the pipeline)
gst-launch-1.0 -e -q ximagesrc xid="$XID" use-damage=false ! video/x-raw,framerate=8/1 \
  ! videocrop top=63 bottom=33 left=16 right=16 ! videoconvert ! pngenc compression-level=1 \
  ! multifilesink location="$FRAMES/%05d.png" &
GST=$!
sleep 1
ros2 bag play "$BAG" --clock --rate "$RATE" > /tmp/record_demo_play.log 2>&1 || true
sleep 3
stop $GST
stop $RVIZ
stop $PF

gst-launch-1.0 -q multifilesrc location="$FRAMES/%05d.png" caps="image/png,framerate=8/1" ! pngdec \
  ! videoconvert ! video/x-raw,format=I420 ! x264enc speed-preset=medium pass=quant quantizer=22 \
  ! video/x-h264,profile=main ! mp4mux ! filesink location="$OUT/$NAME.mp4"

python3 - "$FRAMES" "$OUT/$NAME.gif" <<'EOF'
import glob, sys
from PIL import Image
frames = [Image.open(f).convert('RGB') for f in sorted(glob.glob(sys.argv[1] + '/*.png'))[::2]]  # 4 fps
w = 480
frames = [f.resize((w, int(f.height * w / f.width)), Image.LANCZOS).quantize(colors=64, method=Image.MEDIANCUT)
          for f in frames]
frames[0].save(sys.argv[2], save_all=True, append_images=frames[1:], duration=250, loop=0, optimize=True)
print('%d frames -> %s' % (len(frames), sys.argv[2]))
EOF
rm -rf "$FRAMES"
ls -la "$OUT/$NAME".*
