# D165 smoke: does run_to_sim_time through the ros_gz_bridge ControlWorld service stop gz at T, with a frame at T?
import sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from ros_gz_interfaces.srv import ControlWorld

W = sys.argv[1]
rclpy.init()
n = Node("d165_probe")
clk = [None]
stamps = []
n.create_subscription(Clock, "/clock", lambda m: clk.__setitem__(0, m.clock.sec * 10**9 + m.clock.nanosec), 10)
n.create_subscription(Image, "/ariadne/agent_0/left/image_raw",
                      lambda m: stamps.append(m.header.stamp.sec * 10**9 + m.header.stamp.nanosec),
                      QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE))
cli = n.create_client(ControlWorld, f"/world/{W}/control")
print("service ready:", cli.wait_for_service(timeout_sec=60), flush=True)


def spin(sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        rclpy.spin_once(n, timeout_sec=0.1)


spin(5)
print("paused clock before:", clk[0], "frames:", len(stamps), flush=True)
for target in (2, 4):
    req = ControlWorld.Request()
    req.world_control.run_to_sim_time.sec = target
    fut = cli.call_async(req)
    t0 = time.time()
    while not fut.done() and time.time() - t0 < 30:
        rclpy.spin_once(n, timeout_sec=0.1)
    print(f"run_to {target} s: service result", fut.result().success if fut.done() else "TIMEOUT", flush=True)
    last, still, t0 = None, 0, time.time()
    while time.time() - t0 < 240:          # until the clock has not moved for 15 s of wall time
        spin(1)
        if clk[0] == last:
            still += 1
            if still >= 15:
                break
        else:
            still, last = 0, clk[0]
    tgt = target * 10**9
    print(f"  clock stopped at {clk[0]} ns (target {tgt}); frames {len(stamps)}, last stamp {max(stamps) if stamps else None},"
          f" frame AT target: {tgt in set(stamps)}, frames after target: {sum(s > tgt for s in stamps)}", flush=True)
rclpy.shutdown()
