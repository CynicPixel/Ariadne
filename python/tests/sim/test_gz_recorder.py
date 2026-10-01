"""ariadne_gz_recorder fails LOUDLY (DECISIONS D40).

Each test drives the real recorder binary with a fake gz-transport publisher
(gz.msgs.Clock on a private topic, period 1 ms, end 10 ms) and asserts the exit
code, the JSON report and the error printed for one way a message can go missing.
A clean stream must pass and land in the bag exactly; everything else must fail.

Runs in the sim image with ROS sourced:  infra/scripts/worldgen.sh simtest
"""
import json
import os
import signal
import subprocess
import sys
import time
import uuid

import pytest

ROOT = __file__.rsplit("/python/", 1)[0]
sys.path.insert(0, os.path.join(ROOT, "infra", "scripts"))

import sim_record as S  # noqa: E402

os.environ.setdefault("GZ_PARTITION", f"ariadne_test_{uuid.uuid4().hex[:8]}")
from gz.msgs10.clock_pb2 import Clock  # noqa: E402
from gz.transport13 import Node  # noqa: E402

MS = 1_000_000
END = 10 * MS
_node = None


@pytest.fixture(scope="module", autouse=True)
def built():
    S.build_plugins()


def node():
    global _node
    if _node is None:
        _node = Node()
    return _node


def start(tmp_path, topic, *extra, pair=("gz.msgs.Clock", "rosgraph_msgs/msg/Clock")):
    spec = f"{topic},{topic},{pair[0]},{pair[1]},{MS},0"
    p = subprocess.Popen([S.RECORDER_TOOL, "--out", str(tmp_path / "bag"), "--end-ns", str(END),
                          "--report", str(tmp_path / "report.json"), "--connect-timeout-s", "20",
                          "--topic", spec, *extra],
                         env=dict(os.environ, GZ_TRANSPORT_RCVHWM="0"),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p


def run(tmp_path, stamps_ms, *extra, interrupt=False):
    """Advertise, start the recorder, wait for CONNECTED, publish, collect."""
    topic = f"/t_{uuid.uuid4().hex[:8]}"
    pub = node().advertise(topic, Clock)
    p = start(tmp_path, topic, *extra)
    line = p.stdout.readline().strip()
    assert line == "CONNECTED", (line, p.stderr.read())
    for ms in stamps_ms:
        m = Clock()
        m.sim.sec, m.sim.nsec = int(ms * MS) // 1_000_000_000, int(ms * MS) % 1_000_000_000
        assert pub.publish(m)
        time.sleep(0.002)
    if interrupt:
        time.sleep(1.0)
        p.send_signal(signal.SIGINT)
    out, err = p.communicate(timeout=60)
    rep = json.load(open(tmp_path / "report.json")) if (tmp_path / "report.json").exists() else None
    return p.returncode, out, err, rep, topic


def test_clean_stream_passes_and_is_written_exactly(tmp_path):
    rc, out, err, rep, topic = run(tmp_path, range(0, 11))
    assert rc == 0, err
    assert "COMPLETE" in out and "ERROR" not in err
    t = rep["topics"][topic]
    assert (t["count"], t["first_ns"], t["last_ns"], t["lost"]) == (11, 0, END, 0)
    bag = S.read_bag(str(tmp_path / "bag"))
    assert bag[topic]["stamps"] == list(range(0, END + 1, MS))
    assert bag[topic]["log_ns"] == bag[topic]["stamps"]          # log time = stamp


def test_gap_is_lost_and_fails(tmp_path):
    rc, out, err, rep, topic = run(tmp_path, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
    assert rc == 1 and "FAILED" in out
    assert "LOST 1 message(s) between 4000000 and 6000000" in err
    assert rep["topics"][topic]["lost"] == 1 and rep["topics"][topic]["gaps"] == [[4 * MS, 6 * MS]]


def test_duplicate_fails(tmp_path):
    rc, out, err, rep, topic = run(tmp_path, [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10])
    assert rc == 1 and "duplicate stamp 3000000" in err
    assert rep["topics"][topic]["duplicates"] == 1


def test_out_of_order_fails(tmp_path):
    rc, out, err, rep, topic = run(tmp_path, [0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10])
    assert rc == 1 and "out of order" in err
    assert rep["topics"][topic]["out_of_order"] == 1


def test_irregular_interval_fails(tmp_path):
    rc, out, err, rep, topic = run(tmp_path, [0, 1, 2, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10])
    assert rc == 1 and "irregular interval" in err


def test_late_start_fails(tmp_path):
    rc, out, err, rep, topic = run(tmp_path, range(3, 11))
    assert rc == 1 and "the start was missed" in err
    assert rep["topics"][topic]["late_start"] is True


def test_missing_first_message_fails(tmp_path):
    """With a paused start the first stamp is known exactly: losing only the t = 0
    message (first stamp = one period) must still fail."""
    rc, out, err, rep, topic = run(tmp_path, range(1, 11))
    assert rc == 1 and "first stamp 1000000 ns != expected 0 ns" in err
    assert rep["topics"][topic]["late_start"] is True


def test_paused_repeats_of_zero_are_counted_not_errors(tmp_path):
    rc, out, err, rep, topic = run(tmp_path, [0, 0, 0, *range(1, 11)])
    assert rc == 0, err
    t = rep["topics"][topic]
    assert (t["count"], t["paused_repeats"]) == (11, 2)
    assert S.read_bag(str(tmp_path / "bag"))[topic]["stamps"] == list(range(0, END + 1, MS))


def test_zero_after_start_is_still_an_error(tmp_path):
    rc, out, err, rep, topic = run(tmp_path, [0, 1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert rc == 1 and "out of order" in err


def test_incomplete_on_interrupt_fails(tmp_path):
    rc, out, err, rep, topic = run(tmp_path, range(0, 6), interrupt=True)
    assert rc == 1 and "incomplete" in err
    assert rep["topics"][topic]["last_ns"] == 5 * MS


def test_backlog_cap_aborts_instead_of_dropping(tmp_path):
    rc, out, err, rep, topic = run(tmp_path, range(0, 11), "--max-backlog-mb", "0")
    assert rc == 1 and "ABORTING" in err
    assert rep["aborted"] is True


def test_no_publisher_is_fatal(tmp_path):
    p = start(tmp_path, f"/nobody_{uuid.uuid4().hex[:8]}", "--connect-timeout-s", "2")
    out, err = p.communicate(timeout=60)
    assert p.returncode == 2 and "no publisher discovered" in err and "CONNECTED" not in out


def test_unsupported_type_pair_is_fatal(tmp_path):
    p = start(tmp_path, "/x", pair=("gz.msgs.Clock", "sensor_msgs/msg/Imu"))
    out, err = p.communicate(timeout=60)
    assert p.returncode == 2 and "unsupported type pair" in err
