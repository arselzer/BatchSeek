"""HOST-only: ab_requests.py's AB_REFILL pool (ab_window.sh's CrG part), no server, model or GPU.

one() is replaced by a stub each request of which ends only when the test releases it, so the pool's
sends can be checked step by step: at most K in flight, the next ones sent only once G slots are
free, in the receipt's order, every reply kept in place; and the waves (no AB_REFILL) unchanged.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_requests


class Refill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ab-refill-")
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "arm.json")
        self.mu = threading.Condition()
        self.started, self.release, self.live, self.peak = [], set(), 0, 0

    def receipt(self, n, turns=None):
        reqs = []
        for x in range(n):
            r = {"prompt": [x], "temperature": 0, "max_tokens": 2, "stream": False}
            if turns is not None:
                r = {"chat_turns": turns, "temperature": 0, "max_tokens": 2}
            reqs.append({"name": "q-%d" % x, "family": "q", "request": r})
        path = os.path.join(self.tmp.name, "r.json")
        with open(path, "w") as f:
            json.dump({"benchmark_requests": reqs}, f)
        return path

    def one(self, b, url, replies=None, timeout=600.0):
        i = int(b["name"].split("-")[1])
        with self.mu:
            self.started.append(i)
            self.live += 1
            self.peak = max(self.peak, self.live)
            self.mu.notify_all()
            while i not in self.release:
                self.mu.wait()
            self.live -= 1
        return {"name": b["name"], "family": "q", "wall_s": 0.0, "text": "reply-%d" % i,
                "timings": {"predicted_n": i + 1}, "error": None}

    def stats(self, url, timeout):
        return io.BytesIO(b'{"steps_decode": 1}')

    def env(self, k, g):
        e = {"AB_CONCURRENCY": str(k), "AB_REQUEST_TIMEOUT_S": "", "AB_MAX_TOKENS": ""}
        if g is not None:
            e["AB_REFILL"] = g
        return e

    def start(self, path, n, k, g):
        def go():
            with mock.patch.dict(os.environ, self.env(k, g)), \
                 mock.patch.object(ab_requests, "one", side_effect=self.one), \
                 mock.patch.object(ab_requests.urllib.request, "urlopen", side_effect=self.stats), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.result = ab_requests.run(path, "http://stub", n, self.out)
        t = threading.Thread(target=go)
        t.start()
        return t

    def wait_started(self, want):
        end = time.time() + 5
        with self.mu:
            while sorted(self.started) != want and time.time() < end:
                self.mu.wait(0.05)
            self.assertEqual(sorted(self.started), want)

    def free(self, *ids):
        with self.mu:
            self.release.update(ids)
            self.mu.notify_all()
        time.sleep(0.2)   # a pool that would send now has had the time to

    def test_pool_refills_only_when_g_slots_are_free(self):
        t = self.start(self.receipt(8), 8, 4, "2")
        self.wait_started([0, 1, 2, 3])
        self.free(0)                        # one slot free: under G=2, nothing sent
        self.wait_started([0, 1, 2, 3])
        self.free(2)                        # two free: the next two, in order
        self.wait_started([0, 1, 2, 3, 4, 5])
        self.free(1, 3, 4, 5)
        self.wait_started(list(range(8)))
        self.free(6, 7)
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertLessEqual(self.peak, 4)
        with open(self.out) as f:
            j = json.load(f)
        self.assertEqual([r["text"] for r in j["requests"]], ["reply-%d" % i for i in range(8)])
        self.assertEqual((j["refill"], j["concurrency"], len(j["waves"])), (2, 4, 1))
        self.assertEqual((j["waves"][0]["tokens"], j["waves"][0]["refill"]), (36, 2))
        self.assertTrue(all(r["sent_s"] >= 0 for r in j["requests"]))
        self.assertGreater(j["requests"][4]["sent_s"], j["requests"][3]["sent_s"])

    def test_g1_sends_as_soon_as_one_ends(self):
        t = self.start(self.receipt(5), 5, 3, "1")
        self.wait_started([0, 1, 2])
        self.free(1)
        self.wait_started([0, 1, 2, 3])
        self.free(0, 2, 3, 4)
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertLessEqual(self.peak, 3)

    def test_without_refill_the_waves_are_unchanged(self):
        t = self.start(self.receipt(4), 4, 2, None)
        self.wait_started([0, 1])
        self.free(0)                        # a wave waits for its last request
        self.wait_started([0, 1])
        self.free(1)
        self.wait_started([0, 1, 2, 3])
        self.free(2, 3)
        t.join(5)
        with open(self.out) as f:
            j = json.load(f)
        self.assertEqual(set(j), {"requests", "waves", "concurrency", "request_timeout_s", "stats"})
        self.assertEqual([w["tokens"] for w in j["waves"]], [3, 7])
        self.assertTrue(all("sent_s" not in r for r in j["requests"]))

    def test_refused_settings_send_nothing(self):
        for g in ("5", "0", "x"):
            with self.subTest(g=g), mock.patch.dict(os.environ, self.env(4, g)), \
                 mock.patch.object(ab_requests, "one", side_effect=self.one), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    ab_requests.run(self.receipt(4), "http://stub", 4, self.out)
        turns = [{"user": "a"}, {"assistant_from": "q-0"}, {"user": "b"}]
        with mock.patch.dict(os.environ, self.env(4, "1")), \
             mock.patch.object(ab_requests, "one", side_effect=self.one), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as e:
                ab_requests.run(self.receipt(2, turns), "http://stub", 2, self.out)
        self.assertIn("assistant_from", str(e.exception))
        self.assertEqual(self.started, [])


if __name__ == "__main__":
    unittest.main()
