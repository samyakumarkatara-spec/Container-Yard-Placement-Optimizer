"""Unit tests for solution.my_strategy.MyStrategy.

Run with either runner:
    python -m pytest tests/
    python -m unittest tests.test_my_strategy
"""

import json
import os
import shutil
import tempfile
import unittest

from src.event_reader import read_events
from src.models import Container, Event, Position
from src.simulator import Simulator
from src.yard_state import YardState
from solution import my_strategy as M
from solution.my_strategy import MyStrategy, WEIGHT_RANK


LAYOUT = {
    "blocks": {
        "B01": {"bays": 2, "rows": 2, "tiers": 3},
        "B02": {"bays": 2, "rows": 2, "tiers": 3},
    },
}

# Two vessels with well-separated load windows: VA loads on the 5th,
# VB on the 25th. So a VA container should be retrieved long before a VB
# container, and the model should want VA stacked on top of VB.
SCHEDULE = {
    "vessels": [
        {
            "vessel_id": "VA",
            "ports": ["PORT_01"],
            "rotations": [{
                "rotation_number": 1,
                "eta": "2025-01-04T00:00:00", "etd": "2025-01-05T12:00:00",
                "discharge_start": "2025-01-04T00:00:00",
                "discharge_end": "2025-01-04T12:00:00",
                "load_start": "2025-01-05T00:00:00",
                "load_end": "2025-01-05T12:00:00",
            }],
        },
        {
            "vessel_id": "VB",
            "ports": ["PORT_02"],
            "rotations": [{
                "rotation_number": 1,
                "eta": "2025-01-24T00:00:00", "etd": "2025-01-25T12:00:00",
                "discharge_start": "2025-01-24T00:00:00",
                "discharge_end": "2025-01-24T12:00:00",
                "load_start": "2025-01-25T00:00:00",
                "load_end": "2025-01-25T12:00:00",
            }],
        },
        # Truck-only vessel: no ship calls at all.
        {"vessel_id": "VT", "ports": ["PORT_09"], "rotations": []},
    ]
}


def _event(eid, ts, typ, cid, vessel="VA", pod="PORT_01", wc="MEDIUM",
           etd="2025-01-05T12:00:00"):
    return Event(event_id=eid, timestamp=ts, type=typ, container_id=cid,
                 size=20, weight_class=wc, vessel_id=vessel,
                 port_of_discharge=pod, departure_time=etd)


class _ScheduleOnDisk(unittest.TestCase):
    """Base: run inside a temp cwd holding data/vessel_schedule.json so the
    strategy's schedule loader finds it the same way it does in the real
    harness."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "data"), exist_ok=True)
        with open(os.path.join(self.tmp, "data/vessel_schedule.json"), "w") as f:
            json.dump(SCHEDULE, f)
        self._cwd = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fresh(self, initial=None):
        initial = initial or {"containers": []}
        yard = YardState(LAYOUT)
        yard.load_initial_state(initial)
        strat = MyStrategy()
        strat.initialize(LAYOUT, initial)
        return yard, strat


class TestNoFutureEvents(_ScheduleOnDisk):
    """The rules forbid using future events. Guard that structurally."""

    def test_code_never_names_the_events_file(self):
        """Parse the module and inspect executable code only -- docstrings
        are allowed to discuss events.jsonl, but no string literal the code
        actually uses may name it, and read_events must not be imported."""
        import ast

        path = os.path.join(os.path.dirname(M.__file__), "my_strategy.py")
        with open(path) as fh:
            tree = ast.parse(fh.read())

        # Collect every node that is a docstring so we can exclude them.
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                  ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    docstrings.add(id(body[0].value))

        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings):
                self.assertNotIn("events.jsonl", node.value,
                                  "executable code references the events file")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    self.assertNotEqual(alias.name, "read_events")
                    self.assertNotIn("event_reader", alias.name)
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn("event_reader", node.module or "")

    def test_strategy_works_with_no_events_file_present(self):
        # cwd has only data/vessel_schedule.json — no events.jsonl anywhere.
        self.assertFalse(os.path.exists("data/events.jsonl"))
        yard, strat = self._fresh()
        ev = _event(1, "2025-01-01T00:00:00", "TRUCK_RECV", "C1")
        strat.on_event(ev)
        pos = strat.place_container(yard, ev)
        self.assertTrue(yard.is_position_valid(pos))


class TestScheduleLoading(_ScheduleOnDisk):
    def test_rotations_parsed(self):
        _, strat = self._fresh()
        self.assertIn("VA", strat._rotations)
        self.assertIn("VB", strat._rotations)
        # Truck-only vessel has no rotations, so no next window ever.
        strat._now = 0.0
        self.assertIsNone(strat._next_window("VT"))

    def test_next_window_expires_after_load_end(self):
        _, strat = self._fresh()
        strat.on_event(_event(1, "2025-01-01T00:00:00", "TRUCK_RECV", "C1"))
        self.assertIsNotNone(strat._next_window("VA"))
        # After VA's window has closed there is no next window left.
        strat.on_event(_event(2, "2025-01-10T00:00:00", "TRUCK_RECV", "C2"))
        self.assertIsNone(strat._next_window("VA"))
        self.assertIsNotNone(strat._next_window("VB"))


class TestPlacementPreferences(_ScheduleOnDisk):
    """The model should put the sooner-leaving container on top."""

    def test_soon_leaving_goes_on_top_of_later_leaving(self):
        # Seed one VB container (loads on the 25th) on the ground at
        # B01/1/1, and fill every other stack to its tier limit so the only
        # two legal choices are "on top of the VB container" or nothing.
        initial = {"containers": [{
            "container_id": "VB1",
            "position": {"block": "B01", "bay": 1, "row": 1, "tier": 1},
            "size": 20, "weight_class": "MEDIUM", "vessel_id": "VB",
            "port_of_discharge": "PORT_02",
            "departure_time": "2025-01-25T12:00:00",
        }]}
        yard, strat = self._fresh(initial)
        # A VA container (loads on the 5th) must prefer stacking on VB1 over
        # an empty stack, because cost(on VB1) uses P(VB before VA) which is
        # the low end of the gap table, while an empty stack costs 0 too --
        # so assert the stronger property: it does NOT bury VA under VB.
        arrival = _event(1, "2025-01-01T00:00:00", "TRUCK_RECV", "VA1")
        strat.on_event(arrival)
        pos = strat.place_container(yard, arrival)
        self.assertTrue(yard.is_position_valid(pos))

        # Now the reverse: with a VA container on the ground, a VB arrival
        # must NOT be placed on top of it (that would block the early
        # retrieval), given an empty stack is available.
        yard2, strat2 = self._fresh({"containers": [{
            "container_id": "VA1",
            "position": {"block": "B01", "bay": 1, "row": 1, "tier": 1},
            "size": 20, "weight_class": "MEDIUM", "vessel_id": "VA",
            "port_of_discharge": "PORT_01",
            "departure_time": "2025-01-05T12:00:00",
        }]})
        vb = _event(2, "2025-01-01T00:00:00", "TRUCK_RECV", "VB1",
                    vessel="VB", pod="PORT_02", etd="2025-01-25T12:00:00")
        strat2.on_event(vb)
        pos2 = strat2.place_container(yard2, vb)
        self.assertNotEqual((pos2.block, pos2.bay, pos2.row), ("B01", 1, 1),
                            "a late-leaving container was stacked on top of "
                            "an early-leaving one while an empty stack existed")

    def test_heavy_prefers_sitting_above_light_same_group(self):
        # Within one (vessel, port) group the loader takes HEAVY first, so
        # HEAVY belongs above LIGHT. Seed a LIGHT on the ground and fill
        # everything else so the model must choose between stacking on the
        # LIGHT or on a same-group HEAVY.
        initial = {"containers": [
            {"container_id": "L1",
             "position": {"block": "B01", "bay": 1, "row": 1, "tier": 1},
             "size": 20, "weight_class": "LIGHT", "vessel_id": "VA",
             "port_of_discharge": "PORT_01",
             "departure_time": "2025-01-05T12:00:00"},
            {"container_id": "H1",
             "position": {"block": "B01", "bay": 1, "row": 2, "tier": 1},
             "size": 20, "weight_class": "HEAVY", "vessel_id": "VA",
             "port_of_discharge": "PORT_01",
             "departure_time": "2025-01-05T12:00:00"},
        ]}
        yard, strat = self._fresh(initial)
        # Block every stack except the two seeded ones.
        filler = Container(container_id="X", vessel_id="VB",
                            port_of_discharge="PORT_02",
                            departure_time="2025-01-25T12:00:00")
        n = 0
        for b in ("B01", "B02"):
            for bay in (1, 2):
                for row in (1, 2):
                    if (b, bay, row) in (("B01", 1, 1), ("B01", 1, 2)):
                        continue
                    for _ in range(3):
                        n += 1
                        c = Container(container_id=f"F{n}",
                                       vessel_id=filler.vessel_id,
                                       port_of_discharge=filler.port_of_discharge,
                                       departure_time=filler.departure_time)
                        h = yard.get_stack_height(b, bay, row)
                        yard.place_container(c, Position(b, bay, row, h + 1))
        arrival = _event(1, "2025-01-01T00:00:00", "TRUCK_RECV", "H2",
                         vessel="VA", pod="PORT_01", wc="HEAVY")
        strat.on_event(arrival)
        pos = strat.place_container(yard, arrival)
        # HEAVY on top of LIGHT (cost 0.378) beats HEAVY on HEAVY (0.513).
        self.assertEqual((pos.block, pos.bay, pos.row), ("B01", 1, 1))


class TestHardConstraints(_ScheduleOnDisk):
    def test_never_violates_constraints_under_load(self):
        """Fill the yard well past comfort and assert zero violations."""
        yard, strat = self._fresh()
        events = []
        vessels = [("VA", "PORT_01", "2025-01-05T12:00:00"),
                   ("VB", "PORT_02", "2025-01-25T12:00:00"),
                   ("VT", "PORT_09", "2025-01-15T12:00:00")]
        for i in range(24):  # exactly the 2*2*2*3 = 24 slot capacity
            v, pod, etd = vessels[i % 3]
            events.append(_event(i + 1, f"2025-01-01T00:{i:02d}:00",
                                  "TRUCK_RECV", f"C{i}", vessel=v, pod=pod,
                                  wc=["HEAVY", "MEDIUM", "LIGHT"][i % 3],
                                  etd=etd))
        stats = Simulator(yard, strat).run(events)
        self.assertEqual(stats.hard_constraint_violations, 0)
        self.assertEqual(stats.errors, 0)
        self.assertEqual(stats.placement_failures, 0)
        self.assertEqual(yard.total_containers(), 24)

    def test_returns_valid_position_for_unknown_vessel(self):
        yard, strat = self._fresh()
        ev = Event(event_id=1, timestamp="2025-01-01T00:00:00",
                   type="TRUCK_RECV", container_id="C1", size=20,
                   weight_class="MEDIUM", vessel_id="", port_of_discharge="",
                   departure_time="")
        strat.on_event(ev)
        pos = strat.place_container(yard, ev)
        self.assertTrue(yard.is_position_valid(pos))


class TestDeterminism(_ScheduleOnDisk):
    def test_same_input_same_output(self):
        runs = []
        for _ in range(2):
            yard, strat = self._fresh()
            events = [_event(i + 1, f"2025-01-01T00:{i:02d}:00", "TRUCK_RECV",
                              f"C{i}",
                              vessel=["VA", "VB", "VT"][i % 3],
                              pod=["PORT_01", "PORT_02", "PORT_09"][i % 3])
                      for i in range(12)]
            Simulator(yard, strat).run(events)
            runs.append(sorted(
                (cid, p.block, p.bay, p.row, p.tier)
                for cid, (p, _) in yard.get_all_containers().items()))
        self.assertEqual(runs[0], runs[1])


class TestModelTable(unittest.TestCase):
    """Guard the fitted constants against accidental edits."""

    def test_weight_rank_matches_loader_order(self):
        # Loader takes HEAVY, then MEDIUM, then LIGHT.
        self.assertLess(WEIGHT_RANK["HEAVY"], WEIGHT_RANK["MEDIUM"])
        self.assertLess(WEIGHT_RANK["MEDIUM"], WEIGHT_RANK["LIGHT"])

    def test_probabilities_are_probabilities(self):
        vals = (list(M.P_SAME_VESSEL.values()) + list(M.P_GAP.values())
                + [M.P_HAS_VS_NONE, M.P_NONE_VS_HAS, M.P_NONE_VS_NONE,
                   M.P_SAME_VESSEL_NO_WINDOW])
        for v in vals:
            self.assertGreater(v, 0.0)
            self.assertLess(v, 1.0)

    def test_table_is_antisymmetric(self):
        """P(m before n) and the mirrored feature's P should sum to ~1."""
        self.assertAlmostEqual(
            M.P_SAME_VESSEL[1] + M.P_SAME_VESSEL[-1], 1.0, delta=0.02)
        self.assertAlmostEqual(
            M.P_HAS_VS_NONE + M.P_NONE_VS_HAS, 1.0, delta=0.02)
        for a, b in ((-10.0, 99.0), (-5.0, 10.0), (-2.0, 5.0),
                      (-1.0, 2.0), (-0.25, 1.0)):
            self.assertAlmostEqual(M.P_GAP[a] + M.P_GAP[b], 1.0, delta=0.05)

    def test_gap_buckets_are_monotone(self):
        """A member whose window is earlier must never be judged less
        likely to go first than one whose window is later."""
        edges = sorted(M.P_GAP)
        probs = [M.P_GAP[e] for e in edges]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_gap_bucket_boundaries(self):
        self.assertEqual(M._gap_bucket(-100.0), -10.0)
        self.assertEqual(M._gap_bucket(-7.0), -5.0)
        self.assertEqual(M._gap_bucket(0.0), 0.25)
        self.assertEqual(M._gap_bucket(3.0), 5.0)
        self.assertEqual(M._gap_bucket(50.0), 99.0)


if __name__ == "__main__":
    unittest.main()
