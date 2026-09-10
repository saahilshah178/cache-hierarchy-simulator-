"""Tests for the parameter sweep command.

Sweep rows are checked against hand-computable expectations on the aliasing
workloads, and against the stack-distance profiler where the two must agree
exactly: every LRU row of an associativity sweep is a point on the per-set
miss curve of the same geometry.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import tempfile
import unittest
from collections.abc import Iterable

from cachesim import plot
from cachesim.config import CacheSpec
from cachesim.stackdist import per_set_profile
from cachesim.sweep import (
    ASSOCIATIVITIES,
    DEFAULT_VALUES,
    SweepRow,
    main,
    non_monotonic_pairs,
    parse_size,
    parse_values,
    run_single_level,
    sweep,
    sweep_associativity,
    sweep_grid,
    sweep_size,
    write_csv,
)
from cachesim.workloads import conflict_streams, matmul, write_trace


def as_accesses(gen: Iterable[tuple[int, str]]) -> list[tuple[int, bool]]:
    return [(addr, op == "W") for addr, op in gen]


def base_spec(size: int = 8192, block_size: int = 64, associativity: int = 4) -> CacheSpec:
    return CacheSpec(
        name="L1",
        size=size,
        block_size=block_size,
        associativity=associativity,
        hit_time=4,
        policy="lru",
    )


def rates(rows: list[SweepRow]) -> dict[int | str, float]:
    return {row.value: row.miss_rate for row in rows}


class TestSweepEngine(unittest.TestCase):
    def test_associativity_knee_on_conflict_streams(self) -> None:
        """Four aliasing streams: 100% miss below 4-way, one miss per block after."""
        trace = as_accesses(conflict_streams(words_per_stream=1024))
        rows = sweep(trace, base_spec(), "associativity", list(ASSOCIATIVITIES))
        measured = rates(rows)
        self.assertEqual(measured[1], 1.0)
        self.assertEqual(measured[2], 1.0)
        self.assertEqual(measured[4], 0.125)  # 1 miss per 8 words of each block
        self.assertEqual(measured[8], 0.125)
        self.assertEqual(measured[16], 0.125)

    def test_size_sweep_returns_one_row_per_size(self) -> None:
        trace = as_accesses(conflict_streams(words_per_stream=256))
        rows = sweep(trace, base_spec(), "size", list(DEFAULT_VALUES["size"]))
        self.assertEqual(
            [row.value for row in rows], [kb * 1024 for kb in (1, 2, 4, 8, 16, 32, 64)]
        )
        for row in rows:
            self.assertEqual(row.miss_rate, 0.125)
            self.assertEqual(row.associativity, 4)

    def test_rows_agree_with_the_stack_distance_profile(self) -> None:
        """An LRU associativity sweep is the per-set miss curve, exactly."""
        trace = as_accesses(matmul(n=16))
        blocks = [addr // 64 for addr, _ in trace]
        for size in (4096, 16384):
            rows = sweep(trace, base_spec(size=size), "associativity", [1, 2, 4, 8])
            for row in rows:
                with self.subTest(size=size, ways=row.value):
                    profile = per_set_profile(blocks, row.num_sets)
                    self.assertEqual(row.misses, profile.misses(row.associativity))
                    self.assertEqual(row.compulsory, profile.infinite)

    def test_three_c_columns_add_up(self) -> None:
        trace = as_accesses(conflict_streams(words_per_stream=256))
        for row in sweep(trace, base_spec(), "associativity", [1, 2, 4]):
            self.assertEqual(row.compulsory + row.capacity + row.conflict, row.misses)
            self.assertEqual(row.compulsory + row.capacity, row.shadow_misses)
            self.assertEqual(row.hits + row.misses, row.accesses)

    def test_policy_sweep_runs_every_policy(self) -> None:
        trace = as_accesses(conflict_streams(words_per_stream=256))
        rows = sweep(trace, base_spec(), "policy", ["lru", "fifo", "random"])
        self.assertEqual([row.value for row in rows], ["lru", "fifo", "random"])
        for row in rows:
            self.assertEqual(row.param, "policy")
            self.assertEqual(row.label, str(row.value))

    def test_block_size_sweep_cuts_compulsory_misses(self) -> None:
        """A linear scan touches fewer, larger blocks as the block grows."""
        trace = [(addr, False) for addr in range(0, 16 * 1024, 8)]
        rows = sweep(trace, base_spec(), "block_size", [16, 32, 64, 128])
        compulsory = [row.compulsory for row in rows]
        self.assertEqual(compulsory, [1024, 512, 256, 128])

    def test_amat_formula(self) -> None:
        trace = [(0, False), (0, False), (64, False), (64, False)]
        rows = sweep(trace, base_spec(size=1024, associativity=1), "size", [1024])
        self.assertEqual(rows[0].miss_rate, 0.5)
        self.assertEqual(rows[0].amat, 4 + 0.5 * 100)
        self.assertEqual(rows[0].misses, 2)

    def test_label_formats_sizes(self) -> None:
        trace = [(0, False)]
        rows = sweep(trace, base_spec(), "size", [1024, 32768])
        self.assertEqual([row.label for row in rows], ["1 KB", "32 KB"])

    def test_infeasible_geometry_names_the_parameter(self) -> None:
        trace = [(0, False)]
        with self.assertRaises(ValueError) as caught:
            sweep(trace, base_spec(), "associativity", [3])
        self.assertIn("associativity", str(caught.exception))

    def test_unknown_parameter_and_value_types(self) -> None:
        trace = [(0, False)]
        with self.assertRaises(ValueError):
            sweep(trace, base_spec(), "nonsense", [1])
        with self.assertRaises(ValueError):
            sweep(trace, base_spec(), "size", ["big"])
        with self.assertRaises(ValueError):
            sweep(trace, base_spec(), "policy", ["nope"])


class TestNonMonotonicity(unittest.TestCase):
    def test_monotonic_sweep_reports_nothing(self) -> None:
        trace = as_accesses(conflict_streams(words_per_stream=256))
        rows = sweep(trace, base_spec(), "associativity", list(ASSOCIATIVITIES))
        self.assertEqual(non_monotonic_pairs(rows), [])

    def test_power_of_two_stride_is_not_monotonic_in_associativity(self) -> None:
        """A naive 32x32 matmul in an 8 KB cache gets steadily WORSE with ways.

        The size is fixed, so adding ways removes sets: 8 KB of 64 B blocks
        is 128 sets direct-mapped but only 8 sets at 16-way. The column walk
        of B strides 256 B = 4 blocks, so its blocks land on every fourth
        set; with 128 sets they spread over 32 of them, with 8 sets over
        just 2. Concentrating the same aliasing into fewer sets costs more
        than the extra ways recover, and every step up is a regression.
        """
        trace = as_accesses(matmul(n=32))
        rows = sweep(trace, base_spec(size=8192), "associativity", [1, 2, 4, 8, 16])
        self.assertEqual([row.misses for row in rows], [3864, 4754, 6948, 10792, 19712])
        self.assertEqual([row.num_sets for row in rows], [128, 64, 32, 16, 8])
        pairs = non_monotonic_pairs(rows)
        self.assertEqual([(a.value, b.value) for a, b in pairs], [(1, 2), (2, 4), (4, 8), (8, 16)])

    def test_policy_sweeps_are_never_reported(self) -> None:
        trace = as_accesses(conflict_streams(words_per_stream=128))
        rows = sweep(trace, base_spec(), "policy", ["lru", "fifo", "random"])
        self.assertEqual(non_monotonic_pairs(rows), [])


class TestSweepGrid(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = as_accesses(conflict_streams(words_per_stream=256))

    def test_shape_and_infeasible_cells(self) -> None:
        grid = sweep_grid(self.trace, base_spec(), [1024, 4096], [1, 4, 16])
        self.assertEqual(grid.sizes, (1024, 4096))
        self.assertEqual(grid.associativities, (1, 4, 16))
        # 1 KB with 64 B blocks holds 16 blocks, so 16-way is one set: feasible.
        # 1 KB / (64 * 16) == 1, so every cell here exists.
        self.assertTrue(all(cell is not None for row in grid.rows for cell in row))
        # 512 B cannot be 16-way with 64 B blocks (needs 1024 B).
        narrow = sweep_grid(self.trace, base_spec(), [512], [1, 16])
        self.assertIsNotNone(narrow.rows[0][0])
        self.assertIsNone(narrow.rows[0][1])
        matrix = narrow.miss_rate_matrix()
        self.assertNotEqual(matrix[0][1], matrix[0][1])  # NaN

    def test_cells_match_a_single_sweep(self) -> None:
        grid = sweep_grid(self.trace, base_spec(), [4096, 16384], [1, 2, 4])
        for size, row in zip(grid.sizes, grid.rows, strict=True):
            direct = sweep(self.trace, base_spec(size=size), "associativity", [1, 2, 4])
            for cell, expected in zip(row, direct, strict=True):
                assert cell is not None
                self.assertEqual(cell.misses, expected.misses)
                self.assertEqual(cell.miss_rate, expected.miss_rate)

    def test_flat_rows_skips_infeasible_cells(self) -> None:
        grid = sweep_grid(self.trace, base_spec(), [512, 4096], [1, 16])
        self.assertEqual(len(grid.flat_rows()), 3)
        self.assertTrue(all(row.param == "associativity" for row in grid.flat_rows()))
        self.assertEqual([row.value for row in grid.flat_rows()], [1, 1, 16])

    def test_non_monotonic_rows_are_found(self) -> None:
        """The 8 KB row of a naive 32x32 matmul worsens at every step."""
        trace = as_accesses(matmul(n=32))
        grid = sweep_grid(trace, base_spec(), [8192, 16384], [1, 2, 4])
        found = grid.non_monotonic_rows()
        self.assertEqual(
            [(size, a.value, b.value) for size, a, b in found], [(8192, 1, 2), (8192, 2, 4)]
        )


class TestCsv(unittest.TestCase):
    def test_csv_round_trips_every_counter(self) -> None:
        trace = as_accesses(conflict_streams(words_per_stream=256))
        rows = sweep(trace, base_spec(), "associativity", [1, 4])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "sweep.csv")
            write_csv(path, rows)
            with open(path, newline="") as handle:
                written = list(csv.DictReader(handle))
        self.assertEqual(len(written), 2)
        for record, row in zip(written, rows, strict=True):
            self.assertEqual(int(record["misses"]), row.misses)
            self.assertEqual(int(record["compulsory"]), row.compulsory)
            self.assertEqual(int(record["conflict"]), row.conflict)
            self.assertEqual(int(record["num_sets"]), row.num_sets)
            self.assertAlmostEqual(float(record["miss_rate"]), row.miss_rate, places=6)


class TestParsing(unittest.TestCase):
    def test_parse_size_suffixes(self) -> None:
        self.assertEqual(parse_size("512"), 512)
        self.assertEqual(parse_size("4k"), 4096)
        self.assertEqual(parse_size("4KB"), 4096)
        self.assertEqual(parse_size("1M"), 1 << 20)

    def test_parse_size_rejects_junk(self) -> None:
        for bad in ("", "big", "0", "-4k", "4g"):
            with self.subTest(text=bad), self.assertRaises(argparse.ArgumentTypeError):
                parse_size(bad)

    def test_parse_values(self) -> None:
        self.assertEqual(parse_values("size", "1k, 2k,4k"), [1024, 2048, 4096])
        self.assertEqual(parse_values("associativity", "1,2,16"), [1, 2, 16])
        self.assertEqual(parse_values("policy", "LRU,fifo"), ["lru", "fifo"])

    def test_parse_values_rejects_unknown_policy(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_values("policy", "lru,nope")


class TestLegacyWrappers(unittest.TestCase):
    """The pre-rewrite helpers still work; they now return SweepRow."""

    def test_run_single_level_amat(self) -> None:
        trace = [(0, False), (0, False), (64, False), (64, False)]
        miss_rate, amat, cache = run_single_level(trace, 1024, 64, 1, "lru", 4, 100)
        self.assertEqual(miss_rate, 0.5)
        self.assertEqual(amat, 4 + 0.5 * 100)
        self.assertEqual(cache.misses, 2)

    def test_sweep_associativity_wrapper(self) -> None:
        trace = as_accesses(conflict_streams(words_per_stream=1024))
        with contextlib.redirect_stdout(io.StringIO()):
            rows = sweep_associativity(trace, 8192, 64, "lru", 4, 100)
        self.assertEqual([row.value for row in rows], list(ASSOCIATIVITIES))
        self.assertEqual(rates(rows)[4], 0.125)

    def test_sweep_size_wrapper_uses_bytes(self) -> None:
        trace = as_accesses(conflict_streams(words_per_stream=256))
        with contextlib.redirect_stdout(io.StringIO()):
            rows = sweep_size(trace, 4, 64, "lru", 4, 100)
        self.assertEqual(
            [row.value for row in rows], [kb * 1024 for kb in (1, 2, 4, 8, 16, 32, 64)]
        )
        for row in rows:
            self.assertEqual(row.miss_rate, 0.125)


class TestSweepCLI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.trace = os.path.join(self._tmp.name, "c.trace")
        write_trace(self.trace, conflict_streams(words_per_stream=256))

    def run_main(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as exc:
                code = int(exc.code) if exc.code is not None else 0
        return code, out.getvalue(), err.getvalue()

    def test_bad_flags_fail_with_a_message(self) -> None:
        for argv, fragment in [
            (["--size", "0", self.trace], "positive"),
            (["--assoc", "0", self.trace], "positive integer"),
            (["--block-size", "0", self.trace], "positive"),
            (["--hit-time", "-1", self.trace], "non-negative"),
            (["--size", "3000", self.trace], "not a multiple"),
            (["--assoc", "3", self.trace], "not a multiple"),
            (["--param", "size", "--values", "junk", self.trace], "not a size"),
            (["--param", "policy", "--values", "nope", self.trace], "unknown policy"),
            (["--param", "size", "--grid", self.trace], "mutually exclusive"),
            (["--ways", "1,2", self.trace], "only applies to --grid"),
            (["--param", "size"], "needs a trace"),
        ]:
            with self.subTest(argv=argv):
                code, _, err = self.run_main(argv)
                self.assertEqual(code, 2)
                self.assertIn(fragment, err)

    def test_missing_trace_is_reported(self) -> None:
        code, _, err = self.run_main([os.path.join(self._tmp.name, "nope.trace")])
        self.assertEqual(code, 2)
        self.assertIn("trace not found", err)

    def test_empty_trace_is_rejected(self) -> None:
        path = os.path.join(self._tmp.name, "empty.trace")
        with open(path, "w") as handle:
            handle.write("# nothing here\n")
        code, _, err = self.run_main([path])
        self.assertEqual(code, 2)
        self.assertIn("no accesses", err)

    def test_default_sweeps_print_both_tables(self) -> None:
        code, out, _ = self.run_main(["--size", "8192", "--no-plot", self.trace])
        self.assertEqual(code, 0)
        self.assertIn("associativity sweep", out)
        self.assertIn("size sweep", out)

    def test_single_parameter_sweep(self) -> None:
        code, out, _ = self.run_main(
            ["--param", "block_size", "--values", "32,64", "--no-plot", self.trace]
        )
        self.assertEqual(code, 0)
        self.assertIn("block_size sweep", out)
        self.assertIn("32 B", out)

    def test_grid_prints_a_matrix(self) -> None:
        code, out, _ = self.run_main(
            ["--grid", "--values", "1k,4k", "--ways", "1,16", "--no-plot", self.trace]
        )
        self.assertEqual(code, 0)
        self.assertIn("size x associativity grid", out)
        self.assertIn("16-way", out)

    def test_csv_flag_writes_a_file(self) -> None:
        path = os.path.join(self._tmp.name, "sweep.csv")
        code, out, _ = self.run_main(
            ["--param", "associativity", "--csv", path, "--no-plot", self.trace]
        )
        self.assertEqual(code, 0)
        self.assertIn(f"wrote {path}", out)
        with open(path, newline="") as handle:
            written = list(csv.DictReader(handle))
        self.assertEqual([int(r["associativity"]) for r in written], list(ASSOCIATIVITIES))

    def test_no_plot_writes_nothing(self) -> None:
        out_dir = os.path.join(self._tmp.name, "noplots")
        code, _, _ = self.run_main(["--no-plot", "--plot-dir", out_dir, self.trace])
        self.assertEqual(code, 0)
        self.assertFalse(os.path.exists(out_dir))

    def test_non_monotonic_note_is_printed(self) -> None:
        path = os.path.join(self._tmp.name, "m.trace")
        write_trace(path, matmul(n=32))
        code, out, _ = self.run_main(
            ["--param", "associativity", "--values", "1,2", "--size", "8192", "--no-plot", path]
        )
        self.assertEqual(code, 0)
        self.assertIn("not monotonic", out)


@unittest.skipUnless(plot.available(), "matplotlib is not installed")
class TestSweepPlots(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.trace = os.path.join(self._tmp.name, "c.trace")
        write_trace(self.trace, conflict_streams(words_per_stream=256))

    def run_main(self, argv: list[str]) -> int:
        with contextlib.redirect_stdout(io.StringIO()):
            return main(argv)

    def test_default_sweeps_write_two_plots(self) -> None:
        out_dir = os.path.join(self._tmp.name, "plots")
        self.assertEqual(self.run_main(["--size", "8192", "--plot-dir", out_dir, self.trace]), 0)
        self.assertEqual(
            sorted(os.listdir(out_dir)),
            ["miss_rate_vs_associativity_c.png", "miss_rate_vs_size_c.png"],
        )
        for name in os.listdir(out_dir):
            self.assertGreater(os.path.getsize(os.path.join(out_dir, name)), 0)

    def test_out_dir_is_still_accepted(self) -> None:
        """--out-dir is the pre-rewrite spelling of --plot-dir."""
        out_dir = os.path.join(self._tmp.name, "legacy")
        self.assertEqual(self.run_main(["--size", "8192", "--out-dir", out_dir, self.trace]), 0)
        self.assertEqual(
            sorted(os.listdir(out_dir)),
            ["miss_rate_vs_associativity_c.png", "miss_rate_vs_size_c.png"],
        )

    def test_grid_writes_a_heatmap(self) -> None:
        out_dir = os.path.join(self._tmp.name, "grid")
        self.assertEqual(
            self.run_main(["--grid", "--values", "1k,4k", "--plot-dir", out_dir, self.trace]), 0
        )
        self.assertEqual(os.listdir(out_dir), ["miss_rate_grid_c.png"])
        self.assertGreater(os.path.getsize(os.path.join(out_dir, "miss_rate_grid_c.png")), 0)

    def test_policy_sweep_plots_as_categories(self) -> None:
        path = os.path.join(self._tmp.name, "policy.png")
        rows = sweep(
            as_accesses(conflict_streams(words_per_stream=128)),
            base_spec(),
            "policy",
            ["lru", "fifo"],
        )
        plot.plot_sweep(rows, "policy", path)
        self.assertGreater(os.path.getsize(path), 0)

    def test_plot_grid_rejects_a_ragged_matrix(self) -> None:
        path = os.path.join(self._tmp.name, "bad.png")
        with self.assertRaises(ValueError):
            plot.plot_grid([[0.1, 0.2], [0.3]], ["a", "b"], ["1", "2"], path)
        with self.assertRaises(ValueError):
            plot.plot_grid([], [], [], path)

    def test_plot_sweep_rejects_empty_rows(self) -> None:
        with self.assertRaises(ValueError):
            plot.plot_sweep([], "size", os.path.join(self._tmp.name, "empty.png"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
