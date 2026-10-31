#!/usr/bin/env python3
"""tools/fn_sweep.py's flag tables and --no-mtp, device-free.

The campaign's T1 leg needs a run with no MTP head: a trace never has a 1-row
round (rows = 1 + drafts) and --llama-mtp 0 writes no trace, so the MTP-off
sweep's decode t/s is the only measured ms a token for width 1. --print-flags
prints the argv the arm would launch and exits, so the table is checkable
without a card.

Every cell was red first:
  - the MTP flags stay in the default table (red: --no-mtp's block edited to
    drop only --llama-mtp, or the block removed, leaves the other three);
  - --no-mtp drops all four names AND their values (red: removing the name but
    not the value leaves a dangling value in the argv);
  - a --set after the unit's table does not survive --no-mtp (red: the drop
    moved before the --set loop, the override comes back);
  - the coder table carries the draft vocabulary the card runs used (red: the
    entry removed, coder's argv loses --llama-mtp-vocab while the service file
    and the fixed-width traces both have it).
"""
import subprocess
import sys
import unittest

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
SCRIPT = str(REPO_ROOT / "tools" / "fn_sweep.py")

MTP = ("--llama-mtp", "--llama-mtp-min-p", "--llama-mtp-gguf", "--llama-mtp-vocab")


def flags(*extra):
    r = subprocess.run([sys.executable, SCRIPT, "x", "y", "z", "t", "--unit", "flashnext",
                        "--print-flags"] + list(extra), capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(r.stderr)
    return r.stdout.split()


class FlagTable(unittest.TestCase):
    def test_the_coder_table_carries_the_draft_vocabulary(self):
        # red: the coder entry removed, the arm drafts from all 248,320 head rows
        # while the served unit and the campaign's traces have the vocabulary
        r = subprocess.run([sys.executable, SCRIPT, "x", "y", "z", "t", "--unit", "coder",
                            "--print-flags"], capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(r.stderr)
        f = r.stdout.split()
        self.assertIn("--llama-mtp-vocab", f)
        self.assertEqual(f[f.index("--llama-mtp-vocab") + 1], "z/mtp-vocab.json")
    def test_the_default_table_carries_the_mtp_flags(self):
        f = flags()
        for name in MTP:
            self.assertIn(name, f)
        self.assertEqual(f[f.index("--llama-mtp") + 1], "3")

    def test_no_mtp_drops_every_mtp_flag_and_its_value(self):
        # red: dropping the name but keeping its value leaves a dangling "3"
        f = flags("--no-mtp")
        for name in MTP:
            self.assertNotIn(name, f)
        self.assertNotIn("3", f)          # --llama-mtp's value
        self.assertNotIn("0.5", f)        # --llama-mtp-min-p's value
        self.assertNotIn("z/mtp-flash-next.gguf", f)
        self.assertNotIn("z/mtp-vocab.json", f)

    def test_a_set_override_does_not_survive_no_mtp(self):
        # red: the drop before the --set loop, the override's flag comes back
        f = flags("--set", "--llama-mtp", "6", "--no-mtp")
        self.assertNotIn("--llama-mtp", f)
        self.assertNotIn("6", f)


if __name__ == "__main__":
    unittest.main()
