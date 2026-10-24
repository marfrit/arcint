#!/usr/bin/env python3
"""clean-turns step 1 gate: can the fix be shown to fail and to work?

The campaign's gate is a pair, not a count: on the served unit the same arm
must end EMPTY without the mask and CONTINUE with it. This script runs that
pair against one already-running unit and says which side failed.

Off and on are the two states of the SAME request field (`think_eog_mask`),
so one serving process serves both arms and the unit is loaded once. A unit
started with the operator flag instead is also fine - then the off arm has to
be run against a unit without the flag, and `--flag-on` says so, which changes
what a failure means (see the note it prints).

Prompts name the control tokens in WORDS only; every token string is
assembled from parts at run time, so this file contains none. Prints flags
and counts only - no prompt, no answer text.

Run ON the card host against the served unit. The card procedure
(docs/sop-card-window.md) is the operator's: sampler before the leg, card by
PCI id, zombie sweep after. This script only talks HTTP.
"""
import sys

import think_eog_probe as probe

ARMS = (("off", None), ("on", {"think_eog_mask": True}))


def is_defect(res):
    """The defect, not just an empty stop.

    The defect is a turn-ending token winning a step while the think block is
    still open. An empty stop AFTER the block closed is a legitimate empty
    answer (the model thought, closed the block, and chose to say nothing),
    which the mask is allowed to permit. The two look identical to a client
    that cannot see the block's state, so the verdict rests on `closed`.
    When the unit does not report it (an older binary), an empty stop is read
    as the defect - the conservative call - and the gate says so.
    """
    if res["verdict"] != "EMPTY-STOP":
        return False
    if res["closed"] is None:
        return True
    return res["closed"] is False


def classify(off, on):
    """What one arm pair says. Pure, so a wrong verdict is testable without a
    card: the gate's whole value is that it can say FAILED and NOT-PASSED."""
    off_bad = is_defect(off)
    on_bad = is_defect(on)
    if off_bad:
        # The defect appeared without the mask. Did the mask clear it? The on
        # arm is cleared when it no longer ends mid-block: it may continue into
        # an answer, into a cap-ended empty, or into a closed-block empty
        # answer - only another mid-block empty stop means it survived.
        return "red" if on_bad else "fixed"
    if on_bad:
        return "flipped"        # the defect appears only WITH the mask
    # Neither side ended the turn mid-block. An empty the token cap ended, or
    # an empty answer that closed the block, is a different outcome, so this
    # arm carries no information either way.
    return "neither"


def main():
    argv = list(sys.argv[1:])
    flag_on = "--flag-on" in argv
    argv = [a for a in argv if a != "--flag-on"]
    shapes = [a.split("=", 1)[1] for a in argv if a.startswith("--shapes=")]
    argv = [a for a in argv if not a.startswith("--shapes=")]
    base = argv[0] if argv else "http://localhost:8080"
    model = argv[1] if len(argv) > 1 else "qwen3.6-coder"

    # A card window is time-boxed, so the run may name the shapes. Naming them
    # narrows what a pass means, and the line below says so; NOT SHOWN stays
    # reachable, which is the point of the script.
    chosen = {n: p for n, p in probe.SHAPES.items() if not shapes or n in shapes}
    unknown = [n for n in shapes if n not in probe.SHAPES]
    if unknown:
        print("unknown shape(s): %s; known: %s" % (",".join(unknown), ",".join(probe.SHAPES)))
        return 2
    if shapes and len(chosen) < len(probe.SHAPES):
        print("shapes run: %s of %d (a narrowed run shows those shapes only)"
              % (",".join(sorted(chosen)), len(probe.SHAPES)))

    tally = {"fixed": 0, "red": 0, "neither": 0, "flipped": 0}
    no_closed = 0
    for name, prompt in chosen.items():
        for temp in (0.0, 0.7):
            verdicts = {}
            for label, extra in ARMS:
                verdicts[label] = probe.run(
                    base, model, "%s T=%.1f mask=%s" % (name, temp, label),
                    prompt, extra, max_tokens=4096)
            if verdicts["off"]["closed"] is None or verdicts["on"]["closed"] is None:
                no_closed += 1
            tally[classify(verdicts["off"], verdicts["on"])] += 1
    fixed, red, neither, flipped = (tally["fixed"], tally["red"],
                                   tally["neither"], tally["flipped"])

    print("\n%d arms ended the turn mid-block without the mask; %d of those "
          "stayed closed or answered with it; %d stayed mid-block both ways; "
          "%d did not end mid-block" % (fixed + red, fixed, red, neither))
    if no_closed:
        print("Note: %d arm(s) came from a unit that does not report the block's "
              "closed state, so an empty stop was read as the defect (the "
              "conservative call). Re-run against a unit that reports "
              "reasoning_closed to separate a mid-block stop from a closed-block "
              "empty answer." % no_closed)
    if flipped:
        print("GATE INVALID: %d arm(s) went empty only WITH the mask - the "
              "mask itself ends turns, which is the opposite defect." % flipped)
        return 1
    if red:
        print("GATE FAILED: the mask does not prevent the empty stop.")
        return 1
    if fixed == 0:
        print("GATE NOT SHOWN: no arm provoked the empty stop on this unit, so "
              "nothing was demonstrated either way. This is not a pass.")
        return 1
    print("GATE PASSED: the red case fails without the switch and passes with it.")
    if flag_on:
        print("Note: run with --flag-on means the unit itself was started with "
              "the operator flag, so the off arm here is a request asking the "
              "field off. That pair shows the field is honoured per request; "
              "the default-off claim needs a unit started WITHOUT the flag.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
