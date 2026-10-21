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


def classify(off, on):
    """What one arm pair says. Pure, so a wrong verdict is testable without a
    card: the gate's whole value is that it can say FAILED and NOT-PASSED."""
    off_bad = off.startswith("EMPTY-STOP")
    on_bad = on.startswith("EMPTY-STOP")
    if off_bad:
        # The defect appeared without the mask. Did the mask clear it? Note the
        # mask may clear it into an answer OR into a cap-ended empty; only
        # another empty STOP means it survived.
        return "red" if on_bad else "fixed"
    if on_bad:
        return "flipped"        # empty only WITH the mask: the opposite defect
    # Neither side ended empty at the turn-ending token. An empty the token cap
    # ended is a different defect (the cap, not the token winning a step), so
    # this arm carries no information either way.
    return "neither"


def main():
    argv = list(sys.argv[1:])
    flag_on = "--flag-on" in argv
    argv = [a for a in argv if a != "--flag-on"]
    base = argv[0] if argv else "http://localhost:8080"
    model = argv[1] if len(argv) > 1 else "qwen3.6-coder"

    tally = {"fixed": 0, "red": 0, "neither": 0, "flipped": 0}
    for name, prompt in probe.SHAPES.items():
        for temp in (0.0, 0.7):
            verdicts = {}
            for label, extra in ARMS:
                verdicts[label] = probe.run(
                    base, model, "%s T=%.1f mask=%s" % (name, temp, label),
                    prompt, extra, max_tokens=4096)
            tally[classify(verdicts["off"], verdicts["on"])] += 1
    fixed, red, neither, flipped = (tally["fixed"], tally["red"],
                                   tally["neither"], tally["flipped"])

    print("\n%d arms provoked an empty stop without the mask; %d of those "
          "continue with it; %d stayed empty both ways; %d did not provoke it"
          % (fixed + red, fixed, red, neither))
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
