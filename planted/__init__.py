"""Planted bugs: prove a speech-to-text evaluation gate can fail.

Plant errors of a known kind and size into good model outputs, run the gate, and check it turns red
and finds exactly what was planted. A clean control must stay green. The smallest planted regression
the gate still catches is the smallest real regression it can see.
"""
