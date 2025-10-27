# graphix-mentpy-interface
Methods to allow MBQC patterns in Graphix to be translated to MBQCircuits in MentPy.

Code currently works, but overloads memory due to recursive flow-finding algorithm used in generating the Graphix pattern.
Next step is to force the functions to only use gflow to generate the pattern.