.PHONY: all test figures
all: test
test:                ## invariants + validation-against-anchors
	cd sim && python3 test_reliability_sim.py
figures:             ## phase diagram, crossover, checkpoint optimum, validation
	cd sim && python3 run.py
