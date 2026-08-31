.PHONY: all test validation figures
all: test
test: validation        ## invariants + the validation registry
	cd sim && python3 test_reliability_sim.py
validation:          ## model vs the public record, plus its negative controls
	cd sim && python3 test_validation.py && python3 validation.py
figures:             ## phase diagram, crossover, checkpoint optimum, validation
	cd sim && python3 run.py
