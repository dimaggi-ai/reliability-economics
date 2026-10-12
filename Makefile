.PHONY: all test validation figures research-claims
all: test
test: validation        ## invariants + the validation registry
	cd sim && python3 test_reliability_sim.py
	$(MAKE) research-claims
research-claims:         ## reproduced artifacts and economic prose must agree
	python3 tools/check_research_claims.py
	python3 -m unittest discover -s tools -p test_research_claims.py
validation:          ## model vs the public record, plus its negative controls
	cd sim && python3 test_validation.py && python3 validation.py
figures:             ## phase diagram, crossover, checkpoint optimum, validation
	cd sim && python3 run.py
	python3 sim/recovery_sensitivity.py
	python3 tools/check_research_claims.py --write
