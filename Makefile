df = $$(if [ -d $(PWD)/'.venv' ]; then echo $(PWD)/".venv/bin/docformatter"; else echo "docformatter"; fi)
src_py = $(shell find src -name "*.py")
test_py = $(shell find src -name "*.py")

lint:
	flake8 --exit-zero $(src_py) $(test_py)
	$(df) --check $(src_py) $(test_py)

format:
	black $(src_py) $(test_py)
	$(df) --in-place $(src_py) $(test_py)
