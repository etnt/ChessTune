# Variables
VENV_NAME := venv
PYTHON := python3
PIP := $(VENV_NAME)/bin/pip
REQUIREMENTS := requirements.txt

# Default target
.PHONY: all
all: venv

# Create virtual environment and install requirements
.PHONY: venv
venv: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: $(REQUIREMENTS)
	$(PYTHON) -m venv $(VENV_NAME)
	$(PIP) install -r $(REQUIREMENTS)
	touch $(VENV_NAME)/bin/activate

# Clean up
.PHONY: clean
clean:
	rm -rf $(VENV_NAME)

# Run the Client
.PHONY: categorize
categorize: venv
	$(VENV_NAME)/bin/python categorize-lux-tests.py

.PHONY: group
group: venv
	$(VENV_NAME)/bin/python group-tests.py

# Run tests
.PHONY: test
test: venv
	$(VENV_NAME)/bin/python -m unittest discover -v

# Install a new package and add it to requirements.txt
.PHONY: add
add:
	@read -p "Enter package name: " package; \
	$(PIP) install $$package && $(PIP) freeze | grep -i $$package >> $(REQUIREMENTS)

# Summarize tests
.PHONY: summarize-tests
summarize-tests: venv
	$(eval EXTRA_ARGS ?=)
	$(VENV_NAME)/bin/python summarize_lux_tests.py $(LUX_TESTS_DIR) $(EXTRA_ARGS)
