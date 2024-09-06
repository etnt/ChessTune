# Variables
VENV_NAME := venv
PYTHON := python3
PIP := $(VENV_NAME)/bin/pip
REQUIREMENTS := requirements.txt
PGN_FILE := ficsgamesdb_202401_standard2000_nomovetimes_397466.pgn
ZIPPED_PGN := $(PGN_FILE).gz

# Default target
.PHONY: all
all: venv $(PGN_FILE)

# Create virtual environment and install requirements
.PHONY: venv
venv: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: $(REQUIREMENTS)
	$(PYTHON) -m venv $(VENV_NAME)
	$(PIP) install -r $(REQUIREMENTS)
	touch $(VENV_NAME)/bin/activate

# Unzip PGN file if it doesn't exist
$(PGN_FILE): $(ZIPPED_PGN)
	@if [ ! -f $(PGN_FILE) ]; then \
		echo "Unzipping $(ZIPPED_PGN)..."; \
		gunzip -k $(ZIPPED_PGN); \
	else \
		echo "$(PGN_FILE) already exists."; \
	fi

# Clean up
.PHONY: clean
clean:
	rm -rf $(VENV_NAME)
	rm -f $(PGN_FILE)

# Run the Client
.PHONY: run
run: venv $(PGN_FILE)
	$(VENV_NAME)/bin/python chess-tune.py

# Run tests
.PHONY: test
test: venv
	$(VENV_NAME)/bin/python -m unittest discover -v

# Install a new package and add it to requirements.txt
.PHONY: add
add:
	@read -p "Enter package name: " package; \
	$(PIP) install $$package && $(PIP) freeze | grep -i $$package >> $(REQUIREMENTS)
