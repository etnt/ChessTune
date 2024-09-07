# Variables
VENV_NAME := venv
PYTHON := python3
PIP := $(VENV_NAME)/bin/pip
REQUIREMENTS := requirements.txt
PGN_DIR := PGN
ZIPPED_PGNS := $(wildcard $(PGN_DIR)/*.pgn.gz)
UNZIPPED_PGNS := $(patsubst %.gz,%,$(ZIPPED_PGNS))

# Default target
.PHONY: all
all: venv unzip_pgns

# Create virtual environment and install requirements
.PHONY: venv
venv: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: $(REQUIREMENTS)
	$(PYTHON) -m venv $(VENV_NAME)
	$(PIP) install -r $(REQUIREMENTS)
	touch $(VENV_NAME)/bin/activate

# Unzip all PGN files
.PHONY: unzip_pgns
unzip_pgns: $(UNZIPPED_PGNS)

# Rule to unzip individual PGN files
$(PGN_DIR)/%.pgn: $(PGN_DIR)/%.pgn.gz
	@echo "Unzipping $<..."
	@gunzip -k $<

# Clean up
.PHONY: clean
clean:
	rm -rf $(VENV_NAME)
	rm -f $(UNZIPPED_PGNS)
	rm -rf ./chess_model

# Run the Client
.PHONY: run
run: venv unzip_pgns
	@if [ -z "$(MAX_GAMES)" ] && [ -z "$(RESUME_FROM)" ]; then \
		$(VENV_NAME)/bin/python chess-tune.py $(PGN_FILE); \
	elif [ -z "$(RESUME_FROM)" ]; then \
		$(VENV_NAME)/bin/python chess-tune.py $(PGN_FILE) --max-games $(MAX_GAMES); \
	elif [ -z "$(MAX_GAMES)" ]; then \
		$(VENV_NAME)/bin/python chess-tune.py $(PGN_FILE) --resume-from $(RESUME_FROM); \
	else \
		$(VENV_NAME)/bin/python chess-tune.py $(PGN_FILE) --max-games $(MAX_GAMES) --resume-from $(RESUME_FROM); \
	fi

# Run tests
.PHONY: test
test: venv
	$(VENV_NAME)/bin/python -m unittest discover -v

# Install a new package and add it to requirements.txt
.PHONY: add
add:
	@read -p "Enter package name: " package; \
	$(PIP) install $$package && $(PIP) freeze | grep -i $$package >> $(REQUIREMENTS)
