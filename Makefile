# Makefile (for macOS/Linux)
# This Makefile sets up a virtual environment and supports testing and notebook usage.

ENV_NAME=venv

install:
	# Create virtual environment named 'venv'
	python3 -m venv $(ENV_NAME)
	# Upgrade pip and install dependencies from requirements.txt
	$(ENV_NAME)/bin/pip install --upgrade pip
	$(ENV_NAME)/bin/pip install -r requirements.txt
	# Register the environment as a Jupyter kernel
	$(ENV_NAME)/bin/python -m ipykernel install --user --name=$(ENV_NAME) --display-name="Python ($(ENV_NAME))"

notebook:
	# Launch Jupyter Notebook using venv's kernel
	$(ENV_NAME)/bin/jupyter notebook

freeze:
	# Save current package versions to requirements.txt
	$(ENV_NAME)/bin/pip freeze > requirements.txt

clean:
	# Delete all __pycache__ directories
	find . -type d -name '__pycache__' -exec rm -r {} +

test:
	# Run model loading test in venv
	$(ENV_NAME)/bin/python test_model.py



