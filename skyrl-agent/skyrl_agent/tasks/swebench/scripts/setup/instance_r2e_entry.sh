#!/bin/bash

source ~/.bashrc

/testbed/.venv/bin/python -m ensurepip --default-pip

ln -s /testbed/.venv /root/.venv

ln -s /testbed/.venv/bin/python /root/.local/bin/python

ln -s /testbed/.venv/bin/python /root/.local/bin/python3

find "/testbed/.venv/bin" -type f -executable -exec ln -sf {} "/root/.local/bin/" \;

export PATH=/root/.local/bin:$PATH
export PATH=/testbed/.venv/bin:$PATH

# Install chardet
# uv pip install chardet
# /testbed/.venv/bin/python -m pip install chardet

# for the new search tool
# Install custom BM25 packages for Python 3.6 compatibility
/testbed/.venv/bin/python -m pip install chardet networkx
/testbed/.venv/bin/python -m pip install 'rank-bm25>=0.2.0,<1.0.0'
echo "Custom BM25 components installed successfully"

# Delete all *.pyc files and __pycache__ dirs in current repo
find . -name '*.pyc' -delete
find . -name '__pycache__' -exec rm -rf {} +

# Delete *.pyc files and __pycache__ in /r2e_tests
find /r2e_tests -name '*.pyc' -delete
find /r2e_tests -name '__pycache__' -exec rm -rf {} +

export REPO_PATH="/testbed"
REPO_PATH="/testbed" 
ALT_PATH="/root"
SKIP_FILES_NEW=("run_tests.sh" "r2e_tests")
for skip_file in "${SKIP_FILES_NEW[@]}"; do
    if [ -e "$REPO_PATH/$skip_file" ]; then
        mv "$REPO_PATH/$skip_file" "$ALT_PATH/$skip_file"
    fi
done

# Move /r2e_tests to ALT_PATH
mv /r2e_tests "$ALT_PATH/r2e_tests"

# Create symlink back to repo
ln -s "$ALT_PATH/r2e_tests" "$REPO_PATH/r2e_tests"
