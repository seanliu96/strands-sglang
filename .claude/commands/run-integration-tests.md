Run integration tests in an isolated temporary venv created with uv.

Integration tests require a running SGLang server. Before running any commands, ask the user:

1. "What is the SGLang server base URL?" — offer these options:
   - `http://localhost:30000` (default — assumes local or SSH-tunneled server)
   - Let the user provide a custom URL

2. "Which tool parser?" — offer these options:
   - `hermes` (default — for Hermes/Qwen models)
   - `qwen_xml` (XML format for Qwen models)
   - `glm` (GLM models)

3. "Include VLM tests?" — offer these options:
   - No (default — skip VLM tests, no torch needed)
   - Yes (install torch + torchvision CPU for VLM processor tokenization)

Then proceed:

1. Create a temporary venv at `/tmp/strands-sglang-test-venv` using `uv venv /tmp/strands-sglang-test-venv --python 3.12 -q`
2. Install the package with dev dependencies: `uv pip install -e ".[dev]" --python /tmp/strands-sglang-test-venv/bin/python -q`
3. If VLM tests requested, install torch CPU: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --python /tmp/strands-sglang-test-venv/bin/python -q`
4. Run integration tests with the confirmed URL and tool parser:
   - If VLM tests requested: `/tmp/strands-sglang-test-venv/bin/python -m pytest tests/integration/ -v --tb=short --sglang-base-url=<URL> --tool-parser=<PARSER> $ARGUMENTS`
   - If VLM tests NOT requested: `/tmp/strands-sglang-test-venv/bin/python -m pytest tests/integration/ -v --tb=short --sglang-base-url=<URL> --tool-parser=<PARSER> --ignore=tests/integration/test_vlm_integration.py $ARGUMENTS`

IMPORTANT: Never use the active shell's python/pytest — it may point to a different venv. Always use the temporary venv's python.
