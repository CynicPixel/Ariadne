# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# ariadne-orchestrator (arch §25, design agentic.md D156-D160): the Bedrock tool loop and the MCP server over the
# deterministic core's decision requests. Pins: base by digest (the console's), every package exact.
# The repo is bind-mounted at /work (read-only); the run directory is mounted writable (decisions/, approvals/).
# Credentials come from the environment (live.sh exports the dev host's `aws login` session) or the instance role.
# ---------------------------------------------------------------------------
FROM python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
RUN pip install --no-cache-dir --root-user-action=ignore boto3==1.43.118 mcp==2.3.0 pytest==9.1.1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/work/python
WORKDIR /work
