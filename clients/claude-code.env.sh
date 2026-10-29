# Claude Code -> RunPod endpoint (llama-server's Anthropic Messages API).
# Usage: export RUNPOD_API_KEY=...; source clients/claude-code.env.sh; claude
MODEL=deepseek-v4.1-flash-abliterated

export ANTHROPIC_BASE_URL=https://api.runpod.ai/v2/oqjn0uboys2uoz/openai
export ANTHROPIC_AUTH_TOKEN="$RUNPOD_API_KEY"
export ANTHROPIC_MODEL=$MODEL
export ANTHROPIC_DEFAULT_OPUS_MODEL=$MODEL
export ANTHROPIC_DEFAULT_SONNET_MODEL=$MODEL
export ANTHROPIC_DEFAULT_HAIKU_MODEL=$MODEL   # background tasks too: there is only one model
export CLAUDE_CODE_SUBAGENT_MODEL=$MODEL

# The attribution block changes the start of the system prompt on every request,
# which defeats llama-server's prompt cache.
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072   # matches CTX_SIZE on the endpoint
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=110000  # compact before llama-server rejects the request
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=32000
export API_TIMEOUT_MS=1800000                  # first call after a pause waits for the cold start
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
