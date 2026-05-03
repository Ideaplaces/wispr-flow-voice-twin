# MCP server

The voice twin ships an MCP server that runs as a stdio child process. Any MCP-aware client (Claude Desktop, Claude Code, Cursor, Windsurf, future agents) can launch it and gain seven tools that read your corpus locally.

Nothing crosses the network. The server is a Python process the client spawns; it talks JSON-RPC over stdin and stdout.

## Tools exposed

| Tool | What it does |
|---|---|
| `voice_search` | Top-K nearest dictations to a free-form query. Optional `ctx` filter (ai_chat / team_chat / personal_chat / browser). |
| `voice_topics_list` | Browse the BERTopic clusters with their LLM labels. |
| `voice_topic_show` | Full detail for one topic (top words, sample dictations, recent examples). |
| `voice_topic_find` | Best-matching topic for a query plus three nearby topics for hopping. |
| `voice_draft` | Generate a draft in your voice. Modes: slack, blog, linkedin, twitter, email, rewrite. Returns the draft plus the dictations the model used as reference. |
| `voice_coach` | Critique a draft against your baseline voice. |
| `voice_patterns_list` | Automation candidates from `pipeline/09_patterns.py`. |

## Claude Desktop

Open `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) and add:

```json
{
  "mcpServers": {
    "voice-twin": {
      "command": "/absolute/path/to/wispr-flow-voice-twin/.venv/bin/python",
      "args": ["/absolute/path/to/wispr-flow-voice-twin/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop. Open a new conversation. The seven tools appear in the tools menu and Claude will call them on its own when appropriate.

Linux paths are the same shape; the config file lives at `~/.config/Claude/claude_desktop_config.json`.

## Claude Code

Either drop the same JSON into `~/.config/claude-code/mcp.json`, or run:

```bash
claude mcp add voice-twin --command /absolute/path/to/.venv/bin/python --args /absolute/path/to/mcp_server.py
```

After that any Claude Code session in any directory has the tools available.

## Cursor

Cursor reads MCP config from `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "voice-twin": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

## What the agent can do once it's wired up

A few prompts that exercise the tools:

- "What did I say last quarter about Terraform deploying Sentry? Pull 20 examples." → `voice_search`
- "Draft a LinkedIn post about my week, in my voice." → `voice_draft` with mode=linkedin
- "What topics in my corpus look like things I keep telling someone to do?" → `voice_patterns_list`
- "Coach this email, then give me the tightened version." → `voice_coach`
- "What's the topic-level neighborhood around 'observability'? Show me three nearby topics." → `voice_topic_find`

The agent picks tools on its own. You don't have to name them.

## Verifying it runs

Without restarting any client, you can drive the server by hand to confirm everything is wired:

```bash
(echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}';
 echo '{"jsonrpc":"2.0","method":"notifications/initialized"}';
 echo '{"jsonrpc":"2.0","id":2,"method":"tools/list"}') \
  | .venv/bin/python mcp_server.py | head
```

You should see an initialize response followed by a `tools/list` response with all seven tools.

## What stays local

The server inherits the same provider configuration as the rest of the codebase. With `LLM_PROVIDER=ollama` and `EMBED_PROVIDER=local`, every MCP tool call stays on your machine. The agent that called the tool may be remote (Claude Desktop talks to anthropic.com for its own thinking), but the corpus access does not.

If you want even the model layer to stay local, point Claude Desktop at a local LLM via its config and pair it with the local-provider mode here.
