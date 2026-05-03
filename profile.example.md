---
name: Your Name
nickname: ""
positioning: ""
companies: []
taboo_phrases:
  - fractional CTO
  - synergy
  - leveraging
  - in today's world
  - revolutionize
glossary: {}
preferred_tone: warm, direct, declarative
preferred_punctuation: "commas, colons, line breaks (avoid em dashes)"
---

# Custom prompt overrides

This file is the public template. Copy it to a private location, fill in
your own values, and point the `VOICE_TWIN_PROFILE` environment variable
at the new file. Examples:

    export VOICE_TWIN_PROFILE=~/.voice-twin/profile.md
    export VOICE_TWIN_PROFILE=https://your-bucket.s3.amazonaws.com/profile.md

Frontmatter fields above are required. Optional markdown sections below
override the per-mode prompt fragments. Each `## name` becomes a
`{{section.name}}` value in the prompt templates.

## blog

Voice should read like a coffee with a senior technical leader who has
battle scars and shares them. Lead with stakes and numbers. The why
before the how.

## linkedin

First line is a number, a quoted line, or a moment. Not a setup.
150 to 280 words. No corporate cliches.

## slack

Brief, warm, status-update tone. Drop trailing periods on short lines.
Stack short status lines for vertical breathing room.

## email

Match the tone of the thread you are replying to. Keep paragraphs short.
Default to plain prose, no bullet lists unless the thread already has
them.
