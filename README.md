# learn-you-a-nanochat

An [Obsidian](https://obsidian.md/) vault for learning Karpathy's [nanochat](https://github.com/karpathy/nanochat) codebase by reading, annotating, and rewriting it from memory.

## What's in here

The vault walks through `nanochat/gpt.py` (the entire transformer model in ~512 lines) section by section. Each note follows the same structure:

1. **Concept** — what problem the code solves and why it exists
2. **Code walkthrough** — annotated line-by-line with tensor shape traces
3. **nanoGPT vs nanochat** — what changed from the older nanoGPT architecture
4. **Copywork** — close the note, write the code from memory, diff against the source

## Vault structure

```
vault/
├── Index.md                        # Map of content
├── sections/
│   ├── 00 - The Big Picture.md     # 5 stages of LLM building, 7 building blocks
│   └── 01 - Config and Imports.md  # GPTConfig, imports, nanoGPT vs nanochat diffs
├── reference/
│   ├── Shape Cheatsheet.md         # Every tensor dimension from input to loss
│   ├── Glossary.md                 # Terms, symbols, and meanings
│   ├── nanoGPT vs nanochat.md      # Side-by-side architecture comparison
│   └── PyTorch API Reference.md    # Every PyTorch built-in used in nanochat
└── copywork/                       # Write code from memory here
```

## How to use

1. Clone this repo
2. Open the `vault/` directory in Obsidian
3. Start at `00 - The Big Picture`, work through the sections in order
4. After each section, close the note and rewrite the code from memory into `copywork/`
5. Diff your version against `nanochat/gpt.py`
