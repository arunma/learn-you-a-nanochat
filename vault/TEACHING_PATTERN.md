# Teaching Pattern — How Each Section Is Built

This documents the repeatable process used to create each vault section. Intended for eventual automation as a Claude Code skill.

## Per-section recipe

### 1. Read the source chunk
- Read the exact lines from the original file (e.g., `nanochat/gpt.py` lines 65–127)
- Identify every class, function, and method in the chunk

### 2. Write the vault section with these mandatory components

**a) Header block**
- Obsidian frontmatter (aliases, tags, source lines)
- Phase tag, one-line description
- Source file + line range
- Copywork target (approximate line count)

**b) "What this code does" — plain English summary**
- One paragraph, no code, explains the problem being solved
- Numbered list of the logical steps the code performs

**c) Code walkthrough — annotated original**
- Show the actual code with inline comments
- Break into logical parts (e.g., `__init__` and `forward` separately)
- Explain each line or group of lines

**d) Shape trace for EVERY operation**
- Three-column format: operation | shape | plain English
- Group by logical phase with clear headers
- Show EVERY intermediate step — no dimension jumps
- Highlight new dimensions (bold or color)
- Use concrete values from GPTConfig defaults (768, 6, 128, etc.)

**e) Numeric example where applicable**
- Pick a small concrete case (e.g., RMSNorm on `[200, -400, 100, 300]`)
- Walk through the math step by step with actual numbers
- Show input → computation → output

**f) "Versus" table — nanoGPT vs nanochat**
- Side-by-side comparison for every concept that changed
- Include what the user's earlier notes said vs what nanochat does
- Explain WHY the change was made

**g) Deep dives for non-obvious concepts**
- When a concept is genuinely new (RoPE, value embeddings, flash attention)
- Origin/paper reference
- Intuitive analogy before the math
- Architecture diagram (ASCII) showing where it fits

**h) Dependency map**
- Which other classes/functions use the code in this section
- Helps the learner see how pieces connect

**i) Copywork checkpoint**
- Explicit list of what to write from memory
- Broken into sub-parts (e.g., `__init__` items vs `forward` items)
- "Common traps" — specific mistakes to watch for, phrased as yes/no questions

### 3. Teach it conversationally BEFORE the vault note
- Walk through the key points in chat
- Highlight the 3-5 things most likely to trip up during copywork
- Reference what the user already knows from prior sections

### 4. Review the copywork
- Read the user's `my_nanochat/gpt.py` after they say "done"
- Flag bugs categorized as:
  - **Typos** (wrong name, missing character)
  - **Missing pieces** (forgot a field, skipped a step)
  - **Logic errors** (wrong operation, wrong order)
- Don't flag style differences that don't affect correctness
- User fixes, then re-review

### 5. Commit
- Stage `my_nanochat/gpt.py` and the vault section
- Commit message format:
  ```
  Complete Section NN copywork: [brief description]

  [What was added]. Mistakes caught during copywork:
  - [mistake 1]
  - [mistake 2]
  ```
- Push to GitHub

## Section structure template

```markdown
---
aliases: [...]
tags: [section, phase-N]
source: nanochat/gpt.py lines X–Y
---

# NN — Title

<span class="phase-tag">SECTION N</span> *One-line description*

> **Source:** `nanochat/gpt.py` lines X–Y
> **Copywork target:** ~NN lines

---

## What this code does
[Plain English, no code]

---

## Part 1: [logical chunk]
[Code + annotation]

> [!shape] Shape trace
> [Three-column format]

> [!versus] nanoGPT vs nanochat
> [Side-by-side table]

> [!keyinsight] [Title]
> [The thing that clicks it together]

---

## The complete shape trace — input to output
[Full pipeline for this section]

---

> [!copywork] Copywork checkpoint
> [What to write, common traps]

---

*Previous/Next links*
*Back to [[Index]]*
```

## What makes the copywork review effective

1. **Read their code without seeing the original** first — spot what looks wrong on its own
2. **Then diff** — catch subtle issues (wrong variable name, missing `% 2`)
3. **Categorize** — typo vs logic error vs missing piece (different learning signals)
4. **Don't over-flag** — style differences and personal comments are fine
5. **Phrase as questions** — "Did you mean X?" not "You got X wrong"
