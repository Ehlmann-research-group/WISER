# Code Review and Quality Guide

This document explains how **code review** and **code submittal** work in WISER.  
Its purpose is to help contributors and maintainers write consistent, maintainable, and
readable code - and to make the review process efficient and collaborative.

## Overview

Code review and submittal in WISER are centered on producing **high-quality, maintainable code**
that aligns with the project’s design philosophy and long-term vision.  
This guide covers:

- [Code Review Guidelines](#code-review-guidelines)  
- [Commit Message Standards](#commit-message-standards)  
- [Commenting Style](#commenting-style)  
- [Naming and Code Style](#naming-and-code-style)

All pull requests (PRs) should be tied to an **existing issue**.  
If no issue exists, **create one first** - we provide templates for common issue types to make this easy.

## Code Review Guidelines

Code review in WISER focuses on ensuring that contributions meet both **functional** and **design** expectations.

### What We Look For

| Category | Description |
|-----------|--------------|
| **Maintainability & Readability** | Is the code easy to understand and modify? Are variables and functions well named? |
| **Design Quality** | Is the code modular, well-structured, and aligned with WISER’s architecture? |
| **Correctness** | Does the code do what it’s supposed to do? Are there tests verifying its behavior? |
| **Simplicity** | Can the code be simplified without losing clarity or function? |
| **Style Conformance** | Are naming conventions, comment formats, and indentation consistent with the project’s style? |
| **Documentation Needs** | Does this change need documentation? Should it be added before or after merging? (It’s okay to open a second PR for docs or tests.) |
| **Performance** | Are there any obvious inefficiencies or bottlenecks? |
| **Duplication** | Does similar code already exist elsewhere? If so, can it be reused? |
| **Scope & Focus** | Is the code single-purpose? Avoid adding multiple unrelated changes in one PR. |

### PR Expectations

- Every PR should have a **clear purpose** and be linked to an **issue**.  
- Keep PRs **small and focused**. Avoid exceeding a few hundred lines of changes where possible.  
- If documentation or tests are pending, you may open a **follow-up PR**.  
- Reviewers are encouraged to comment on **design clarity, naming, and simplicity** just as much as functionality.


## Commit Message Standards

Commit messages communicate the *why* behind changes.

### Format

A proper commit message should look like this:

>Short subject line (max 50 chars)
>
>Detailed body explaining what changed and why.

Wrap lines at 50 characters.

### Guidelines

- Keep commit messages **clear and concise**.  
- Describe **what** was done and **why**.  
- Long, detailed messages are helpful during PR review but are less important after a squash merge.  
- Avoid long unbroken lines; insert newlines around 50 characters.

### Merge Commits

When squash merging a PR:
- The **PR title** becomes the **merge commit title**, followed by the PR number (e.g., `feat: add spectral viewer #298`).  
- The **PR description** serves as the commit body.  
- After merging:
  - **Delete your branch** both locally and on GitHub.  
  - **Pull from `main`** to sync your local repository.

## Commenting Style

Comments clarify *why* code exists - not just *what* it does.  
Focus on explaining reasoning, design choices, and non-obvious logic.

> Reference: [Do’s and Don’ts of Commenting Code](https://blog.openreplay.com/dos-and-donts-of-commenting-code/)

### General Principles

- **Explain the “why.”** Use comments to clarify design decisions, algorithms, or tricky logic.  
- **Avoid redundancy.** Don’t restate what the code already clearly expresses.  
- **Comment for others.** Write for future maintainers and collaborators.  
- **Use inline comments sparingly.** Prefer comments at function or block level for clarity.  
- **At API boundaries**, explain how to use or extend the code.

### Formatting Conventions

| Type | Example | Description |
|------|----------|-------------|
| **File/Module Header** | `"""Brief description of what this file/module does and why it exists."""` | At the top of every file. |
| **Class/Function Docstrings** | `"""Explain purpose, behavior, and key parameters."""` | Triple quotes. |
| **Block Comments** | `"""Describe logic behind a block of code."""` | Triple quotes. |
| **Inline Comments** | `# Explain non-obvious line or condition` | Use `#` sparingly for complex expressions. |

---

## Naming and Code Style

Code should be **self-describing**: good names make code easier to read and maintain.

### Python Naming Conventions

| Element | Convention | Example |
|----------|-------------|----------|
| **Classes** | `PascalCase` | `ImageProcessor` |
| **Functions** | `snake_case` | `load_dataset()` |
| **Variables** | `snake_case` | `spectral_band` |
| **Constants** | `ALL_CAPS` | `MAX_CACHE_SIZE` |

### GUI Widget Naming

Since WISER uses **PySide2**, a wrapper around **Qt** (a C++ library), some functions may follow Qt’s C++ naming conventions - this is acceptable when overriding or subclassing Qt components.

To maintain clarity in GUI code, widgets should use consistent prefixes:

| Widget Type | Prefix | Example |
|--------------|---------|----------|
| Push Button | `btn_` | `btn_apply_stretch` |
| Label | `lbl_` | `lbl_filename` |
| Widget | `wdgt_` | `wdgt_spectral_viewer` |
| Combo Box | `cbox_` | `cbox_bandselector` |
| Tool Button | `tbtn_` | `tbtn_zoom_in` |
| Line Edit | `ledit_` | `ledit_path_input` |
| Spin Box | `sbox_` | `sbox_frame_index` |

### Enforced Code Style

Python formatting is **enforced automatically**:
- Style rules are defined in **`pyproject.toml`**.  
- They are validated by **pre-commit hooks** and **GitHub Actions**.  
- Always run pre-commit checks locally before pushing to avoid CI failures.

## Summary

In WISER, code review and submission are designed to:
- Maintain **clarity**, **simplicity**, and **design consistency**.  
- Encourage **collaboration** and **shared responsibility** for quality.  
- Keep the codebase healthy, documented, and performant.

By following these guidelines, we ensure that WISER remains sustainable and welcoming to contributors - now and in the future.
