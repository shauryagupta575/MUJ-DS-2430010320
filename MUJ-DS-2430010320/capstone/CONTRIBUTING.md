# Contributing

## Workflow

1. **Pick or create an issue** and assign it to yourself.
2. **Create a branch** from `main`:
   ```bash
   git checkout main && git pull
   git checkout -b <type>/<issue-number>-<short-name>
   ```
   Types: `feature/`, `fix/`, `docs/`, `data/`, `test/`, `deploy/`. Example: `feature/2-vit-finetuning`.
3. **Commit** small, meaningful changes:
   ```bash
   git add <files>
   git commit -m "Add ViT phase-2 fine-tuning (#2)"
   git push -u origin <branch-name>
   ```
4. **Open a pull request** into `main`. Fill in the template, link the issue with `Closes #<number>`, and request a review from at least one teammate.
5. **Review.** The reviewer checks the code runs, results are reproducible, and docs are updated; they either approve or request changes.
6. **Merge** only after approval, using **Squash and merge**. Delete the branch afterwards.

## Rules

- Never push directly to `main`.
- Do not commit datasets, model weights (`*.keras`, `*.h5`) or API keys.
- Update `capstone/results/` and the README when a change alters reported numbers.
