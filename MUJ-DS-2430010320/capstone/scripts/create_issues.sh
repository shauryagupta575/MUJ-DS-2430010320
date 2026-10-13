#!/usr/bin/env bash
# Creates labels and the five starter issues with the GitHub CLI (https://cli.github.com).
# Usage: run `gh auth login` once, then from the repository root:
#   bash capstone/scripts/create_issues.sh
set -euo pipefail

DATA_OWNER="shauryagupta575"      # change to a teammate's username if they own this
MODEL_OWNER="shauryagupta575"      # model development
TEST_OWNER="shauryagupta575"
DOCS_OWNER="shauryagupta575"
DEPLOY_OWNER="shauryagupta575"

gh label create dataset       --color 0E8A16 --description "Data collection and preparation" --force
gh label create model         --color D93F0B --description "Model development"              --force
gh label create testing       --color FBCA04 --description "Evaluation and validation"      --force
gh label create documentation --color 0075CA --description "Docs and reports"               --force
gh label create deployment    --color 5319E7 --description "Packaging and demo"             --force

gh issue create --title "Dataset Collection: download, validate and document the FIRE dataset" \
  --label dataset --assignee "$DATA_OWNER" \
  --body "Download the Kaggle FIRE dataset (999 images), check for corrupt files, document class counts and the stratified 70/15/15 split in capstone/data/README.md."

gh issue create --title "Model Development: ViT-B/16 transfer learning and CNN baselines" \
  --label model --assignee "$MODEL_OWNER" \
  --body "Train ViT-B/16 (frozen then last-4-block fine-tune), ResNet50, MobileNetV2 and the SE-attention CNN. Record metrics in capstone/results/."

gh issue create --title "Testing: multi-seed evaluation and external dataset test" \
  --label testing --assignee "$TEST_OWNER" \
  --body "Rerun the top models over 5 seeds (mean ± SD) and evaluate on an external dataset (FlameVision or DeepFire)."

gh issue create --title "Documentation: README, installation guide, CO4/CO5 reports" \
  --label documentation --assignee "$DOCS_OWNER" \
  --body "Keep README results current, maintain capstone/INSTALLATION.md, and finalise the CO4 and CO5 reports."

gh issue create --title "Deployment: Streamlit demo for single-image fire detection" \
  --label deployment --assignee "$DEPLOY_OWNER" \
  --body "Export the ViT-B/16 model and build a simple upload-an-image demo; add the run command or link to the README."

echo "Done. View issues with: gh issue list"
