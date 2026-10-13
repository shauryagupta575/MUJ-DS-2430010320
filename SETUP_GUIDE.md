# Setup Guide — single repository `MUJ-DS-2430010320`

Everything (personal portfolio + capstone) lives in one public repository.

## Push it (VS Code terminal)

1. Create the repo at <https://github.com/new>: name `MUJ-DS-2430010320`, **Public**, no README / .gitignore / license.
2. In VS Code, open the `MUJ-DS-2430010320` folder, then **Terminal → New Terminal**:

```bash
git init
git add .
git commit -m "Initial portfolio and capstone: code, notebooks, reports, results, presentation"
git branch -M main
git remote add origin https://github.com/shauryagupta575/MUJ-DS-2430010320.git
git push -u origin main
```

## After pushing

| Step | What to do |
| --- | --- |
| 1 Join organization | Accept the invite at <https://github.com/settings/organizations>; complete your profile |
| 3 Instructor | Repo → **Settings → Collaborators → Add people** → instructor's username |
| 5 README | Fill in Training Program details and instructor username in `README.md` |
| 6 Weekly updates | Each week: make a real change, add a row to the progress log, `git add . && git commit -m "Week N: ..." && git push` |
| 7 Team | If you have teammates, add them as collaborators and to the table in `capstone/README.md` |
| 9 Issues | **Issues → New issue** shows the 5 templates; or run `bash capstone/scripts/create_issues.sh` with the GitHub CLI |
| 10 Pull requests | Follow `capstone/CONTRIBUTING.md`: branch → commit → push → PR → review → merge |
| 12 Submit | Repo URL `https://github.com/shauryagupta575/MUJ-DS-2430010320` for both the personal and capstone fields, plus `presentations/Wildfire_Detection_CNN_Presentation.pptx` |

Confirm with your instructor that submitting the same URL for personal and capstone is acceptable.
