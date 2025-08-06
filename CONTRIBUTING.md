# Contributing to MIC Academic Assistant
## 1. Fork and Clone the Repository
1. Click the "Fork" button at the top right of [this repo](https://github.com/micvitc/acad-assistant).
2. Clone your forked copy:
   ```
   git clone https://github.com/<your-username>/acad-assistant.git
   cd acad-assistant
   ```
3. Add the original repo as an upstream remote:
   ```
   git remote add upstream https://github.com/micvitc/acad-assistant.git
   ```
## 2. Create a Feature Branch
Create your own branch for development:
```
git fetch upstream
git checkout -b feat/your-feature-name upstream/dev
```
(Please use a meaningful name like feat/resume-parser or fix/upload-bug)
## 3. Setup and Test Locally
Make sure Docker and Docker Compose are installed.
```
 docker compose up
```
Visit: ```http://localhost:8000``` to test endpoints. (Or your defined port)
## 4. Commit and Push your Code
```
git add .
git commit -m "Add: PDF summarizer endpoint" (Example, Please add meaningful commit messages)
git push origin feat/your-feature-name
```
## 5. Create a Pull Request
1.Go to your fork on GitHub.

2. Click "Compare & Pull Request".

3. Choose:

  Base repo: micvitc/acad-assistant

  Base branch: dev

  Compare: your feature branch

4.Fill in a clear title and description.

5. Submit the PR.

All pull requests will be reviewed by the project lead. You'll receive feedback if needed. Once approved, your code will be merged

## 6. Contribution Tips
Follow consistent code style.

Comment where needed.

Ensure Docker build works without breaking.

Regularly pull updates:
```
git fetch upstream
git merge upstream/dev
```
