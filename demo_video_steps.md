# MLOps Demo Video – Step-by-Step Guide (< 5 min)

## What to Show
**Complete MLOps workflow: Code change → CI/CD → Deployed model prediction**

---

## Part 1: Local Setup & Code Change (~1 min)

1. **Open terminal, go to project**
   ```bash
   cd /path/to/cats_dogs_classification_mlops
   ```

2. **Make a small code change** (to trigger CI/CD)
   - Edit `README.md`: add a line like `# Last updated: [today's date]`
   - Or edit `app.py`: add a comment `# MLOps Assignment 2 - Cats vs Dogs API`

3. **Commit and push**
   ```bash
   git add .
   git commit -m "Demo: code change for CI/CD workflow"
   git push -u origin main
   ```

---

## Part 2: Show GitHub Actions (~1.5 min)

4. **Open GitHub repo in browser**
   - Go to: https://github.com/Sid245439/cats_dogs_classification_mlops

5. **Click "Actions" tab** – show the running workflow
   - Test job
   - Train job
   - Build job (Docker image)
   - Deploy job (smoke tests)
   - Wait until all green

---

## Part 3: Local Docker Demo (~2 min)

6. **Build and run locally**
   ```bash
   cd /path/to/cats_dogs_classification_mlops
   
   # Ensure you have a trained model (or run training first)
   python scripts/download_data.py
   python scripts/prepare_data.py
   python -c "from src.training import train_and_track; train_and_track(epochs=2)"
   
   # Build Docker image
   docker build -t cats-dogs-mlops .
   
   # Run container
   docker run -d -p 8000:8000 -v $(pwd)/models:/app/models:ro --name cats-dogs-api cats-dogs-mlops
   
   # Wait for startup
   sleep 15
   ```

7. **Show prediction**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Prediction (need a cat/dog image)
   curl -X POST -F "file=@path/to/cat_image.jpg" http://localhost:8000/predict
   ```
   Or run smoke test: `python scripts/smoke_test.py http://localhost:8000`

8. **Cleanup**
   ```bash
   docker rm -f cats-dogs-api
   ```

---

## Quick One-Liner Script
Use `./scripts/demo_video.sh` (see below) for a condensed run.

---

## Recording Tips
- Use OBS, QuickTime (macOS), or built-in screen recorder
- Split screen: terminal + GitHub Actions
- Keep narration short: "Code change → Push → CI runs tests, trains model, builds image → Deploy → Prediction works"
