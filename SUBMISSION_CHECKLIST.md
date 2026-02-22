# Assignment Submission – Quick Checklist

## Run Demo (≈5 min)
```bash
cd ~/Documents/cats_dogs_classification_mlops
source venv/bin/activate
pip install -r requirements.txt   # if needed
./scripts/demo_full.sh
```

## If `libglib2.0-0` fails in Docker
Edit Dockerfile – remove lines 5–8:
```
# Delete the entire RUN apt-get... block, keep only:
# Python deps
COPY requirements.txt .
```

## Zip for Submission
```bash
cd ~/Documents
zip -r MLOPS_Assignment2.zip cats_dogs_classification_mlops \
  -x "*/venv/*" -x "*/.git/*" -x "*/data/raw/*" -x "*/mlruns/*" -x "*/__pycache__/*"
```

**Add to zip:** `models/model.h5` (trained model from demo)

## Video (under 5 min)
1. Run `./scripts/demo_full.sh`
2. Capture: download → train → Docker build → API → health + prediction

## Deliverables
- [ ] MLOPS_Assignment2.zip (code, configs, model)
- [ ] Screen recording link
