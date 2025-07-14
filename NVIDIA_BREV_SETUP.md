# NVIDIA Brev Training Setup Guide

Complete step-by-step guide to train your embeddings-to-prompts model on NVIDIA Brev.

## 🚀 Quick Start (5 minutes)

### 1. Create NVIDIA Brev Account

1. Go to [NVIDIA Brev](https://brev.nvidia.com/)
2. Click **"Get Started"** → Sign up with GitHub/email
3. Complete account verification

### 2. Create GPU Launchable

1. **Click "Create Launchable"**
2. **Configure GPU Instance:**
   - **Name**: `embeddings-to-prompts-training`
   - **GPU**: NVIDIA RTX 4090 or A100 (recommended)
   - **RAM**: 32GB+ 
   - **Storage**: 50GB+
   - **Region**: US-East (usually fastest)

3. **Container Configuration:**
   - **Base Image**: `nvidia/cuda:11.8-devel-ubuntu20.04`
   - **Or use**: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel`

### 3. Upload Your Project

**Option A: GitHub Integration**
1. Push your project to GitHub
2. In Brev, connect your GitHub repo
3. Select the `embeddings-to-prompts` repository

**Option B: Direct Upload**
1. Create a ZIP file of your project
2. Upload via Brev's file manager
3. Extract to `/workspace/`

### 4. Launch Instance

1. Click "Launch"
2. Wait 2-3 minutes for GPU provisioning
3. Click **"Connect"** when ready

## 📦 Files to Upload

Make sure these files are in your workspace:

```
embeddings-to-prompts/
├── data/embeddings/training_data.jsonl    # 51,602 training examples
├── config.yaml                            # Model configuration  
├── scripts/train.py                       # Training script
├── models/architecture.py                 # Model definitions
├── requirements.txt                       # Dependencies
├── setup_brev.sh                         # Environment setup
└── run_brev_training.py                  # Training launcher
```

## 🛠️ Environment Setup

Once connected to your Brev instance:

```bash
# 1. Navigate to workspace
cd /workspace

# 2. Make setup script executable
chmod +x setup_brev.sh

# 3. Run setup (takes ~5 minutes)
./setup_brev.sh
```

This will:
- ✅ Install CUDA 11.8 + PyTorch
- ✅ Install all Python dependencies
- ✅ Set up directories and environment
- ✅ Configure GPU access

## 🎯 Start Training

### Automatic Training (Recommended)

```bash
# Activate environment
source /opt/venv/bin/activate

# Run complete training pipeline
python run_brev_training.py
```

This will:
1. ✅ Check GPU availability
2. ✅ Verify training data (51,602 examples)
3. ✅ Set up environment variables
4. ✅ Start training with progress logs
5. ✅ Save trained model to `models/saved/best_model.pt`

### Manual Training

If you prefer manual control:

```bash
# Activate environment
source /opt/venv/bin/activate

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Start training
python scripts/train.py --gpu --save-model
```

## 📊 Monitor Training

### Real-time Logs
```bash
# Watch training progress
tail -f brev_training.log
```

### TensorBoard (Optional)
```bash
# Start TensorBoard
tensorboard --logdir=logs --host=0.0.0.0 --port=6006

# Access at: http://your-instance-ip:6006
```

### GPU Usage
```bash
# Monitor GPU utilization
nvidia-smi -l 1
```

## ⏱️ Expected Training Time

| GPU Type | Training Time | Memory Usage |
|----------|---------------|--------------|
| RTX 4090 | 1-2 hours | 16GB+ |
| A100 | 45-90 minutes | 20GB+ |
| V100 | 2-3 hours | 16GB+ |

## 🎉 After Training

### Check Results
```bash
# Training summary
cat logs/training_summary.json

# Model file
ls -la models/saved/best_model.pt
```

### Test Inference
```bash
# Test the trained model
python scripts/inference.py \
  --model-path models/saved/best_model.pt \
  --embedding-file test_embeddings.json
```

### Download Model
```bash
# Zip the trained model
zip -r trained_model.zip models/saved/ logs/

# Download via Brev file manager
```

## 🔧 Troubleshooting

### Common Issues

**1. GPU Not Available**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version
```

**2. Out of Memory**
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 16  # Reduce from 32
```

**3. Training Data Not Found**
```bash
# Verify data file
ls -la data/embeddings/training_data.jsonl
head -n 1 data/embeddings/training_data.jsonl
```

**4. Dependencies Missing**
```bash
# Reinstall requirements
pip install -r requirements.txt
```

### Performance Tips

**1. Optimize GPU Usage**
```bash
# Set optimal batch size
export BATCH_SIZE=32

# Use mixed precision
export USE_AMP=1
```

**2. Monitor Resources**
```bash
# Check disk space
df -h

# Check memory
free -h

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 💰 Cost Estimation

| GPU Type | Cost/Hour | Expected Total |
|----------|-----------|----------------|
| RTX 4090 | $0.50-1.00 | $1-2 |
| A100 | $1.50-3.00 | $2-4 |
| V100 | $0.80-1.50 | $2-3 |

## 🚀 Next Steps

After successful training:

1. **✅ Download your trained model**
2. **✅ Test inference locally**
3. **✅ Deploy to production**
4. **✅ Implement Stage 2 (Reinforcement Learning)**

## 📞 Support

If you encounter issues:

1. **Check logs**: `cat brev_training.log`
2. **NVIDIA Brev Docs**: [docs.nvidia.com/brev](https://docs.nvidia.com/brev)
3. **Community**: NVIDIA Developer Forums

---

**🎯 You're all set! Your model will be ready for inference in 1-3 hours.** 