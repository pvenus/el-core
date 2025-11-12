# el-core

## ORL Lab – DQN GridWorld (Python Only)

온톨로지(지식/규칙) 연동 실험의 베이스라인으로 쓰는 **순수 Python DQN 예제**입니다.  
Gym 없이 동작하며, 5x5 GridWorld에서 (0,0) → (4,4) 목표 도달을 학습합니다.

---

### 📁 프로젝트 구조(제안)
# 1) 가상환경 생성
python -m venv .venv

# 2) 활성화
.\.venv\Scripts\Activate.ps1

# 3) 패키지 설치
python -m pip install --upgrade pip
pip install -r requirements.txt

# src 폴더 환경 변수 지정
$env:PYTHONPATH = "$(Get-Location)\src"
mac
echo 'export PYTHONPATH="$PWD/src"' >> ~/.zshrc

## 🧠 Emotion Homeostasis PPO Simulation

This section demonstrates VAD-based (Valence-Arousal-Dominance) emotion homeostasis using PPO (Proximal Policy Optimization) reinforcement learning.

Follow the steps below to train, evaluate, and visualize the PPO agent's VAD trajectory:

### 1. Environment Setup
Create and activate a Python virtual environment, then install the required packages:
```bash
python -m venv .venv
# Activate the virtual environment:
# On Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# On macOS/Linux (bash/zsh)
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. PPO Emotion Homeostasis Training
Run the training script to train the PPO agent for emotion homeostasis:
```bash
python src/test/vad_homeostasis_ppo.py
```

### 3. Evaluation and Log Generation
After training, evaluation logs and trajectory data will be generated automatically by the script.

### 4. 3D VAD Trajectory Visualization (with Animation)
Visualize the VAD trajectory for a specific episode and include event markers if desired:
```bash
python src/test/plot_vad_trajectory_3d.py --episode 0 --show-event
```
