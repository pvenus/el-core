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
