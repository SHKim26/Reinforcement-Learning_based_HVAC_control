# Reinforcement-Learning_based_HVAC_control

## Abstract

이 프로젝트는 **에어컨과 환풍기를 활용해 i) 실내 온도 불균형을 해소하고, ii) 설정온도를 달성하는 iii) 비용 효율적인 제어**를 강화학습으로 도출한다. 

## Background

- Motivation : 환풍기를 통한 적절한 공기순환율 제어는 실내 온도 불균형을 해소하고, 냉난방 장치의 효율을 극대화 할 수 있다.

![Image](https://github.com/user-attachments/assets/8d97a5ec-cb99-4809-9577-492b83a77b00)

- Effects of Fan : 환풍기를 가동한 오른쪽의 경우, 실내 온도가 더 빠르게 감소하고 불균형도 해결됨.

- Needs for optimal control : 환풍기와 에어컨의 강도를 적절히 조절하면, 실내 온도 불균형을 해소하면서도 공간 전체가 타겟 온도에 도달하는 비용 효율적인 제어가 도출가능함.

## Markov Deicion Process (MDP)

### State (관측값)
- 형태: (20, 20) 2D 격자
- 의미: 각 셀의 실내 온도 [20, 35] °C
- 에이전트는 매 스텝 **현재 방 전체 온도 분포**를 그대로 관측.

### Action (행동)
- 연속형 2차원 벡터: `[ac_strength, fan_power]`
  - `ac_strength` : 에어컨 강도, [0, 10]
  - `fan_power`   : 환풍기 강도, [0, 10]
- Stable-Baselines3 입장에서는 `Box(low=[0,0], high=[10,10])`.

### Reward (보상)
- 한 스텝 보상은 다음 항의 합으로 구성.
  - **온도 개선도**: 목표 온도(25°C)에서 평균 편차가 줄어들수록 보상 ↑
  - **온도 균일성**: 현재 온도 분포의 표준편차가 작을수록 보상 ↑
  - **에너지 사용량 패널티**: AC·Fan 강도가 클수록 보상 ↓
  - **목표 도달 보너스**: 모든 격자가 허용 오차(±1.5°C) 안에 들어오면 큰 보상 ↑
  - **시간 패널티**: 에피소드가 길어질수록 보상 ↓ → 빠른 수렴 유도

## Results

![Image](https://github.com/user-attachments/assets/fb98ec80-95bb-4a6f-9a6f-320189230937)

- Intelligent behavior : 환풍기와 에어컨의 적절한 제어를 통해, 실내 온도 불균형을 빠르게 해소하고 타겟 온도 25 도에 도달함.

## Implementation details

- 전체 코드 : `rl_hvac.py`

- 환경: `gymnasium` 기반 커스텀 환경 `HVACEnv`

- 에이전트: Stable-Baselines3의 `PPO`

---

### Python

* `gymnasium`
* `numpy`
* `matplotlib`
* `stable-baselines3`
* `pillow`

---

### RL agent (PPO)

- 알고리즘: `PPO("MlpPolicy", ...)` (Stable-Baselines3)
- 네트워크 구조:
  - 정책/가치망 모두 2층 MLP, 각 층 256 유닛: `pi=[256, 256], vf=[256, 256]`
- 주요 하이퍼파라미터:
  - `learning_rate = 1e-3`
  - `gamma = 0.99`
  - `n_steps = 1024` 
  - `batch_size = 32`
  - `n_epochs = 10`
  - `clip_range = 0.2`
  - `ent_coef = 0.01` 

## Future works

본 프로젝트는 단순화된 가정으로 한계가 존재합니다
- 공기역학 단순화
- 2D 격자 셋팅의 매우 작은 공간
- ...

추후 i) 공기 역학 시뮬레이션 프로그램과의 결합 ii) 실제 주거 구조 데이터 활용 iii) 다수의 냉난방 장치 활용 등을 고려하여, 더 발전된 HVAC control 을 도출할 수 있습니다.


