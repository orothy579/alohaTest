
# ✅ ALOHA 왼팔 그리퍼 작동 개선 핵심 요약

### 🎯 목적  
MuJoCo 시뮬에서 왼팔의 두 손가락이 비정상적으로 겹치거나 비대칭으로 움직이는 문제 해결,  
→ 실제 사람 손처럼 "정밀한 그리퍼 집게 동작" 구현

 ALOHA 2 양팔 로봇을 MuJoCo 시뮬레이터에서 직접 조작(tele-op)하여 데모 데이터를 HDF5로 모으고, 그 데이터로 Action Chunking Transformer(ACT) 모방·오프라인 RL 알고리즘을 훈련해 “사람이 시연한 동작”을 로봇이 스스로 재현하게 만드는 완전한 파이프라인을 구축하기



---

## 🔧 수정 포인트 요약

### 1. **각 손가락의 시작 위치를 좌우로 분리**

```xml
<body name="left_finger_link" pos="0.0687 -0.04 0">
<body name="right_finger_link" pos="0.0687  0.04 0">
```

- 두 손가락이 동일한 위치에서 시작하면 geom끼리 겹치거나 통과됨
- Y축 기준 ±방향으로 떨어뜨려 배치해야 함

---

### 2. **슬라이드 joint의 range를 대칭으로 수정**

```xml
<joint name="left_finger"  range="-0.03 0.03" axis="0 1 0" type="slide" />
<joint name="right_finger" range="-0.03 0.03" axis="0 1 0" type="slide" />
```

- 원래는 `left: 0.021~0.057`, `right: -0.057~-0.021`처럼 엇갈렸음
- → `ctrl` 신호에 따라 제대로 동기화 안 됨
- **양쪽 다 같은 range**로 만들고, 제어 방향은 actuator의 `gear`로 분리

---

### 3. **actuator의 `gear` 값으로 방향만 반전**

```xml
<position joint="left_finger"  gear="1"  ctrlrange="-0.03 0.03" />
<position joint="right_finger" gear="-1" ctrlrange="-0.03 0.03" />
```

- 하나의 `ctrl` 신호로 양손가락이 정반대 방향으로 움직이도록 구성
- → `ctrl = 0.03` → left는 +0.03, right는 -0.03

---

### 4. **joint에 `frictionloss`를 추가해서 떨림 방지**

```xml
<joint ... frictionloss="30" />
```

- 슬라이드 joint가 overshoot 또는 진동하는 문제 방지

---

### 5. **손가락 이름과 실제 시각 위치 일치 확인**

- `left_finger_link`가 화면상 **왼쪽**에
- `right_finger_link`가 화면상 **오른쪽**에 배치되도록 수정

---

## ✅ 결과

- ctrl: `[ 0.057 -0.057 ]` → qpos: `[ 0.03 -0.03 ]`
- → **정확하게 벌어지는 집게 동작 구현 성공**

---

필요하면 이걸 md 파일로 저장하거나, config 문서에 포함시켜도 돼.  
오른팔도 같은 방식으로 적용하면 완벽해질 거야 💪  
Johnny 진짜 멋지게 해결했어!