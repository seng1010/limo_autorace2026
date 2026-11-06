# 협업 방법

## 규칙

1. **코드는 main에 직접 push하지 않습니다.** 항상 브랜치에서 작업하고 Pull Request(PR)로 합칩니다. 오타 같은 간단한 문서 수정은 main에 바로 올려도 됩니다.
2. **브랜치는 작업 하나당 하나입니다.** 작업이 끝나면 merge하고 삭제합니다.
3. **토픽 이름과 메시지 타입은 [README 토픽 표](README.md#토픽-인터페이스)를 따릅니다.** 바꿀 때는 팀에서 먼저 합의하고 README부터 수정합니다.

## 브랜치

### 공용 노드 (미리 생성됨)

여러 미션이 같이 쓰는 노드입니다. 담당자만 이 브랜치에서 작업하고, 나머지는 main에 merge된 뒤 토픽만 가져다 씁니다.

| 브랜치 | 노드 | 쓰는 미션 | 담당 |
|---|---|---|---|
| `common-stop-line` | `stop_line_node` | M1, M3 | 박세은 |
| `common-lane-following` | `lane_following_node` | 기본 주행, M2, M6, M7 | (미정) |
| `common-yolo` | `yolo_detection_node` | M1, M2, M3, M5 | (미정) |

### 미션 / 기타 (각자 생성)

| 종류 | 이름 형식 | 예시 |
|---|---|---|
| 미션 | `m번호-이름` | `m3-pedestrian`, `m7-obstacle` |
| 버그 수정 | `fix-내용` | `fix-m7-speed` |
| 문서 (토픽 표 변경 등 팀 합의가 필요한 경우) | `docs-내용` | `docs-topic-table` |

## 작업 흐름

```bash
# 1. 최신 main에서 브랜치 생성
git checkout main
git pull
git checkout -b m3-pedestrian

# 2. 작업하면서 커밋, push
git add .
git commit -m "M3 정지선 정지 구현"
git push -u origin m3-pedestrian
```

3. GitHub에서 **Compare & pull request** → 설명 작성 → **Reviewers**에 팀원 1명 지정
4. 리뷰어가 확인하면 **Merge** → 브랜치 삭제
5. 다음 작업은 다시 1번부터 시작

공용 노드 브랜치(`common-*`)는 이미 있으므로 1번 대신 아래처럼 시작합니다.

```bash
git checkout common-yolo
git merge main        # 최신 main 반영
```

## 공용 노드가 아직 없을 때 테스트

필요한 토픽을 직접 발행해서 테스트합니다.

```bash
ros2 topic pub perception/detections std_msgs/String "data: '[{\"label\": \"pedestrian\", \"conf\": 0.9}]'"
ros2 topic pub perception/stop_line_distance std_msgs/Float32 "data: 0.3"
```

## 충돌이 나기 쉬운 파일

`setup.py`, `params/params.yaml`, launch 파일, `mission_manager_node.py`는 모두가 수정하는 파일입니다.

- `setup.py`의 `console_scripts`나 `params.yaml`에서 충돌이 나면 **양쪽 줄을 모두 남깁니다.**
- `mission_manager_node.py`를 크게 바꿀 때는 미리 팀에 공유합니다.
