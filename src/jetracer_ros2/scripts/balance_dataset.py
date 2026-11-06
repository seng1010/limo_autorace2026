#!/usr/bin/env python3
"""
JetRacer 데이터셋 cx(조향) 분포 확인 + 직진 편중 다운샘플링

train.py와 동일하게 파일명 {cx:03d}_{cx:03d}_{index:06d}_{timestamp}.jpg 에서
cx(0~300, 150=직진)를 읽어 구간을 나누고 개수를 센다.

기본값(--max_per_bin 없이 실행)은 히스토그램만 출력하고 파일은 건드리지 않는다.
--max_per_bin을 주면 그 개수를 초과하는 bin에서 초과분을 무작위로 골라
영구 삭제가 아니라 <section_dir>/_excluded/ 로 옮겨둔다 (되돌리고 싶으면 다시 옮기면 됨).

Usage:
    # 1) 먼저 분포만 확인
    python balance_dataset.py --data_dir ~/jetracer_dataset --section 2

    # 2) bin당 최대 400장으로 다운샘플링 (초과분은 _excluded/ 로 이동)
    python balance_dataset.py --data_dir ~/jetracer_dataset --section 2 --max_per_bin 400
"""

import argparse
import os
import glob
import random
import shutil
from collections import defaultdict


def cx_of(path: str) -> int:
    return int(os.path.basename(path).split('_')[0])


def bin_of(cx: int, bin_width: int) -> int:
    return (cx // bin_width) * bin_width


def print_histogram(bins: dict, bin_width: int, max_per_bin: int = None):
    total = sum(len(v) for v in bins.values())
    max_count = max((len(v) for v in bins.values()), default=0)
    bar_width = 50
    print(f'  총 {total}장')
    print(f'  {"cx범위":>12}  {"개수":>6}  분포')
    for start in sorted(bins):
        count = len(bins[start])
        bar_len = int(bar_width * count / max_count) if max_count else 0
        bar = '#' * bar_len
        marker = '  <- 직진(150 포함)' if start <= 150 < start + bin_width else ''
        over = ''
        if max_per_bin is not None and count > max_per_bin:
            over = f'  (초과 {count - max_per_bin}장 삭제 예정)'
        print(f'  [{start:3d}-{start + bin_width - 1:3d}]  {count:6d}  {bar}{marker}{over}')


def process_section(section_dir: str, bin_width: int, max_per_bin: int, seed: int):
    paths = sorted(glob.glob(os.path.join(section_dir, '*.jpg')))
    if not paths:
        print(f'[{os.path.basename(section_dir)}] 이미지 없음 — skip')
        return

    bins = defaultdict(list)
    for p in paths:
        bins[bin_of(cx_of(p), bin_width)].append(p)

    print(f'\n[{os.path.basename(section_dir)}]')
    print_histogram(bins, bin_width, max_per_bin)

    if max_per_bin is None:
        return

    random.seed(seed)
    excluded_dir = os.path.join(section_dir, '_excluded')
    moved = 0
    for start, group in bins.items():
        if len(group) <= max_per_bin:
            continue
        random.shuffle(group)
        excess = group[max_per_bin:]
        os.makedirs(excluded_dir, exist_ok=True)
        for p in excess:
            shutil.move(p, os.path.join(excluded_dir, os.path.basename(p)))
        moved += len(excess)

    if moved:
        print(f'  → {moved}장을 {excluded_dir} 로 이동함 (완전 삭제 아님, 되돌리려면 다시 옮기면 됨)')
    else:
        print('  → 초과분 없음, 이동한 파일 없음')


def main():
    parser = argparse.ArgumentParser(
        description='JetRacer 데이터셋 cx 분포 확인 및 직진 편중 다운샘플링')
    parser.add_argument('--data_dir', required=True,
                        help='Root of dataset (contains section_N/ folders)')
    parser.add_argument('--section', default='all',
                        help='Section id 1-5, or "all"')
    parser.add_argument('--bin_width', type=int, default=10,
                        help='히스토그램 bin 폭 (cx 기준, default: 10)')
    parser.add_argument('--max_per_bin', type=int, default=None,
                        help='bin당 최대 개수. 지정 안 하면 분포만 출력하고 아무것도 안 지움.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    sections = list(range(1, 6)) if args.section == 'all' else [int(args.section)]
    for sec in sections:
        section_dir = os.path.join(args.data_dir, f'section_{sec}')
        process_section(section_dir, args.bin_width, args.max_per_bin, args.seed)


if __name__ == '__main__':
    main()
