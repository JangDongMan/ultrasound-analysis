#!/usr/bin/env python3
"""
classify_faces.py
usdata/data 의 CSV 파일을 얼굴 부위 8개 카테고리로 분류하여
classification/ 디렉토리에 심볼릭 링크를 생성합니다.

사용법:
  python classify_faces.py           # 분류 실행 (이미 있는 링크는 덮어쓰기)
  python classify_faces.py --dry-run # 분류 결과만 출력 (링크 미생성)
  python classify_faces.py --stats   # 카테고리별 통계만 출력
"""

import os
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

# ─── 경로 ─────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
USDATA      = SCRIPT_DIR / "usdata" / "data"
CLASS_DIR   = SCRIPT_DIR / "classification"

# ─── 8개 카테고리 ──────────────────────────────────────────────
CATEGORIES = ["이마", "관자놀이", "볼", "입술주위", "광대뼈", "턱", "턱선목"]

# ─── 분류 규칙 (우선순위 높은 순 — 복합/특수 키워드 먼저) ──────
# (매칭 키워드, 카테고리)
RULES = [
    # ── 턱선목 (jaw line + neck) ──────────────────────────────
    ("턱선목",                      "턱선목"),
    ("잎술옆 턱선 바로아래 목",       "턱선목"),
    ("귀 아래쪽 턱선",               "턱선목"),
    ("턱선",                        "턱선목"),   # 턱선_우, 턱선_좌

    # ── 턱 (chin) ────────────────────────────────────────────
    ("아래잎술 옆 & 턱선 위",         "턱"),
    ("턱 중앙",                     "턱"),
    ("턱",                          "턱"),       # 턱_우, 턱_좌, 턱_중

    # ── 관자놀이 ─────────────────────────────────────────────
    ("이마 끝 관자놀이",             "관자놀이"),
    ("광대활 위 관자놀이",           "관자놀이"),
    ("관자놀이",                    "관자놀이"),

    # ── 이마 ─────────────────────────────────────────────────
    ("이마",                        "이마"),

    # ── 광대뼈 ───────────────────────────────────────────────
    ("광대뼈",                      "광대뼈"),
    ("광대활",                      "광대뼈"),

    # ── 눈주위 → 관자놀이 ────────────────────────────────────
    ("눈 아래 돌출광대뼈",           "관자놀이"),
    ("눈썹옆",                      "관자놀이"),
    ("눈썹 옆",                     "관자놀이"),
    ("눈옆",                        "관자놀이"),
    ("눈 아래",                     "관자놀이"),

    # ── 입술주위 ─────────────────────────────────────────────
    ("잎술과 인중",                  "입술주위"),
    ("잎술옆",                      "입술주위"),   # 잎술옆_우, 잎술옆_좌
    ("아래잎술",                    "입술주위"),
    ("콧구멍",                      "입술주위"),
    ("코옆",                        "입술주위"),   # 코옆_우, 코옆_좌

    # ── 볼 ───────────────────────────────────────────────────
    ("돌출광대뼈 바로아래 볼",        "볼"),
    ("아래귀 옆 볼",                 "볼"),
    ("이하선",                      "볼"),
    ("볼옆",                        "볼"),        # 볼옆_우, 볼옆_좌
    ("볼",                          "볼"),        # 볼_우, 볼_좌

]


def classify_position(pos_text: str) -> str:
    """부위 텍스트를 8개 카테고리 중 하나로 분류. 미매칭 시 '미분류' 반환."""
    for keyword, category in RULES:
        if keyword in pos_text:
            return category
    return "미분류"


def extract_position(filename: str) -> str:
    """파일명에서 부위 텍스트 추출.
    패턴: ...(숫자)부위텍스트_[MF].csv
    반환: '부위텍스트' (없으면 빈 문자열)
    """
    m = re.search(r'\(\d+\)(.+?)_[MF]\.csv$', filename)
    if m:
        return m.group(1).strip()
    return ""


def make_link_name(csv_path: Path) -> str:
    """심볼릭 링크 이름 생성 — 중복 없도록 상위 디렉토리 이름 포함."""
    parts = csv_path.relative_to(USDATA).parts  # e.g. ('260528', '조세일', 'xxx.csv')
    return "__".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="링크 생성 없이 분류 결과만 출력")
    parser.add_argument("--stats", action="store_true",
                        help="카테고리별 통계만 출력")
    args = parser.parse_args()

    if not USDATA.exists():
        print(f"ERROR: 데이터 디렉토리 없음: {USDATA}")
        return

    # 디렉토리 초기화 (기존 링크 전체 삭제 후 재생성 — 카테고리 변경 시 오염 방지)
    if not args.dry_run and not args.stats:
        CLASS_DIR.mkdir(exist_ok=True)
        for cat in CATEGORIES:
            cat_dir = CLASS_DIR / cat
            if cat_dir.exists():
                for f in cat_dir.iterdir():
                    f.unlink(missing_ok=True)
            cat_dir.mkdir(exist_ok=True)

    csv_files = sorted(USDATA.rglob("*.csv"))
    print(f"전체 CSV: {len(csv_files)}개\n")

    rows = []
    count   = defaultdict(int)
    unclassified = []

    for csv_path in csv_files:
        pos_text = extract_position(csv_path.name)
        if not pos_text:
            continue  # (00) 또는 패턴 불일치 스킵

        category = classify_position(pos_text)

        if category == "미분류":
            unclassified.append((str(csv_path.relative_to(SCRIPT_DIR)), pos_text))
            continue

        count[category] += 1
        rows.append({
            "file":     str(csv_path.relative_to(SCRIPT_DIR)),
            "position": pos_text,
            "category": category,
        })

        if not args.dry_run and not args.stats:
            link_path = CLASS_DIR / category / make_link_name(csv_path)
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(csv_path.resolve())

    # ── 통계 출력 ───────────────────────────────────────────
    print("=" * 40)
    print(f"{'카테고리':<10} {'개수':>6}  {'비율':>6}")
    print("-" * 40)
    total = sum(count.values())
    for cat in CATEGORIES:
        n = count[cat]
        pct = n / total * 100 if total else 0
        print(f"  {cat:<8} {n:>6}  {pct:>5.1f}%")
    print("-" * 40)
    print(f"  {'합계':<8} {total:>6}")
    if unclassified:
        print(f"\n미분류: {len(unclassified)}개")
        for fpath, pos in unclassified[:10]:
            print(f"  [{pos}] {fpath}")
        if len(unclassified) > 10:
            print(f"  ... 외 {len(unclassified)-10}개")

    if args.dry_run or args.stats:
        return

    # ── summary.csv 저장 ────────────────────────────────────
    summary_path = CLASS_DIR / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "position", "category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n저장 위치  : {CLASS_DIR}")
    print(f"요약 파일  : {summary_path}")
    print("\n각 카테고리 디렉토리에 심볼릭 링크가 생성되었습니다.")


if __name__ == "__main__":
    main()
