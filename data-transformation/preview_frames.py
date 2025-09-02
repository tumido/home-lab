#!/usr/bin/env python3
"""
preview_last_and_first_frames.py

Preview the LAST frame of the first MP4 (left) and the FIRST frame of the second MP4 (right).

Usage:
  python preview_last_and_first_frames.py [workdir] --a videoA.mp4 --b videoB.mp4

Other Hotkeys:
  q / ESC : quit
  n       : next pair in the folder (shift by one, second video becomes first)
  p       : previous pair in the folder (shift by one, first video becomes second)
"""
# pylint: disable=no-member,missing-function-docstring,missing-class-docstring
from enum import Enum
from pathlib import Path
import argparse
import os
import glob
from typing import List, NamedTuple
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def list_mp4s_in_dir(dirpath: str) -> List[str]:
    files = glob.glob(os.path.join(dirpath, "*.mp4"))
    # Case-insensitive sort by filename (not full path)
    files.sort(key=lambda p: os.path.basename(p).lower())
    return files


def read_first_frame(path: Path):
    cap = cv2.VideoCapture(filename=path.as_posix())
    if not cap.isOpened():
        return None
    for pos in (1, 2, 3, 5, 10, 20):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if ok and frame is not None and cv2.countNonZero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 100000:
            print(f"first frame is: {pos}")
            cap.release()
            return frame
    cap.release()
    return None


def read_last_frame(path: Path):
    cap = cv2.VideoCapture(path.as_posix())
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        # Try a few from the end in case the absolute last frame is a keyframe gap
        for back in (1, 2, 3, 5, 10):
            pos = max(total - back, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if ok and frame is not None:
                cap.release()
                return frame
    # Fallback: iterate
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last = frame
    cap.release()
    return last


class Image(NamedTuple):
    label: str
    first_frame: cv2.Mat
    last_frame: cv2.Mat


class BlankPlace(Enum):
    LEFT = 0
    RIGHT = 1
    NO_BLANK = 2


TARGET_H = 200


def hconcat_resize(im_list, interpolation=cv2.INTER_CUBIC):
    im_list_resize = [
        cv2.resize(
            im,
            (int(im.shape[1] * TARGET_H / im.shape[0]), TARGET_H),
            interpolation=interpolation,
        )
        for im in im_list
    ]
    return cv2.hconcat(im_list_resize)


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [
        cv2.resize(
            im,
            (w_min, int(im.shape[0] * w_min / im.shape[1])),
            interpolation=interpolation,
        )
        for im in im_list
    ]
    return cv2.vconcat(im_list_resize)


def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [
        hconcat_resize(im_list_h, interpolation=interpolation)
        for im_list_h in im_list_2d
    ]
    return vconcat_resize_min(im_list_v, interpolation=interpolation)


def resize(img: cv2.Mat):
    return cv2.resize(img, (int(img.shape[1] * TARGET_H / img.shape[0]), TARGET_H))


def compare(a: cv2.Mat, b: cv2.Mat):
    a_grayscale = cv2.cvtColor(resize(a), cv2.COLOR_BGR2GRAY)
    b_grayscale = cv2.cvtColor(resize(b), cv2.COLOR_BGR2GRAY)

    score, diff = structural_similarity(a_grayscale, b_grayscale, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    filled = resize(a).copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            cv2.drawContours(filled, [c], 0, (0, 255, 0), -1)

    return score, filled


TEXT_ARGS = (cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def label_image(image: cv2.Mat, label: str):
    new_image = image.copy()
    cv2.putText(new_image, label, (10, 25), *TEXT_ARGS)
    return new_image



def display(images: List[Image], ends=None):
    stack = []
    for idx, image in enumerate(images):
        first = label_image(resize(image.first_frame), "FIRST")
        last = label_image(resize(image.last_frame), "LAST")
        if ends:
            stack.append([first, last])
        elif idx == 0:
            blank = np.zeros(images[0].first_frame.shape, dtype=np.uint8)
            stack.append([first, last, blank])
        elif idx < len(images):
            score, filled = compare(images[idx-1].last_frame, image.first_frame)
            banner = np.zeros((40, first.shape[1] * 3, 3), dtype=np.uint8)
            banner_text = f"{images[idx-1].label} -> {image.label}:  {(score * 100):.3f}%"
            cv2.putText(banner, banner_text, (10, 25), *TEXT_ARGS)
            stack.append([banner])
            stack.append([filled, first, last])


    return concat_tile_resize(stack)


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Folder to analyze")
    parser.add_argument("--a", help="Video to take LAST frame from (left)")
    parser.add_argument("--b", help="Video to take FIRST frame from (right)")
    parser.add_argument("--ends", help="Compare folder ends", action="store_true")
    args = parser.parse_args()

    playlist = list_mp4s_in_dir(args.path)
    if not playlist:
        print(f"No .mp4 files found in {args.path}")
        return

    # Try to align indices to current selection (if present in the same dir).
    def index_or_none(name):
        try:
            return playlist.index(os.path.abspath(name))
        except ValueError:
            return 0

    a_path: Path = Path(args.path) / (args.a + ".mp4" if args.a else playlist[0])
    b_path: Path = Path(args.path) / (
        args.b + ".mp4"
        if args.b
        else playlist[index_or_none(a_path) + 1] if not args.ends else playlist[-1]
    )

    right_idx = index_or_none(b_path)
    left_idx = index_or_none(a_path)

    while True:
        print("left (LAST): ", a_path, left_idx)
        print("right (FIRST):", b_path, right_idx)

        if args.ends:
            i = Image(
                os.path.basename(a_path),
                read_first_frame(a_path),
                read_last_frame(b_path),
            )
            combo = display([i], os.path.basename(b_path))
        else:
            # Load frames per current selection
            a = Image(
                os.path.basename(a_path),
                read_first_frame(a_path),
                read_last_frame(a_path),
            )
            b = Image(
                os.path.basename(b_path),
                read_first_frame(b_path),
                read_last_frame(b_path),
            )

            combo = display([a, b])
        cv2.imshow(args.path, combo)

        # Use waitKeyEx to capture arrow keys reliably
        k = cv2.waitKeyEx(0)

        if k in (27, ord("q")):  # ESC or q
            break

        elif k == ord("s"):
            out = f"preview_{os.path.basename(a_path)}__{os.path.basename(b_path)}.png"
            cv2.imwrite(out, combo)
            print(f"Saved {out}")

        if args.ends:  # Do not walk if ends requested
            continue

        elif k == ord("n"):
            if len(playlist) == 1:
                pass
            elif right_idx < len(playlist) - 1:
                a_path = b_path
                left_idx = right_idx
                right_idx = right_idx + 1
                b_path = Path(playlist[right_idx])
        elif k == ord("p"):
            if len(playlist) == 1:
                pass
            elif left_idx > 0:
                b_path = a_path
                right_idx = left_idx
                left_idx = left_idx - 1
                a_path = Path(playlist[left_idx])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
