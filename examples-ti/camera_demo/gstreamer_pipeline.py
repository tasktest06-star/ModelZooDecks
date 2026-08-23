"""
GStreamer camera pipeline for TI EdgeAI.
Mirrors the pattern used by TexasInstruments/edgeai-gst-apps.
Falls back to OpenCV when GStreamer/TIDL not available.
"""

import argparse
import time
import numpy as np
import cv2

GSTREAMER_PIPELINES = {
    "classification": (
        "v4l2src device={device} ! "
        "video/x-raw,width=1280,height=720,framerate=30/1 ! "
        "videoconvert ! video/x-raw,format=RGB ! "
        "videoscale ! video/x-raw,width={width},height={height} ! "
        "appsink name=sink max-buffers=2 drop=true sync=false"
    ),
    "detection": (
        "v4l2src device={device} ! "
        "video/x-raw,width=1920,height=1080,framerate=30/1 ! "
        "tiovxldc sensor-name=SENSOR_SONY_IMX390_RCM_UB953_D3 "
        "dcc-isp-file=/opt/imaging/imx390/dcc_viss.bin ! "
        "tiovxmultiscaler name=sc sc.src_0 ! queue ! "
        "tiovxdlpreproc out-pool-size=4 channel-order=0 data-type=float32 "
        "mean-0=0 mean-1=0 mean-2=0 scale-0=0.003921 scale-1=0.003921 scale-2=0.003921 ! "
        "tidlinferer model={model_path} ! "
        "appsink name=sink drop=true sync=false"
    ),
    "segmentation": (
        "v4l2src device={device} ! "
        "video/x-raw,width=1280,height=720,framerate=30/1 ! "
        "tiovxdlpreproc out-pool-size=4 channel-order=0 data-type=float32 "
        "mean-0=128 mean-1=128 mean-2=128 scale-0=0.00784 scale-1=0.00784 scale-2=0.00784 ! "
        "tidlinferer model={model_path} ! "
        "tiovxdlcolorblend target=1 ! kmssink sync=false"
    ),
}

MODEL_PRESETS = {
    "yolox_s_lite":       {"task": "detection",      "width": 640, "height": 640},
    "mobilenet_v2_lite":  {"task": "classification",  "width": 224, "height": 224},
    "deeplabv3plus_lite": {"task": "segmentation",    "width": 512, "height": 512},
    "yoloxpose_s_lite":   {"task": "detection",       "width": 640, "height": 640},
}


def run_python_demo(model_name: str, source: str, model_path: str):
    """OpenCV fallback demo — runs on any host, shows the GStreamer pipeline on exit."""
    preset = MODEL_PRESETS.get(model_name, {"task":"classification","width":224,"height":224})
    w, h = preset["width"], preset["height"]

    cap_src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_src)
    use_synth = not cap.isOpened()
    if use_synth:
        print(f"Using synthetic source (could not open {source})")

    print(f"\nStarting {preset['task']} demo | Model: {model_name} | Input: {w}x{h}")
    print("(Python fallback — on TI hardware replace with GStreamer pipeline)\n")

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            if use_synth:
                frame = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
                time.sleep(1/30)
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            frame_count += 1
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            cv2.putText(frame, f"Model: {model_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {frame_count}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            cv2.putText(frame, "Run on TI AM68A for actual TIDL inference", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,255), 1)

            cv2.imshow(f"TI EdgeAI -- {preset['task']}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        elapsed = time.time() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s = {fps:.1f} FPS (CPU sim)")
        print("\n--- GStreamer pipeline for TI hardware ---")
        task = preset["task"]
        if task in GSTREAMER_PIPELINES:
            print("gst-launch-1.0 " + GSTREAMER_PIPELINES[task].format(
                device="/dev/video0", width=w, height=h, model_path=model_path))


def main():
    parser = argparse.ArgumentParser(description="TI EdgeAI Camera Demo")
    parser.add_argument("--model",      default="yolox_s_lite",
                        choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--source",     default="0")
    parser.add_argument("--print_gst",  action="store_true",
                        help="Print GStreamer pipeline and exit")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = f"/opt/model_zoo/{args.model}"

    if args.print_gst:
        preset = MODEL_PRESETS[args.model]
        print(GSTREAMER_PIPELINES.get(preset["task"], "No pipeline defined").format(
            device="/dev/video0", width=preset["width"],
            height=preset["height"], model_path=args.model_path))
        return

    run_python_demo(args.model, args.source, args.model_path)


if __name__ == "__main__":
    main()
