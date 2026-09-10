# Live Scene Narrator

Captures a live camera feed, runs object detection and tracking on it, then uses an LLM to turn batches of raw detections into a running natural-language description of what's happening in the scene, updated continuously as the feed plays.

## What it does

Instead of reading raw bounding boxes, class labels, and confidence scores off a live feed, this pipeline captures frames from a webcam, runs detection and tracking on each one, and periodically hands a batch of those observations to an LLM acting as a temporal scene interpreter. The model reasons across the batch like a human watching a short clip, noticing what's moving, what's stationary, and what the overall situation looks like, and returns a short conversational paragraph instead of a technical report. Each new batch is chained to the previous one, so the description evolves over time instead of resetting from scratch every cycle.

## How it works

1. **Camera setup.** OpenCV opens the default camera, reads its native resolution, and computes a scale factor so every frame is resized down to a fixed max dimension (360px) before processing, keeping detection fast regardless of source resolution.
2. **Detection & tracking.** Each sampled frame is run through a YOLOE segmentation/tracking model, which returns per-object class, confidence, and a persistent track ID across frames.
3. **Frame sampling.** Not every captured frame is processed, every other frame is grabbed and discarded to reduce load, while the camera buffer is still drained each loop to avoid backlog.
4. **Structured detection records.** For every detection above a confidence threshold (`0.2`), the script builds a JSON-serializable record: normalized center/box coordinates, corner coordinates, area, aspect ratio, and coarse horizontal/vertical region labels.
5. **Batching.** Frame records accumulate in memory and are also appended to a `.jsonl` log file. Every 60 processed frames, the accumulated batch is flushed to the summary step and cleared.
6. **LLM scene interpretation.** The batch is sent to a Gemini model with a detailed system prompt enforcing conservative, evidence-based interpretation, no inventing objects or actions, merging noisy or conflicting class labels into broader categories, inferring rough movement from position and box-size changes, and describing the scene in 2-5 casual sentences.
7. **Output.** Each generated description is printed and appended to `Description.txt`, building up a running narrative of the session alongside the raw `Video_Data.jsonl` detection log.
8. **Preview.** A live annotated preview window shows the tracked/segmented feed; pressing `q` exits the loop and closes the window.

# Temporal Scene-Description Design

This project uses a lightweight streaming summarization workflow:

- **Sample:** Grab frames from the live feed at a fixed sub-sampling rate and run detection/tracking on each one.
- **Batch:** Accumulate a fixed window of frame observations before sending anything to the LLM.
- **Interpret:** Prompt an LLM to reason over the batch as a temporal sequence, using object persistence, position, and box-size changes to infer movement.
- **Chain:** Carry state forward via the previous interaction ID instead of re-sending prior summaries, so later batches update rather than restate the scene.

The detector output is never treated as ground truth on its own, it's noisy per-frame evidence that the LLM step is explicitly instructed to merge, filter, and interpret conservatively.


## Tech stack

- Python
- OpenCV for camera capture, resizing, and preview rendering
- YOLOE (Ultralytics) for object detection and tracking
- Gemini API for temporal scene summarization
- JSON / JSONL for structured detection logging
## Setup

Install dependencies:

```
pip install -r requirements.txt
```

Set the `GEMINI_API_KEY` environment variable to your Gemini API key before running the script.
Make sure a webcam is available at index `0`, or update `cv2.VideoCapture(0)` to the correct camera index.


## Usage

Update the following near the top of the script:

- `target_max` with the maximum resized dimension used for processing.
- `fc` with the frame-subsampling factor inside `get_frame_info()`.
- The batch flush interval if you want summaries more or less often.
- The confidence threshold used to filter out low-confidence detections.
Then Run:
```
python main.py
```

Press `q` in the preview window to stop. Detection logs are written to `Video_Data.jsonl` and scene descriptions are appended to `Description.txt` as the session runs.

## Known limitations & Next Fixes
- **Fixed batch size.** The 60-frame flush interval is a flat count, not time-based, so summary cadence shifts if processing speed varies.
- **Single camera, single track history.** The script assumes one local camera and one continuous interaction chain; there's no support for multiple feeds or resetting context mid-session without restarting.
- **No retry logic on LLM failures.** If a summary call errors out, that batch's data is lost rather than retried or cached for a later attempt.
-**No UI.** This runs entirely from the command line with a single OpenCV preview window, every tunable value has to be edited directly in the script.

## Motivation

Watching a raw detection feed of class labels and bounding boxes doesn't tell you what's actually going on, it takes you yourself going through and providing context to create a story from the data. Since I like automating things, the next step was letting a model do the translation part... continuously: sample the feed, keep enough temporal context to notice real movement instead of noise, and produce a plain-language description of the scene as it evolves through time.
