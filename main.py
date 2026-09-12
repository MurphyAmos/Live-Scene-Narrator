import os
import time
import json

import cv2

from ultralytics import YOLOE

from google import genai
from vectordb import Memory

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
#more params 
model = YOLOE("yoloe-26l-seg-pf.pt")
#set up our local db 
memory = Memory(memory_file="./my_local_vectordb")

#instead of using response.id we can just input our previous summary directly into our prompt 
previous_scene_summary = ""
def prompt_summary(data_summary):
    global previous_scene_summary
    #response for vectordb is based upon current data with previous data as context 
    response = client.interactions.create(
        model="gemini-3.5-flash-lite",                            
        system_instruction="""You are a temporal visual-scene interpretation model.
            You will receive multiple JSON observations from the same ongoing video or live camera scene. Interpret them together over time and return a concise, natural-language description of what a human observer would most likely say is happening.

            Your output will be stored in a vector database and may later be retrieved as memory for another language model. Therefore, each description must be understandable on its own, semantically informative, and useful even when retrieved without the observations that produced it.

            Use previous interactions only as temporal memory. The newest observations are always the strongest evidence. Re-evaluate the scene on every call rather than blindly continuing the previous interpretation.

            Each observation may contain:

            * `frame_id`: local frame number; may reset or repeat
            * `timestamp`: observation time
            * `detections`
            * `track_id`: temporary object association
            * `class`: detector label
            * `confidence`
            * `position`: normalized `x`, `y`, `width`, `height`

            Temporal reasoning:

            * Use observation order and timestamps to determine progression.
            * Never infer sequence length from `frame_id`.
            * Repeated or reset frame IDs do not imply duplicate observations.
            * Use `track_id` only as a temporary object association, not permanent identity.
            * When multiple observations exist, compare position and apparent size to infer rough movement.
            * Increasing apparent size may indicate movement toward the camera.
            * Decreasing apparent size may indicate movement away from the camera.
            * A change in horizontal or vertical position may indicate movement across the scene.
            * Objects remaining in roughly the same location over several observations are probably stationary.
            * Objects appearing near an image edge and later disappearing may have exited the visible scene.
            * Objects newly appearing from an edge may have entered the visible scene.
            * Only say that movement cannot be determined when exactly one observation is available.

            Spatial interpretation:

            * `x ≈ 0`: left side
            * `x ≈ 0.5`: center
            * `x ≈ 1`: right side
            * `y ≈ 0`: upper image
            * `y ≈ 1`: lower image

            Detector outputs are noisy. Interpret persistent patterns rather than individual predictions:

            * Prefer repeated and persistent detections over isolated ones.
            * Ignore isolated low-confidence detections unless supported elsewhere.
            * Merge overlapping or semantically similar labels when they likely refer to the same physical object.
            * Normalize fluctuating labels into a stable broader category whenever possible.
            * Never invent objects merely to make the scene coherent.

            Examples:

            * `television`, `ktv`, `tv sitcom`, and similar labels in one fixed area → television/display
            * `cup`, `bowl`, `plate`, and similar overlapping labels → container/dish when the exact type is unclear
            * repeated `glasses` detections around a person's face → glasses
            * changing furniture labels in one stable location → use the most defensible broader furniture category

            If a television or monitor stays fixed while detections within its area change, infer that the display may be showing changing visual content. Do not mistake people or objects shown on a screen for physical objects in the room when the spatial evidence suggests they belong to the display.

            Infer environments conservatively. Persistent combinations such as a person, couch, bed, pillow, television, desk, chair, computer, or cup may suggest an indoor living or work space. Use cautious phrasing such as "appears to be" when the environment is uncertain.

            Describe actions conservatively. Prefer descriptions such as:

            "A person near the camera shifts position and leans slightly."

            Do not claim a specific action such as picking something up, speaking, eating, opening something, or interacting with another object unless the observations strongly support it.

            Vector-memory requirements:

            * Make every description understandable without needing the previous summary.
            * Explicitly name important entities instead of relying heavily on pronouns.
            * Prefer stable, concrete terminology such as "person", "television", "chair", "vehicle", or "bicycle".
            * Use the same general term for the same type of object whenever possible.
            * Naturally include the likely environment when it is reasonably supported.
            * Include meaningful spatial information such as left, center, right, foreground, or background when useful.
            * Include important temporal changes such as entering, leaving, approaching, moving away, crossing the scene, shifting, appearing, or disappearing.
            * Mention important stationary context when it helps identify the scene later.
            * Preserve distinctive details that could make this memory useful for semantic retrieval.
            * Avoid vague references such as "it", "that thing", "something", "they", or "there" when the referenced object can instead be named clearly.
            * Avoid filler, unnecessary adjectives, speculation, storytelling, or poetic language.
            * Do not repeat long lists of detected objects.
            * Do not write keywords, tags, database metadata, headings, or structured fields. Encode useful retrieval information naturally into the prose.

            When previous scene state:


            * Use it to understand continuity and longer-term changes.
            * Do not assume old information is still true when the newest observations contradict it.
            * Retain older scene details only when the newest evidence still supports them.
            * Prefer describing what is currently happening while briefly preserving meaningful changes from the immediate past.
            * If a previously important object or person is no longer visible, mention that it appears to have left only when the observations support that conclusion.

            Focus on:

            * the likely environment
            * important people or objects currently present
            * meaningful movement or changes
            * entry, exit, approach, retreat, crossing, leaning, or shifting
            * important stationary scene elements
            * meaningful changes relative to recent context
            * the likely overall situation

            Do not narrate every detected object.

            Do not mention frame IDs, timestamps, coordinates, track IDs, confidence values, JSON, bounding boxes, vector databases, embeddings, detection batches, or the interpretation process unless explicitly asked.

            Do not discuss detector instability unless it materially affects what can reasonably be concluded.

            Output rules:

            * Return ONLY the scene description.
            * Return one coherent paragraph.
            * Usually use 2-5 sentences and approximately 50-120 words.
            * Make the paragraph self-contained.
            * Use concrete nouns and clear actions.
            * Sound conversational and observational rather than technical.
            * Give more weight to the newest observations than older context.
            * Preserve older context only when the new observations continue to support it.
            * Do not repeat the previous description verbatim unless essentially nothing has changed.
            * Prefer information-rich sentences over unnecessary verbosity.

        """,
        input = f"""
            Previous scene state:
            {previous_scene_summary}

            New observations:
            {data_summary}

            Update the scene description based on the new observations.
            """
    )
    #load response into memory as a "previous response" for next prompt context
    previous_scene_summary = response.output_text
    #load scence context into external vectordb
    memory.save(previous_scene_summary, data_summary)
    return response

camera = cv2.VideoCapture(0)
#get resolution    
src_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
src_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

#force scale each video...
target_max = 240
##if width is bigger go on width else go on height for verticle
if src_width >= src_height:
    scale = target_max / src_width
else:
    scale = target_max / src_height
width = int(src_width * scale)
height = int(src_height * scale)

def get_frame_info():
    os.system("cls" if os.name == "nt" else "clear")
    fc = 2
    count,preview = 0, True
    frame_list = []

    while True:
        count+=1
        if count % fc != 0:
            success = camera.grab()
            if not success:
                break
            continue
        success, frame = camera.read()
        #if we get a frame
        if success:
            frame = cv2.resize(frame,(width, height), interpolation=cv2.INTER_NEAREST)
            #track detections from image
            results = model.track(frame,persist=True,verbose=False)
            frame_timestamp = time.time()
            #return out current frame data dict
            frame_data = {
                "frame_id": count,
                "timestamp": frame_timestamp,
                "detections": []
            }
            #for the results we have return out the data for each result
            for result in results: 
                for i, cls_id in enumerate(result.boxes.cls):
                    cls_id = int(cls_id)
                    confidence = float(result.boxes.conf[i])
                    #if low confidence skip
                    if confidence < .2:
                        continue    
                    #else take confidence the current ID the class and position and write it to jsonl file
                    x, y, w, h = result.boxes.xywhn[i].cpu().tolist()
                    x1, y1, x2, y2 = result.boxes.xyxyn[i].cpu().tolist()
                    
                    area = w * h
                    aspect_ratio = w / h if h != 0 else 0

                    detect = {
                        "track_id": (
                            int(result.boxes.id[i].item())
                            if result.boxes.id is not None
                            else None
                        ),

                        "class": result.names[cls_id],
                        "class_id": cls_id,
                        "confidence": confidence,

                        "position": {
                            "center_x": x,
                            "center_y": y,
                            "width": w,
                            "height": h,

                            "left": x1,
                            "top": y1,
                            "right": x2,
                            "bottom": y2
                        },

                        "spatial": {
                            "area": area,
                            "aspect_ratio": aspect_ratio,

                            "horizontal_region": (
                                "left" if x < .33
                                else "right" if x > .66
                                else "center"
                            ),

                            "vertical_region": (
                                "top" if y < .33
                                else "bottom" if y > .66
                                else "middle"
                            )
                        }
                    }
                    frame_data["detections"].append(detect)
                
            with open("Description.jsonl", "a") as f:
                f.write(json.dumps(frame_data) + "\n")
            frame_list.append(frame_data)
            #every 30/fc frames we generate a summary prompt for the external vector embedding 
            if count % 30 == 0:
                y = prompt_summary(frame_list).output_text
                #clear for a new batch of frame data
                frame_list.clear()  
        #preview, if q is pressed release
        if preview:
            cv2.imshow("YOLOE Segmentation",results[0].plot())
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):                
                camera.release()
                cv2.destroyAllWindows() 
                break
if __name__ == "__main__":
    get_frame_info()

