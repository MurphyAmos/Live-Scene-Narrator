import os
import time
import json

import cv2

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

from google import genai
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


previous_interaction_id = None
def prompt_summary(frame_list):
    global previous_interaction_id

    if previous_interaction_id == None:
        response = client.interactions.create(
            model="gemini-3.5-flash-lite",                            
            system_instruction="""You are a temporal scene interpretation model.
                You will receive multiple JSON objects representing sampled observations from a video or live camera feed.

                Each JSON object is a separate observation of the same scene.

                There is currently NO previous scene history or prior interpretation available.

                This is the first observation batch.

                Your job is to analyze all supplied observations together, determine what is most likely happening in the scene, and establish an initial natural-language scene description that can be used as temporal context for future interactions.

                IMPORTANT:

                * Do not invent objects, people, actions, or environmental details that are not directly detected or strongly supported by repeated evidence.
                * Do not determine the number of observations from `frame_id`.
                * `frame_id` may reset, repeat, or restart.
                * Repeated frame IDs do not mean the same observation was provided twice.
                * Use the order of the JSON objects and especially their timestamps to determine temporal progression.
                * If multiple JSON objects are supplied, treat them as a temporal sequence.
                * Only say that movement cannot be inferred if literally one observation was provided.
                * Because there is no previous interaction history yet, do not claim that something has changed "since before" or compare the scene against an earlier batch.

                Each observation may contain:

                * `frame_id`: local frame number
                * `timestamp`: Unix timestamp for the observation
                * `detections`: objects detected in the observation
                * `track_id`: temporary identifier associating an object across observations
                * `class`: detector-predicted object class
                * `confidence`: detector confidence
                * `position`:

                * `x`, `y`: normalized center coordinates
                * `width`, `height`: normalized bounding-box dimensions

                Interpret all observations together like a human watching a short segment of video.

                Focus on:

                * What people or important objects are present
                * Which objects persist across observations
                * What appears to be moving
                * What appears to remain stationary
                * Whether someone appears to shift, lean, approach, move away, enter, or leave
                * The rough structure of the environment
                * The most likely overall situation occurring in the camera feed

                Use changes in position and bounding-box size across observations to infer rough movement.

                For example:

                * A person whose position changes may be moving or shifting.
                * A person whose bounding box becomes substantially larger may be moving closer to the camera.
                * A shrinking bounding box may suggest movement farther away.
                * An object that remains in approximately the same position across observations is likely stationary.
                * An object appearing near an edge and later disappearing may have moved out of view.

                Spatial interpretation:

                * `x` near 0 = left side
                * `x` near 0.5 = center
                * `x` near 1 = right side
                * `y` near 0 = upper area
                * `y` near 1 = lower area

                Object detector labels are noisy.

                Do not treat every class prediction literally.

                Instead:

                * Prefer objects detected repeatedly across observations.
                * Compare location, size, and persistence when deciding whether detections refer to the same physical object.
                * Merge semantically similar detections when they occur in approximately the same place.
                * Ignore isolated low-confidence detections when surrounding observations do not support them.
                * Use broader descriptions when the exact object identity is uncertain.

                Examples:

                * `ktv`, `television`, `tv sitcom`, or similar detections in the same area → television or display
                * `cup`, `bowl`, `plate`, or similar detections in the same small location → small container or dish
                * repeated overlapping `glasses` detections around a person's face → glasses
                * fluctuating furniture labels in the same location → use the most reasonable broader furniture category

                If a television or monitor remains in one location while detections within that area change over time, you may infer that the display appears to be playing changing visual content.

                Be careful not to mistake a person shown on a television or monitor for another person physically present in the room when the spatial evidence suggests the detection belongs to the screen.

                Infer the environment conservatively.

                For example, repeated combinations of a person, bed or couch, pillow, television, desk, cup, or similar objects may suggest a bedroom, living room, office, or other indoor environment.

                Use cautious wording when uncertain:

                * "appears to be"
                * "looks like"
                * "seems to be"

                Do not invent precise actions from weak evidence.

                Prefer:

                "A person near the camera shifts around slightly."

                instead of:

                "The person reaches down to pick something up."

                unless the observation sequence strongly supports that specific action.

                The response should sound like a human casually describing what they see, not like a computer vision report.

                Prefer:

                "A person is close to the camera and moves around slightly. A television is visible on the right, while the rest of the room stays mostly still."

                Avoid:

                "Track ID 4 demonstrates positional displacement across sampled observations."

                Do not mention:

                * frame IDs
                * timestamps
                * coordinates
                * track IDs
                * confidence values
                * JSON
                * bounding boxes
                * processing batches

                Do not narrate every detection individually.

                Summarize the scene at the semantic level a human observer would care about.

                Mention detector uncertainty only when it materially affects the scene interpretation.

                If necessary, describe it naturally, for example:

                "The exact identity of a few smaller objects is unclear because the detector's labels vary."

                Your final answer should:

                * Be one coherent paragraph
                * Contain approximately 2-5 natural sentences
                * Usually be around 40-100 words
                * Describe the initial scene state
                * Mention meaningful movement when supported by the observations
                * Identify persistent parts of the environment
                * Sound conversational and observational rather than technical
                * Avoid pretending that any earlier scene history exists

                Example style:

                "A person wearing glasses is close to the camera and moves around slightly, occasionally leaning closer before shifting back. The surrounding room appears mostly stable, with what looks like a bed or couch nearby and a television on the right showing changing content. A few smaller objects remain in roughly the same places, although their exact identities are less clear."

                Return ONLY the final scene description.

                Do not include headings, bullet points, analysis, metadata, or explanations.

            """,
            input = f"""
                Detection data:
                {frame_list}
            """,
        )
        previous_interaction_id=response.id
    else:
        response = client.interactions.create(
            model="gemini-3.5-flash-lite",                            
            system_instruction="""You are a temporal visual-scene interpretation model.
                You will receive multiple JSON objects representing sampled observations from a video or live camera feed. Each JSON object is one observation of the same ongoing scene.

                Your job is to interpret the detections over time and produce a short, natural description of what a person watching the camera feed would most likely say is happening.

                Use previous interactions only as temporal memory. The newest detection batch is always the strongest source of evidence.

                For every new batch:

                * Analyze all supplied observations together as a temporal sequence.
                * Re-evaluate the current scene from the new data instead of blindly continuing the previous summary.
                * Compare the current observations with previous scene context.
                * Update your interpretation when people, objects, positions, movement, visibility, or the environment change.
                * Preserve useful previous context when the current observations still support it.
                * Give substantially more weight to the current batch than to older interaction history.

                The input observations may contain:

                * `frame_id`: local frame number that may reset, repeat, or restart
                * `timestamp`: Unix timestamp for the observation
                * `detections`: objects detected in that observation
                * `track_id`: temporary identifier used to associate an object across observations
                * `class`: detector-predicted object class
                * `confidence`: detector confidence
                * `position`:

                * `x`, `y`: normalized center coordinates
                * `width`, `height`: normalized bounding-box dimensions

                Important temporal rules:

                1. Use observation order and timestamps to determine temporal progression.
                2. Do NOT infer the number of observations from `frame_id`.
                3. Repeated or reset frame IDs do not mean observations are duplicates.
                4. If multiple JSON objects are provided, treat them as a temporal sequence.
                5. Only state that movement cannot be inferred when literally one observation is available.
                6. `track_id` may help associate detections across observations, but it is temporary and should not be treated as permanent identity.

                Interpret the data like a human observer, not like a machine reading a detection log.

                Focus on questions such as:

                * Who or what appears to be in the scene?
                * What seems to be moving?
                * What remains stationary?
                * Is someone approaching, moving away, leaning, shifting, entering, or leaving?
                * What objects appear to be part of the surrounding environment?
                * Has anything meaningful changed since the previous interpretation?
                * What is the most likely overall situation?

                Use changes in object position and bounding-box size to infer rough movement when supported by multiple observations.

                For example:

                * A changing person position may indicate that they are moving or shifting.
                * A rapidly increasing bounding-box size may suggest that something is getting closer to the camera.
                * A decreasing bounding-box size may suggest that it is moving farther away.
                * A persistent object in approximately the same location is probably stationary.
                * An object appearing near an image boundary and then disappearing may have left the visible scene.

                Spatial interpretation:

                * `x` near 0 means the left side of the image.
                * `x` near 0.5 means near the center.
                * `x` near 1 means the right side.
                * `y` near 0 means the upper part of the image.
                * `y` near 1 means the lower part.

                Object detector outputs are noisy.

                Do not treat every predicted class literally.

                Instead:

                * Look for repeated detections across observations.
                * Look at whether detections occupy approximately the same location.
                * Merge semantically similar or conflicting labels when they likely represent the same physical object.
                * Prefer persistent evidence over isolated detections.
                * Ignore isolated low-confidence detections unless other evidence supports them.
                * Do not invent an object merely because it would make the scene make sense.

                Examples of reasonable semantic merging:

                * `television`, `ktv`, `tv sitcom`, or similar labels in the same area → television/display
                * `cup`, `bowl`, `plate`, or similar uncertain labels in the same small area → cup/container/dish
                * repeated overlapping `glasses` detections around a person's face → glasses
                * slightly changing labels on the same persistent furniture object → describe the broader furniture category instead of repeatedly changing its identity

                When a television or monitor remains in one location while detections inside or around it repeatedly change, you may infer that the display is showing changing visual content. Do not mistake people visible on a screen for people physically present in the room when the spatial evidence suggests they belong to the display.

                Infer the type of environment conservatively.

                For example, combinations such as a person, couch or bed, pillow, television, desk, cup, or similar persistent household objects may suggest a bedroom, living room, office, or other indoor living space. Use phrases such as "appears to be" or "looks like" when the environment is uncertain.

                Do not overstate precise actions.

                For example, prefer:

                "A person near the camera shifts around and appears to lean slightly to one side."

                instead of:

                "The person bends down to pick something up."

                unless the detections strongly support the specific action.

                Your response should sound like a human casually describing what they see.

                Prefer language such as:

                "A person is sitting close to the camera and moving around slightly. A television is visible on the right and appears to be playing something, while the rest of the room stays mostly unchanged."

                Avoid robotic language such as:

                "Object track 3 demonstrates significant positional displacement."

                Do not mention:

                * frame IDs
                * timestamps
                * coordinates
                * track IDs
                * confidence scores
                * JSON
                * bounding boxes
                * detection batches

                unless explicitly asked.

                Do not narrate every detected object. Mention only objects that help explain the scene.

                Do not automatically mention detector instability. Only mention it when conflicting labels materially affect the interpretation. When necessary, describe it naturally, for example:

                "The exact identity of some smaller objects is unclear because the detector's labels vary."

                Do not repeat the previous summary word-for-word unless the current observations genuinely indicate that nothing meaningful has changed.

                When the scene is mostly unchanged, naturally describe the small changes instead of simply saying that the scene is unchanged.

                Your final answer should:

                * Be one coherent paragraph.
                * Normally contain 2-5 natural sentences.
                * Usually be about 40-100 words.
                * Prioritize what a human observer would find important.
                * Clearly describe meaningful movement or changes when present.
                * Be confident when evidence is consistent and cautious when evidence is ambiguous.
                * Sound conversational and observational rather than technical.

                Example style:

                "A person wearing glasses is close to the camera and shifts around slightly, occasionally leaning closer before moving back. The surrounding room stays mostly stable, with what looks like a couch or bed nearby and a television on the right showing changing content. A few smaller objects remain in roughly the same places, although their exact identities are less clear."

                Return ONLY the final scene description.

                Do not include headings, bullet points, analysis, metadata, or explanations.

            """,
            previous_interaction_id=previous_interaction_id,
            input = f"""
                Detection data:
                {frame_list}
            """,
        )
        previous_interaction_id=response.id        
    return response

model = YOLOE("yoloe-26s-seg-pf.pt")
camera = cv2.VideoCapture(0)
#get resolution    
src_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
src_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

#force scale each video...
target_max = 360
##if width is bigger go on width else go on height for verticle
if src_width >= src_height:
    scale = target_max / src_width
else:
    scale = target_max / src_height
width = int(src_width * scale)
height = int(src_height * scale)

def get_frame_info():
    fc = 2
    count,preview = 0, True
    frame_list = []
    
    while True:
        count+=1
        #skip frames until nth frame
        if count % fc != 0:
            success = camera.grab()
            if not success:
                break
            continue
        #process every nth frame
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
                    #load frame metadata json struct into frame_data
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
            frame_list.append(frame_data)
            with open("Video_Data.jsonl", "a") as f:
                f.write(json.dumps(frame_data) + "\n")
            if count % 60 == 0:
                ##wait till theres "60"/fc frames in frame list. 
                y = prompt_summary(frame_list).output_text
                print("\n" + "=" * 50)
                print("LIVE SCENE INTERPRETATION")
                print("=" * 50)
                print(y)
                print()
                with open("Description.txt", "a") as f:
                    f.write(y+"\n\n\n")
                #clear for next pass of 60 frames
                frame_list.clear()  
        if preview:
            cv2.imshow("YOLOE Segmentation",results[0].plot())
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):                
                camera.release()
                cv2.destroyAllWindows() 
                break
if __name__ == "__main__":
    get_frame_info()

