"""
Gemini Robotics-ER high-level task planner.
Looks at camera images and selects the best task prompt from a set of allowed prompts.
"""

import logging
import time
import cv2
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiPlanner:
    def __init__(self, api_key: str, high_level_task: str, allowed_prompts: list,
                 model: str = "gemini-robotics-er-1.5-preview",
                 query_interval_seconds: float = 1.0): #3.0):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.high_level_task = high_level_task
        self.allowed_prompts = allowed_prompts
        self.query_interval = query_interval_seconds
        self.last_query_time = 0.0
        self.current_prompt = None
        logger.info(f"GeminiPlanner initialized. Task: '{high_level_task}'")

        self.current_task_cube = None
        #self.current_task_attempts = 0
        #self.max_attempts_before_switch = 3
        #self.completed_cubes = set()
        self.was_at_bucket = False

    def should_query(self) -> bool:
        return (time.time() - self.last_query_time) >= self.query_interval

    def query(self, images_rgb: dict) -> str:
        """Send camera images (RGB) to Gemini, get back an allowed task prompt."""
        self.last_query_time = time.time()

        contents = []
        for cam_name, img_rgb in images_rgb.items():
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            contents.append(
                types.Part.from_bytes(data=buf.tobytes(), mime_type='image/jpeg')
            )

        current_status = ""
        if self.current_prompt:
            current_status = f"\nThe robot is currently executing: \"{self.current_prompt}\". If this task appears to still be in progress (the robot is moving toward or grasping the target object), keep the same prompt. Only change the prompt if the current task appears completed (the target cube is in the bucket or no longer visible on the table) or if the robot is idle."

        prompt = f"""You are a high-level planner for a bimanual robot arm.
Your overall goal: {self.high_level_task}

You must respond with EXACTLY one of these allowed prompts:
{chr(10).join(f'- "{p}"' for p in self.allowed_prompts)}
{current_status}

Look at the camera images showing the current scene. Based on what you see,
pick the single most appropriate next prompt to execute.
Respond with ONLY the prompt text, nothing else."""

        contents.append(prompt)

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        result = response.text.strip().strip('"')

        if result in self.allowed_prompts:
            self.current_prompt = result
            return result

        # Fuzzy match: pick allowed prompt with most word overlap
        logger.warning(f"Gemini returned unexpected prompt: '{result}', using closest match")
        best = max(self.allowed_prompts, key=lambda p: len(set(p.split()) & set(result.split())))
        self.current_prompt = best
        return best

    #def narrate_and_annotate(self, image_rgb, current_task: str) -> tuple:
    def narrate_and_annotate(self, images_rgb: dict, current_task: str) -> tuple:
        """Ask Gemini to narrate the scene and return object locations.
        Returns (narration_text, list of {"box": [y1,x1,y2,x2], "label": str})
        """
        self.last_query_time = time.time()

        #_, buf = cv2.imencode('.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        #image_part = types.Part.from_bytes(data=buf.tobytes(), mime_type='image/jpeg')
        image_parts = []
        for cam_name, img in images_rgb.items():
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            image_parts.append(types.Part.from_bytes(data=buf.tobytes(), mime_type='image/jpeg'))

        #V3: top prompt = with high view only, botttom with high and low views
        #prompt = f"""Look at this overhead image of a robot workspace. There are five cubes (red, blue, yellow, brown, pink) and one green bucket.
        prompt = f"""Look at these images of a robot workspace. The first image is an overhead view, the second is a front view. There are five cubes (red, blue, yellow, brown, pink) and one green bucket.
        
        For each cube, report: "on table" if you can see it on the table, "in bucket" if you can see it in the bucket or cannot see it at all, or "unsure" if you are not confident.

        The robot is attempting: "{current_task}"

        Return JSON only:
        {{"narration": "Red cube: on table. Blue cube: in bucket. Yellow cube: unsure. Brown cube: on table. Pink cube: in bucket.", "success": false}}

        success = true means the target cube from the task is "in bucket" in your narration. Otherwise false."""

        # #PROMPT v2
        # prompt = f"""Look at this overhead image of a robot workspace. There are five cubes (red, blue, yellow, brown, pink) and one green bucket.

        # For each cube, report: "on table", "in bucket", or "unsure" if occluded by the robot arm.
        # If a cube is not visible and the robot arm is not blocking the view, report "in bucket".

        # The robot is attempting: "{current_task}"

        # Return JSON only:
        # {{"narration": "Red cube: on table. Blue cube: in bucket. Yellow cube: unsure. Brown cube: on table. Pink cube: in bucket.", "success": false}}

        # success = true means the target cube from the task is "in bucket" in your narration. Otherwise false."""

        # #PROMPT V1
        # prompt = f"""Look at this overhead image of a robot workspace. There are five cubes (red, blue, yellow, brown, pink) and one green bucket.

        # For each cube, report: "on table", "in bucket", or "unsure" if occluded by the robot arm.
        # If a cube is not visible and the robot arm is not blocking the view, report "in bucket".

        # The robot is attempting: "{current_task}"

        # Return JSON only:
        # {{"narration": "Red cube: on table. Blue cube: in bucket. Yellow cube: unsure. Brown cube: on table. Pink cube: in bucket."}}"""

        # #PROMPT V0: this version draws the boxes
        # prompt = f"""You are observing a bimanual robot arm performing tasks on a table from an overhead camera.
        # The robot is currently attempting: "{current_task}"

        # The scene contains five cubes (red, blue, yellow, brown, pink) and one green bucket.

        # Do two things:

        # 1. NARRATE: For each of the five cubes, state whether it is on the tabletop, being held by the robot, or in the green bucket. If a cube is not visible, assume it is in the bucket.

        # 2. DETECT: Return a bounding box for every cube and the green bucket that you can see in the image.

        # Respond in this exact JSON format and nothing else:
        # {{
        # "narration": "Red cube: on table. Blue cube: in bucket. Yellow cube: being held. Brown cube: on table. Pink cube: in bucket.",
        # "objects": [
        #     {{"box": [y1, x1, y2, x2], "label": "red cube"}},
        #     {{"box": [y1, x1, y2, x2], "label": "green bucket"}}
        # ]
        # }}
        # The box coordinates are in [y1, x1, y2, x2] format normalized to 0-1000.
        # Only include objects in the "objects" list that are visible in the image."""

        response = self.client.models.generate_content(
            model=self.model,
            contents=[*image_parts, prompt], #contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_budget=128)
            )
        )

        import json
        try:
            text = response.text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(text)
            logger.info(f"Gemini raw success: {result.get('success')}")
            #return result.get("narration", ""), result.get("objects", [])
            return result.get("narration", ""), result.get("objects", []), result.get("success", False)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            #return "", []
            return "", [], False

    def detect_objects(self, images_rgb: dict, current_task: str) -> tuple:
            """Ask Gemini to narrate scene and return bounding boxes for all visible objects.
            Returns (narration_text, list of {"box": [y1,x1,y2,x2], "label": str})
            """
            self.last_query_time = time.time()

            image_parts = []
            for cam_name, img in images_rgb.items():
                _, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                image_parts.append(types.Part.from_bytes(data=buf.tobytes(), mime_type='image/jpeg'))

            prompt = f"""Look at these images of a robot workspace. The first image is an overhead view, the second is a front view.
    There are five cubes (red, blue, yellow, brown, pink) and one green bucket.

    The robot is attempting: "{current_task}"

    For each cube, report its status and return a bounding box for every cube and the green bucket you can see.

    Return JSON only:
    {{
    "narration": "Red cube: on table. Blue cube: in bucket. Yellow cube: on table. Brown cube: on table. Pink cube: in bucket.",
    "objects": [
        {{"box": [y1, x1, y2, x2], "label": "red cube"}},
        {{"box": [y1, x1, y2, x2], "label": "green bucket"}}
    ]
    }}
    The box coordinates are in [y1, x1, y2, x2] format normalized to 0-1000.
    Only include objects that are visible in the image."""

            response = self.client.models.generate_content(
                model=self.model,
                contents=[*image_parts, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    thinking_config=types.ThinkingConfig(thinking_budget=128)
                )
            )

            import json
            try:
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                result = json.loads(text)
                return result.get("narration", ""), result.get("objects", [])
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Failed to parse Gemini response: {e}")
                return "", []

    #def plan_next_task(self, image_rgb, allowed_prompts: list) -> tuple:
    def plan_next_task(self, images_rgb: dict, allowed_prompts: list) -> tuple:
        """Observe scene, decide which cube to pick up next.
        Returns (task_prompt, narration, objects)
        """
        self.last_query_time = time.time()

        # If all 5 cubes are completed, don't query further
        #if len(self.completed_cubes) >= 5:
        #    return None, "All cubes in bucket.", []
        
        # _, buf = cv2.imencode('.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        # image_part = types.Part.from_bytes(data=buf.tobytes(), mime_type='image/jpeg')
        image_parts = []
        for cam_name, img in images_rgb.items():
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            image_parts.append(types.Part.from_bytes(data=buf.tobytes(), mime_type='image/jpeg'))

        # Build context about current state
        # history = ""
        # if self.current_task_cube:
        #     history = f"\nThe robot has been trying to pick up the {self.current_task_cube} cube for {self.current_task_attempts} attempt(s)."
        # if self.completed_cubes:
        #     history += f"\nThese cubes have been successfully placed in the bucket already: {', '.join(self.completed_cubes)}."
        #{history} line 3 in prompt

        prompt = f"""Look at these images of a robot workspace. The first image is an overhead view, the second is a front view.
        There are five cubes (red, blue, yellow, brown, pink) and one green bucket.

        For each cube, report: "on table" if you can see it on the table, "in bucket" if you can see it in the bucket or cannot see it at all, or "unsure" if you are not confident.

        Choose which cube the robot should pick up next:
        - Only choose a cube that is "on table"
        - Prefer the cube closest to the right side of the overhead image
        - If no cubes are on the table, set task to "ALL_DONE"

        Pick from these tasks only:
        {chr(10).join(f'- "{p}"' for p in allowed_prompts)}

        Return JSON only:
        {{"narration": "Red cube: on table. Blue cube: in bucket. ...", "task": "pick up red cube and place in green bucket", "task_cube_color": "red"}}"""

    #     #prompt V0
    #     prompt = f"""You are a high-level planner for a bimanual robot arm.
    # The overall goal is to pick up all cubes from the table and place them in the green bucket.
    # The scene contains five cubes (red, blue, yellow, brown, pink) and one green bucket.
    # {history}

    # Analyze the image and do three things:

    # 1. CUBES: For each of the five cubes, state whether it is on the tabletop, being held by the robot, or in the green bucket.
    # IMPORTANT: Only report a cube as "on table" if you can clearly see it on the table surface.
    # If a cube is not visible anywhere in the image, report it as "in bucket".

    # 2. TASK: Choose which cube the robot should pick up next.
    # Rules:
    # - Only choose a cube that is clearly visible on the table.
    # - Prefer the cube closest to the right side of the image (nearest to normalized coordinates x=1000, y=500), as this is where the robot's primary arm is.
    # - If the robot is currently holding a cube or moving toward one, keep the same cube as the task. Do NOT switch mid-action.
    # - If the current cube was just successfully placed in the bucket, choose a different cube that is still on the table.
    # - If the robot has failed to pick up the same cube {self.max_attempts_before_switch} times in a row, choose a different cube from the table. The robot can come back to the failed cube later.
    # - If no cubes are visible on the table, respond with "ALL_DONE".

    # 3. DETECT: Return bounding boxes for all visible cubes and the green bucket.

    # You must choose the task from EXACTLY one of these prompts:
    # {chr(10).join(f'- "{p}"' for p in allowed_prompts)}
    # Or respond with "ALL_DONE" if all cubes are in the bucket.

    # Respond in this exact JSON format and nothing else:
    #     {{
    #     "narration": "Red cube: on table. Blue cube: in bucket. ...",
    #     "robot_at_bucket": false,
    #     "task": "pick up red cube and place in green bucket",
    #     "task_cube_color": "red",
    #     "objects": [
    #         {{"box": [y1, x1, y2, x2], "label": "red cube"}},
    #         {{"box": [y1, x1, y2, x2], "label": "green bucket"}}
    #     ]
    #     }}
    #     The box coordinates are in [y1, x1, y2, x2] format normalized to 0-1000.
    #     Only include objects in the "objects" list that are visible in the image.
    #     Set "robot_at_bucket" to true if either robot arm is currently over or very near the green bucket, false otherwise."""

        # Respond in this exact JSON format and nothing else:
        # {{
        # "narration": "Red cube: on table. Blue cube: in bucket. ...",
        # "task": "pick up red cube and place in green bucket",
        # "task_cube_color": "red",
        # "objects": [
        #     {{"box": [y1, x1, y2, x2], "label": "red cube"}},
        #     {{"box": [y1, x1, y2, x2], "label": "green bucket"}}
        # ]
        # }}
        # The box coordinates are in [y1, x1, y2, x2] format normalized to 0-1000.
        # Only include objects in the "objects" list that are visible in the image."""

        #contents = [image_part, prompt]
        contents = [*image_parts, prompt]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_budget=128) #0
            )
        )

        import json
        try:
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(text)

            narration = result.get("narration", "")
            objects = result.get("objects", [])
            task = result.get("task", "")
            task_cube = result.get("task_cube_color", "")

            #temp for debug
            #logger.info(f"GR:{narration}")
            #logger.info(f"GR:{task}")
            #logger.info(f"GR:{task_cube}")

            #robot_at_bucket = result.get("robot_at_bucket", False)
            #self._update_attempt_tracking(narration, robot_at_bucket)

            # Handle ALL_DONE
            if task == "ALL_DONE":
                logger.info("Gemini: All cubes placed! Task complete.")
                return None, narration, objects

            #if task_cube != self.current_task_cube:
            #    logger.info(f"Gemini: Switching to {task_cube} cube")
            self.current_task_cube = task_cube

            # Track cube switches
            # if task_cube != self.current_task_cube:
            #     if self.current_task_cube:
            #         # Check if previous cube succeeded
            #         cube_name = f"{self.current_task_cube} cube"
            #         if f"{cube_name}: in bucket" in narration.lower():
            #             self.completed_cubes.add(self.current_task_cube)
            #             logger.info(f"Gemini: {self.current_task_cube} cube completed!")
            #     self.current_task_cube = task_cube
            #     self.current_task_attempts = 1
            #     logger.info(f"Gemini: Switching to {task_cube} cube")
            # else:
            #     self.current_task_attempts += 1

            # # Track attempts and cube switches
            # if task_cube != self.current_task_cube:
            #     # Switching to a new cube
            #     if self.current_task_cube and self.current_task_attempts < self.max_attempts_before_switch:
            #         # Gemini wants to switch too early — override and keep current
            #         if self.current_task_cube + " cube" not in narration.lower().replace("in bucket", ""):
            #             # Current cube is gone from table, it's in bucket — success!
            #             self.completed_cubes.add(self.current_task_cube)
            #             logger.info(f"Gemini: {self.current_task_cube} cube completed!")
            #         else:
            #             # Still on table but Gemini wants to switch — keep going
            #             task = f"pick up {self.current_task_cube} cube and place in green bucket"
            #             task_cube = self.current_task_cube
            #             #self.current_task_attempts += 1
            #             logger.info(f"Gemini: Keeping {self.current_task_cube} cube (attempt {self.current_task_attempts})")
            #     else:
            #         # Valid switch (max attempts reached or first task)
            #         self.current_task_cube = task_cube
            #         self.current_task_attempts = 1
            #         logger.info(f"Gemini: Switching to {task_cube} cube")
            #else:
            #    # Same cube, increment attempts
            #    self.current_task_attempts += 1

            # Validate task is in allowed list
            if task not in allowed_prompts:
                best = max(allowed_prompts, key=lambda p: len(set(p.split()) & set(task.split())))
                task = best

            return task, narration, objects

        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            # Fall back to current task
            if self.current_task_cube:
                return f"pick up {self.current_task_cube} cube and place in green bucket", "", []
            return None, "", []

    # def _update_attempt_tracking(self, narration: str, robot_at_bucket: bool):
    #     """Count attempts by detecting robot visits to bucket area."""
    #     if not self.current_task_cube:
    #         return
        
    #     cube_name = f"{self.current_task_cube} cube"
    #     narration_lower = narration.lower()
        
    #     #Success: cube now in bucket
    #     if f"{cube_name}: in bucket" in narration_lower:
    #         logger.info(f"Success! {self.current_task_cube} cube placed in bucket")
    #         self.completed_cubes.add(self.current_task_cube)
    #         self.current_task_cube = None
    #         self.current_task_attempts = 0
    #         self.was_at_bucket = False
    #         return
    #     # Success detected by narration — just reset bucket tracking
    #     # Actual completion is handled by plan_next_task's switch logic
    #     # if f"{cube_name}: in bucket" in narration_lower:
    #     #     logger.info(f"Bucket tracking: {self.current_task_cube} cube reported in bucket")
    #     #     self.was_at_bucket = False
    #     #     return

    #     # Track bucket visits
    #     if robot_at_bucket and not self.was_at_bucket:
    #         self.was_at_bucket = True
    #     elif not robot_at_bucket and self.was_at_bucket:
    #         self.was_at_bucket = False
    #         if f"{cube_name}: on table" in narration_lower:
    #             self.current_task_attempts += 1
    #             logger.info(f"Failed attempt {self.current_task_attempts} for {self.current_task_cube} cube")

    #     # Correct false completions: if a "completed" cube is back on table, un-complete it
    #     for color in list(self.completed_cubes):
    #         if f"{color} cube: on table" in narration_lower:
    #             self.completed_cubes.discard(color)
    #             logger.info(f"Correction: {color} cube back on table, removing from completed")

    def draw_annotations(self, image_rgb, objects: list):
        """Draw bounding boxes and labels on image. Returns BGR image for cv2 display."""
        import numpy as np
        img = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # Color map for labels
        color_map = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "pink": (203, 192, 255),
            "yellow": (0, 255, 255),
            "brown": (42, 42, 165),
            "green": (0, 255, 0),
        }

        for obj in objects:
            box = obj.get("box", [])
            label = obj.get("label", "")
            if len(box) != 4:
                continue

            # Convert normalized 0-1000 coords to pixel coords
            y1 = int(box[0] * h / 1000)
            x1 = int(box[1] * w / 1000)
            y2 = int(box[2] * h / 1000)
            x2 = int(box[3] * w / 1000)

            # Pick color based on label
            color = (0, 255, 0)  # default green
            for color_name, bgr in color_map.items():
                if color_name in label.lower():
                    color = bgr
                    break

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img
    
    def get_images_from_observation(self, observation_dict: dict, camera_features: list) -> dict:
        """Extract RGB images from robot observation dict (already RGB)."""
        images_rgb = {}
        for cam in camera_features:
            if cam == 'observation.images.cam_high':
                images_rgb[cam] = observation_dict[cam].numpy()
        return images_rgb
    
#######################################################################################

        # prompt = f"""You are observing a bimanual robot arm performing tasks on a table from an overhead camera.
        # The robot is currently attempting: "{current_task}"

        # Your PRIMARY job is to determine if this task has succeeded.
        # Think step by step:
        # Step 1: What color cube does the task mention? 
        # Step 2: In your narration, what is the status of that cube?
        # Step 3: If the status is "in bucket", then success is true. Otherwise false.

        # Also describe the scene:
        # For each of the five cubes (red, blue, yellow, brown, pink), state whether it is on the table, being held, or in the bucket.
        # Only report a cube as "on table" if you can clearly see it on the table surface.
        # If a cube is not visible anywhere, report it as "in bucket".

        # Also return bounding boxes for all visible cubes and the green bucket.

        # Respond in this exact JSON format and nothing else:
        # {{
        # "success": false,
        # "narration": "Red cube: on table. Blue cube: in bucket. Yellow cube: being held. Brown cube: on table. Pink cube: in bucket.",
        # "objects": [
        #     {{"box": [y1, x1, y2, x2], "label": "red cube"}},
        #     {{"box": [y1, x1, y2, x2], "label": "green bucket"}}
        # ]
        # }}
        # The box coordinates are in [y1, x1, y2, x2] format normalized to 0-1000.
        # Only include objects in the "objects" list that are visible in the image."""

        # prompt = f"""You are observing a bimanual robot arm performing tasks on a table from an overhead camera.
        # The robot is currently attempting: "{current_task}"

        # The scene contains five cubes (red, blue, yellow, brown, pink) and one green bucket.

        # Do two things:

        # 1. NARRATE: For each of the five cubes, state whether it is on the tabletop, being held by the robot, or in the green bucket.
        # IMPORTANT: Only report a cube as "on table" if you can clearly see it on the table surface.
        # If a cube is not visible anywhere in the image, report it as "in bucket" since it is likely inside and occluded.
        # Do NOT guess that a cube is on the table if you cannot see it.

        # 2. DETECT: Return a bounding box for every cube and the green bucket that you can see in the image.

        # Respond in this exact JSON format and nothing else:
        # {{
        # "narration": "Red cube: on table. Blue cube: in bucket. Yellow cube: being held. Brown cube: on table. Pink cube: in bucket.",
        # "success": false,
        # "objects": [
        #     {{"box": [y1, x1, y2, x2], "label": "red cube"}},
        #     {{"box": [y1, x1, y2, x2], "label": "green bucket"}}
        # ]
        # }}
        # The box coordinates are in [y1, x1, y2, x2] format normalized to 0-1000.
        # Only include objects in the "objects" list that are visible in the image.
        # Set "success" to true if the target cube from the current task is reported as "in bucket" in your narration above, false otherwise. For example, if the task is "pick up blue cube and place in green bucket" and your narration says "Blue cube: in bucket", then success is true."""
        
        #Set "success" to true if the current task has been completed (the target cube is now in the bucket), false otherwise."""

        # Respond in this exact JSON format and nothing else:
        # {{
        # "narration": "Red cube: on table. Blue cube: in bucket. Yellow cube: being held. Brown cube: on table. Pink cube: in bucket.",
        # "objects": [
        #     {{"box": [y1, x1, y2, x2], "label": "red cube"}},
        #     {{"box": [y1, x1, y2, x2], "label": "green bucket"}}
        # ]
        # }}
        # The box coordinates are in [y1, x1, y2, x2] format normalized to 0-1000.
        # Only include objects in the "objects" list that are visible in the image."""
