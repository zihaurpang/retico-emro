import os
import retico_core
from retico_core import abstract, UpdateMessage, UpdateType
from retico_core.text import TextIU

class ActionTextIU(TextIU):
    @staticmethod
    def type():
        return TextIU.type()
    def __repr__(self):
        return f"{self.type()} - ({self.creator.name()}): {self.get_text()}"

  class ActionExecutionModule(abstract.AbstractModule):
    @staticmethod
    def name():
        return "Action Execution Module"
    @staticmethod
    def description():
        return "Execute GRED‐generated actions on the robot."
    @staticmethod
    def input_ius():
        return [ActionTextIU]

    def __init__(self, robot, **kwargs):
        super().__init__(**kwargs)
        self.robot = robot

    def execute(self, actions_str):
        # make sure images are loaded before any change_image call
        self.robot.populate_images()
        # strip out any leftover model markers
        # actions_str is like "set_volume_30 say_text_Hi! move_head_0_-10_0_80 ..."
        for token in actions_str.split():
            # normalize say_text_<msg> → say_<msg> # The original trained model used "say_text_" prefix
            if token.startswith("say_text_"):
                token = "say_" + token[len("say_text_"):]

            # handle display_face_ tokens → change_image('X.png', timeout)
            if token.startswith("display_face_"):
                # only keep last two parts: image base + timeout, add “.png”
                rest = token[len("display_face_"):]
                parts = rest.split("_")
                image_base = parts[-2]
                timeout    = parts[-1]
                token = f"change_image_{image_base}.png_{timeout}"
                # print(f"Converting display_face token: {token}")

            # handle drive_track_ tokens → drive_track(left, right)
            if token.startswith("drive_track_"):
                # split into left and right parts
                rest = token[len("drive_track_"):]
                parts = rest.split("_")
                if len(parts) == 2:
                    left, right = parts
                elif len(parts) == 3:
                    left, right, _ = parts
                token = f"drive_track_{left}_{right}"

            parts = token.split("_")
            # check if first two segments form a valid robot method
            two_part = "_".join(parts[:2])
            if len(parts) > 1 and hasattr(self.robot, two_part):
                name = two_part
                arg_parts = parts[2:]
            else:
                name = parts[0]
                arg_parts = parts[1:]

            # convert args: try int(), else keep as string
            args = []
            for a in arg_parts:
                try:
                    args.append(int(a))
                except ValueError:
                    args.append(a)

            # dispatch to robot
            getattr(self.robot, name)(*args)

    def process_update(self, update_message):
        # for every ADD update carry the payload into execute()
        for iu, typ in update_message:
            if typ == UpdateType.ADD:
                print(f"Executing action IU: {iu}")
                self.execute(iu.payload)
