import os
from imgui_bundle import imgui

from splatviz_utils.gui_utils import imgui_utils
from imgui_bundle._imgui_bundle import portable_file_dialogs
from widgets.widget import Widget


class LoadWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Load")
        self.plys: list[str] = [""]
        self.use_splitscreen = False
        self.highlight_border = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            plys_to_remove = []
            for i, ply in enumerate(self.plys):
                if imgui_utils.button(f"Browse {i + 1}", width=viz.button_w):
                    files_from_dialog = portable_file_dialogs.open_file("Select .ply or .yml", os.getcwd(), filters=[]).result()#
                    if len(files_from_dialog) > 0:
                        self.plys[i] = files_from_dialog[0]
                imgui.same_line()
                if len(self.plys) > 1:
                    if imgui_utils.button(f"Remove {i + 1}", width=viz.button_w):
                        plys_to_remove.append(i)
                    imgui.same_line()
                imgui.text(f"Scene {i + 1}: ")

            for i in plys_to_remove[::-1]:
                self.plys.pop(i)
            if imgui_utils.button("Add Scene", width=viz.button_w):
                files_from_dialog = portable_file_dialogs.open_file("Select .ply or .yml", os.getcwd(), filters=[]).result()  #
                if len(files_from_dialog) > 0:
                    self.plys.append(files_from_dialog[0])

            if len(self.plys) > 1:
                use_splitscreen, self.use_splitscreen = imgui.checkbox("Splitscreen", self.use_splitscreen)
                highlight_border, self.highlight_border = imgui.checkbox("Highlight Border", self.highlight_border)

        viz.args.highlight_border = self.highlight_border
        viz.args.use_splitscreen = self.use_splitscreen
        viz.args.ply_file_paths = self.plys
        viz.args.current_ply_names = [
            ply.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_") for ply in self.plys
        ]
