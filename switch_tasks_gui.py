import threading
from typing import Callable
import dearpygui.dearpygui as dpg


class SwitchTaskGUI():
    def __init__(self, on_button_click: Callable):
        self.on_button_click = on_button_click
        dpg.create_context()
        dpg.create_viewport(title="Switch Task GUI",
                            width=600,
                            height=800,
                            resizable=False)
    def show_ui(self):
        with dpg.window(tag="switch_task_window",
                        label="Switch task window",
                        width=600,
                        height=800,
                        on_close=dpg.stop_dearpygui):
            with dpg.child_window(tag = "button_screen",
                                  label="button screen",
                                  ):
                dpg.add_button(
                    label="button 0",
                    callback=self.on_button_click)

    def run(self):
        dpg.setup_dearpygui()
        self.show_ui()
        dpg.show_viewport(minimized=True)
        dpg.start_dearpygui()
        dpg.destroy_context()


if __name__ == "__main__":
    a = SwitchTaskGUI(lambda: None)
    a_t = threading.Thread(target=a.run)
    a_t.start()
    
