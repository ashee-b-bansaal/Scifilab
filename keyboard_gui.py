import logging
from typing import Callable
import threading
import queue
import dearpygui.dearpygui as dpg


class KeyboardGUIInput():
    def __init__(self,
                 new_keyboard_input_handler_list: list[Callable],
                 logger: logging.Logger
                 ) -> None:

        self.logger: logging.Logger = logger
        dpg.create_context()
        with dpg.font_registry():
            # first argument ids the path to the .ttf or .otf file
            self.default_font = dpg.add_font("NotoSerifCJKjp-Medium.otf", 30)
        dpg.create_viewport(title="KeyboardGUIInput",
                            width=600,
                            height=800,
                            resizable=False)
        self.msg_queue: queue.Queue = queue.Queue()
        self._need_keyboard_input_lock: threading.Lock = threading.Lock()
        self._need_keyboard_input_bool = False


        self._new_keyboard_input_handler_list: list[Callable] = new_keyboard_input_handler_list
        def _new_keyboard_input_handler(sender, app_data, user_data):
            dpg.mutex()
            dpg.set_item_label("input_text", "input not needed")
            print(sender, app_data, user_data)
            for handler in self._new_keyboard_input_handler_list:
                handler(app_data)
                
            dpg.set_value("input_text","" )
            with self._need_keyboard_input_lock:
                self._need_keyboard_input_bool = False

        self.new_keyboard_input_handler = _new_keyboard_input_handler


    def new_message_handler(self, msg: str):
        self.msg_queue.put_nowait(msg)

    def need_keyboard_input(self):
        with self._need_keyboard_input_lock:
            print("BUHHHHHH")
            self._need_keyboard_input_bool = True

        
    def show_ui(self):

        with dpg.window(
                tag="keyboard_window",
                label="Keyboard Input",
                width=600,
                height=800,
                on_close = dpg.stop_dearpygui):
            with dpg.child_window(
                    tag="log_screen",
                    label="Messages",
                    height = 600
                    ):
                pass
            with dpg.child_window(
                    tag = "input",
                    label = "Interpreter Input",
            ):
                dpg.add_input_text(
                    tag = "input_text",
                    label="input not needed",
                    on_enter=True,
                    callback=self.new_keyboard_input_handler,
                )
            dpg.bind_font(self.default_font)
                
    def exit(self):
        pass

    def run(self):
        dpg.setup_dearpygui()
        self.show_ui()
        dpg.show_viewport(minimized=True)

        while dpg.is_dearpygui_running():
            if self.msg_queue.qsize() != 0:
                text = self.msg_queue.get_nowait()
                color = (255, 255, 255)
                if "DHH" in text:
                    color = (255, 0, 0)
                elif "ChatGPT" in text:
                    color = (0, 255, 0)
                elif "Hearing" in text:
                    color = (255, 0, 255)
                dpg.add_text(text, parent="log_screen", tracked=True, wrap=550, color=color)
            with self._need_keyboard_input_lock:
                if self._need_keyboard_input_bool and dpg.get_item_label("input_text") != "need input":
                    dpg.set_item_label("input_text","need input")
                    print("just set input_text label to need input")
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

if __name__ == "__main__":
    import time
    def print_callback(text):
        print(text)
        
    logger = logging.getLogger(__name__)
    a = KeyboardGUIInput(
        new_keyboard_input_handler_list=[print_callback],
        logger=logger)
    gui_t = threading.Thread(target=a.run, daemon=True)
    gui_t.start()
    for i in range(100):
        a.new_message_handler("hello")
        a.new_message_handler("yo")
    a.need_keyboard_input()
    time.sleep(10)
    
    

