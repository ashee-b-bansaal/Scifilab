from typing import Callable
import threading
import queue
import dearpygui.dearpygui as dpg

#stop_dearpygui

class KeyboardGUIInput():
    def __init__(self, new_keyboard_input_handler: Callable) -> None:
        dpg.create_context()
        dpg.create_viewport(title="KeyboardGUIInput", width=600, height=800)
        self.msg_queue: queue.Queue = queue.Queue()
        self.new_keyboard_input_handler = new_keyboard_input_handler

    def new_message_handler(self, msg: str):
        self.msg_queue.put_nowait(msg)
    
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
                dpg.add_text("bruh")
                dpg.add_text("bruh_1")
            with dpg.child_window(tag = "input",
                               label = "Interpreter Input",
                               ):
                dpg.add_input_text(
                    tag = "input_text",
                    label="input text",
                    on_enter=True,
                    callback=self.new_keyboard_input_handler,
                )
                
    def exit(self):
        pass

    def run(self):
        dpg.setup_dearpygui()
        self.show_ui()
        dpg.show_viewport(minimized=True)

        while dpg.is_dearpygui_running():
            if self.msg_queue.qsize() != 0:
                text = self.msg_queue.get_nowait()
                dpg.add_text(text, parent="log_screen", tracked=True)
                
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

if __name__ == "__main__":
    def print_callback(sender, app_data, user_data):
        print(sender, app_data, user_data)
        dpg.mutex()
        dpg.set_value("input_text","" )
        
        
    a = KeyboardGUIInput(new_keyboard_input_handler=print_callback)
    for i in range(100):
        a.new_message_handler("hello")
        a.new_message_handler("yo")
    a.run()
    
    

