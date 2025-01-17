import speech_recognition as sr

class InputHandler():
    def record_voice_input(self):
        """
        Records voice input from the microphone and returns the transcribed text.
        """
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            #print("Listening... (Speak now)")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = recognizer.recognize_google(audio)
                #print(f"Voice input recognized: {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand the audio.")
                return ""
            except sr.RequestError as e:
                print(f"Error with the speech recognition service: {e}")
                return ""
            except Exception as e:
                print(f"Error during voice recording: {e}")
                return ""
    # NEEDS CORRECTION - THE INPUT SHOULD NOT BE IN TERMINAL 
    # def record_keyboard_input(self):
    #     """
    #     Records text input from the keyboard and returns it.
    #     """
    #     user_input = input("Enter your text: ")
    #     return user_input
