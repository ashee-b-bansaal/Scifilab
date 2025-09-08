import numpy as np
from pprint import pprint
import argparse
import datetime

def convert_log_time_to_sec(log_timestamp: str):
    return datetime.datetime.strptime(log_timestamp, "%Y_%m_%d_%H_%M_%S_%f").timestamp()


def parse_phone_log(log_path):
    with open(log_path, 'r') as f:
        lines = [line.rstrip() for line in f]
    events_list = []
    for i in range(len(lines)):
        timestamp_str, event = lines[i][:23],lines[i][24:] 
        timestamp = convert_log_time_to_sec(timestamp_str)
        events_list.append((timestamp, timestamp_str, event))
    return events_list

def index_string_startwith(target_list, prefix_string, stop_word):
    # print(target_list)
    for i, target_str in enumerate(target_list):
        if target_str == stop_word: return -1
        if target_str.startswith(prefix_string):
            return i
    return -1

def delay_between_2_events(event_list: list[tuple[float,str,str]], first_event: str, second_event: str):
    total_delay = 0.0
    event_str_list = [event[-1] for event in event_list]
    # print(event_str_list)
    delay_pairs = []
    for i, (timestamp, timestamp_str, event) in enumerate(event_list):
        if event.startswith(first_event):
            # print(i)
            llm_options_event_index = index_string_startwith(event_str_list[i + 1:], second_event, first_event)
            # print(llm_options_event_index)
            if llm_options_event_index == -1:
                continue
            llm_options_event_index += (i + 1)
            # print(first_event, second_event)
            llm_options_event = event_list[llm_options_event_index]
            total_delay += (llm_options_event[0] - timestamp)
            delay_pairs.append((event_list[i], llm_options_event))
    return total_delay, delay_pairs

def dhh_speaking_events(event_list):
    dhh_spoken = []
    for i, (timestamp, timestamp_str, event) in enumerate(event_list):
        if event.startswith("DHH:"):
            dhh_spoken.append(event_list[i])
    return dhh_spoken

def hearing_speaking_events(event_list):
    hearing_list = []
    for i, (timestamp, timestamp_str, event) in enumerate(event_list):
        if event.startswith("Hearing:"):
            hearing_list.append(event_list[i])
    return hearing_list



def last_index_startswith(event_list, prefix: str):
    index = -1
    for i, (timestamp, timestamp_str, event) in enumerate(event_list):
        if event.startswith(prefix):
            index = i

    return index

def index_startswith(event_list, prefix: str):
    for i, (timestamp, timestamp_str, event) in enumerate(event_list):
        if event.startswith(prefix):
            return i
    return -1
    

def phone_typing_start_typing_to_end_tts(event_list):
    pairs = []
    tts_end_prefix = "TTS stopped speaking"
    dhh_prefix = "DHH:"
    for i, (timestamp, timestamp_str, event) in enumerate(event_list):
        if event.startswith(dhh_prefix):
            index_start_typing = last_index_startswith(event_list[:i], "DHH Start Typing")
            start_typing_event = event_list[index_start_typing]
            index_tts_end = index_startswith(event_list[i:], tts_end_prefix) + i
            tts_end_event = event_list[index_tts_end]
            pairs.append((start_typing_event, tts_end_event))
    return pairs




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--path",
                        help = "path to interaction log file")
    args = parser.parse_args()
    events = parse_phone_log(args.path)
    print(events)
    total_raw_time = events[-1][0] - events[0][0]
    print("Total raw time", total_raw_time)

    delay_from_voice_recording_request_to_voice_rec, voice_recording_request_to_voice_rec_pairs = (delay_between_2_events(events, "Voice Recognition Requested", "Hearing:"))
    # print(voice_recording_request_to_voice_rec_pairs)
    print("Total delay from voice recording to recognition", delay_from_voice_recording_request_to_voice_rec)
    print("Mean delay from voice recording to recognition",
          np.mean([voice_recognition_event[0] - voice_recording_end_event[0] for  voice_recording_end_event,voice_recognition_event in voice_recording_request_to_voice_rec_pairs]))
    print("Std delay from voice recording to recognition",
          np.std([voice_recognition_event[0] - voice_recording_end_event[0] for  voice_recording_end_event,voice_recognition_event in voice_recording_request_to_voice_rec_pairs]))

    print("time without delay", total_raw_time - delay_from_voice_recording_request_to_voice_rec)

    dhh_speaking_events_list = dhh_speaking_events(events)
    hearing_speaking_events_list = hearing_speaking_events(events)
    print("number of hearing speaking", len(hearing_speaking_events_list))
    print("number of dhh speaking", len(dhh_speaking_events_list))


    num_words_hearing_speaking = [len(phrase[-1][len("Hearing:"):].split()) for phrase in hearing_speaking_events_list]
    print("Mean number of words from hearing speaking",
          np.mean(num_words_hearing_speaking))
    print("Std number of words from hearings speaking",
          np.std(num_words_hearing_speaking))

    num_words_dhh_typing = [len(phrase[-1][5:].split()) for phrase in dhh_speaking_events_list]
    print("Mean number of words from dhh typing",
          np.mean(num_words_dhh_typing))
    print("Std number of words from dhh typing",
          np.std(num_words_dhh_typing))
    print("Max number of words from dhh typing",
          max(num_words_dhh_typing))
    print("Min number of words from dhh typing",
          min(num_words_dhh_typing))

    
    total_typing_time, typing_start_end = delay_between_2_events(events, "DHH Start Typing", "DHH:")
    typing_time_dhh = [end[0] - start[0] for start, end in typing_start_end]
    print(typing_start_end)
    print("total typing time", total_typing_time)
    print("mean typing time", np.mean(typing_time_dhh))
    print("std typing time", np.std(typing_time_dhh))




    typing_tts_pair = phone_typing_start_typing_to_end_tts(events)
    typing_and_tts_time = [end[0] - start[0] for start, end in typing_tts_pair]
    print(typing_and_tts_time)
    # for start, end in typing_tts_pair:
        # start_timestamp, start_timestamp_str, start_event = start
        # end_timestamp, end_timestamp_str, end_event = end
        # print("======START EVENT======")
        # print(start_timestamp_str, start_event)
        # print(end_timestamp_str, end_event)
        # print(end_timestamp - start_timestamp)
        # print("======END EVENT======")

    total_time = events[-1][0] - events[0][0]
    mean_typing_and_tts_time = np.mean(typing_and_tts_time)
    std_typing_and_tts_time = np.std(typing_and_tts_time)
    print("total time", events[-1][0] - events[0][0])
    print("mean typing and tts time", mean_typing_and_tts_time)
    print("std typing and tts time", std_typing_and_tts_time)
    print("number of interaction", len(typing_and_tts_time))
    print(total_time, mean_typing_and_tts_time, std_typing_and_tts_time, len(typing_and_tts_time),sep=",")

    print("======================")



