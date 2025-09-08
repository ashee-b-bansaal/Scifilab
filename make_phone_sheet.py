from datetime import datetime
from pprint import pprint
import argparse
from calculate_phone_time import parse_phone_log, dhh_speaking_events, hearing_speaking_events
import subprocess

def print_events(events):
    for timestamp, timestamp_str, event in events:
        print(timestamp_str, event)

def print_interactions(interactions):
    for i, interaction in enumerate(interactions):
        # print(f'==== interaction {i} ======')
        print_events(interaction)


def get_index_lower_bound_timestamp(events, lower_bound_timestamp):
    return next((i for i, (timestamp, timestamp_str, event) in enumerate(events) if timestamp >= lower_bound_timestamp), -1)

def get_index_upper_bound_timestamp(events, upper_bound_timestamp):
    return next((len(events) - i - 1 for i, (timestamp, timestamp_str, event) in enumerate(events[::-1]) if timestamp <= upper_bound_timestamp), -1)

def get_interactions(events):
    """
    get the interactions by looking at hearing and dhh events
    """
    dhh_events = dhh_speaking_events(events)
    hearing_events = hearing_speaking_events(events)
    interactions = []
    for i, h_event in enumerate(hearing_events):
        h_timestamp, h_timestamp_str, h_e = h_event
        current_interaction = [h_event]
        if i == len(hearing_events) - 1:
            current_interaction.extend(dhh_events[get_index_lower_bound_timestamp(dhh_events, h_timestamp):])
            interactions.append(current_interaction)
            dhh_events = []
            break
        dhh_lower_bound_index = get_index_lower_bound_timestamp(dhh_events, h_timestamp)
        dhh_upper_bound_index = get_index_upper_bound_timestamp(dhh_events, hearing_events[i+1][0])
        current_interaction.extend(dhh_events[dhh_lower_bound_index : dhh_upper_bound_index + 1])
        interactions.append(current_interaction)
        dhh_events = dhh_events[dhh_upper_bound_index + 1: ]
    return interactions

def get_participant_num(path: str) -> int:
    return int(path.split("_")[-4])

def get_task_num(path: str) -> int:
    return int(path.split("_")[-2])

def timestamp_to_str(timestamp: float):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")

def get_hearing_speaking_time(events, hearing_event):
    hearing_speaking_start_event_list = [event for event in events if event[-1].startswith("Voice Recording Start")]
    hearing_speaking_end_event_list = [event for event in events if event[-1].startswith("Voice Recording Stop")]
    hearing_start_index = get_index_upper_bound_timestamp(hearing_speaking_start_event_list, hearing_event[0])
    hearing_end_index = get_index_upper_bound_timestamp(hearing_speaking_end_event_list, hearing_event[0])
    start_speaking_time = hearing_speaking_start_event_list[hearing_start_index][0]
    end_speaking_time = hearing_speaking_end_event_list[hearing_end_index][0]
    return end_speaking_time - start_speaking_time

def get_sub_interaction_start_end_event(events, dhh_event):
    """
    sub_interaction is from the start of dhh typing to when tts finishes
    """
    dhh_start_typing_event_list = [event for event in events if event[-1].startswith("DHH Start Typing")]
    tts_fin_event_list = [event for event in events if event[-1].startswith("TTS stopped speaking")]
    sub_start_index = get_index_upper_bound_timestamp(dhh_start_typing_event_list, dhh_event[0])
    start_event = dhh_start_typing_event_list[sub_start_index]
    start_typing_time = start_event[0]
    end_typing_time = dhh_event[0]
    try:
        end_tts_event = [e for e in tts_fin_event_list if (e[-1].endswith(dhh_event[-1][len("DHH: "):]) and e[0] > dhh_event[0])][0]
        end_tts_time = end_tts_event[0]
    except:
        # print([e for e in tts_fin_event_list if e[-1].endswith(dhh_event[-1][len("DHH: "):])])
        # print(tts_fin_event_list)
        # print(dhh_event)
        # raise RuntimeError("brij")
        end_tts_time = dhh_event[0] + get_tts_phrase_length(dhh_event[-1][len("DHH: "):])
    return start_typing_time, end_typing_time, end_tts_time

def csv_lines_from_interaction(events, interaction):
    """
    an interaction is a speaking event followed by one or more dhh event
    return csv lines with the following header:
    sub_interaction_number,hearing_spoken_phrase,hearing_num_words,dhh_text,dhh_text_num_words,typing_start_timestamp,typing_end_timestamp,typing_time,tts_time
    """
    csv_lines = []
    hearing_spoken_phrase = interaction[0][-1][len("Hearing: "):]
    hearing_num_words = len(hearing_spoken_phrase.split())
    hearing_speaking_time = get_hearing_speaking_time(events, interaction[0])    
    for sub_interaction_number, dhh_event  in enumerate(interaction[1:]):
        dhh_text=dhh_event[-1][len("DHH: "):]
        dhh_text_num_words = len(dhh_text.split())
        start_typing_time, end_typing_time, end_tts_time = get_sub_interaction_start_end_event(events, dhh_event)
        typing_time = end_typing_time - start_typing_time
        csv_lines.append([
            sub_interaction_number + 1,
            hearing_spoken_phrase,
            hearing_speaking_time,
            hearing_num_words, 
            dhh_text,
            dhh_text_num_words,
            timestamp_to_str(start_typing_time),
            timestamp_to_str(end_typing_time),
            typing_time,
            end_tts_time - end_typing_time
        ])
    # print("---interaction---")
    # print_events(interaction)
    # print("---interaction---")    
    return csv_lines



def events_to_csv(participant_num, task_num, events):
    interactions = get_interactions(events)
    csv_lines = []
    interaction_num = 1
    for i, interaction in enumerate(interactions):
        interaction_lines = csv_lines_from_interaction(events, interaction)
        if interaction_lines == []: continue

        csv_lines.extend([[participant_num, task_num, interaction_num] + line for line in interaction_lines])
        interaction_num += 1
    return csv_lines
        
    
    

def get_tts_phrase_length(s: str):
    subprocess.run(["python", "tts.py", "--text", s, "--output-file", "tmp.wav"], capture_output=True)
    import librosa
    return librosa.get_duration(path='tmp.wav')
    

# an interaction essentially consists of hearing person speaking then dhh response. A subinteraction is each response that a dhh person gives. An interaction then is: hearing person -> subinteractions.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--path",
                        help = "path to interaction log file")
    parser.add_argument("-o",
                        "--output",
                        help = "path to output file")    
    # /home/nam/scifilab/realtime-llm/phone_data_user_study_04_07_25/participant_6_task_2_phone.txt
    args = parser.parse_args()

    p_num = get_participant_num(args.path)
    t_num = (get_task_num(args.path))

    events = parse_phone_log(args.path)
    # print_events(events)
    dhh_events = dhh_speaking_events(events)
    hearing_events = hearing_speaking_events(events)
    # print("======== DHH events ======")
    # print_events(dhh_events)
    # print("======== DHH events ======")
    # print("======== Hearing events ======")
    # print_events(hearing_events)
    # print("======== Hearing events ======")

    # print("======== interactions ======")
    interactions = get_interactions(events)
    # print_interactions(interactions)
    assert sum([len(i) for i in interactions]) == len(dhh_events) + len(hearing_events)
    for interaction in interactions:
        print("start ==================================\n")
        print(interaction)
        # print(csv_lines_from_interaction(events, interaction))
        print("end ==================================\n")
    # print(get_tts_phrase_length("I love my food"))
    
    csv_lines = ["participant_num,task_num,interaction_number,sub_interaction_number,hearing_spoken_phrase,hearing_speaking_time,hearing_num_words,dhh_text,dhh_text_num_words,typing_start_timestamp,typing_end_timestamp,typing_time,tts_time".split(",")]  +  events_to_csv(p_num, t_num, events)
    for line in csv_lines:
        line = ["\"" + str(i) + "\"" for i in line]
        print(",".join(line))
    with open(args.output, 'w') as f:
        for line in csv_lines:
            f.write(",".join(["\"" + str(i) + "\"" for i in line]) + "\n")
    print("hearing_speaking_time")
    for i, title in enumerate(csv_lines[0]):
        if title == "hearing_speaking_time":
            for line in csv_lines[1:]:
                print(line[i])
    print("typing_start_timestamp,typing_end_timestamp,typing_time,tts_time")
    for i, title in enumerate(csv_lines[0]):
        if title == "typing_start_timestamp":
            for line in csv_lines[1:]:
                print(",".join([str(j) for j in line[i:]]))
    


    
