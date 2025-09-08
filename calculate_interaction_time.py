import ast
import numpy as np
from pprint import pprint
import argparse
import datetime

def convert_log_time_to_sec(log_timestamp: str):
    return datetime.datetime.strptime(log_timestamp, "%Y-%m-%d %H:%M:%S,%f").timestamp()


def parse_interaction_log(log_path: str):
    with open(log_path, 'r') as f:
        lines = [line.rstrip() for line in f]
    events = []
    for i, line in enumerate(lines):
        timestamp_str = line[:23]
        try:
            timestamp = convert_log_time_to_sec(timestamp_str)
            events.append(line)
        except:
            events[-1] = events[-1] + line

    event_list = [(convert_log_time_to_sec(event[:23]),event[:23], event.split("-INFO-")[-1]) for event in events]
    event_list.sort(key=lambda x: x[0])
    return event_list


def index_string_startwith(target_list, prefix_string):
    # print(target_list)
    for i, target_str in enumerate(target_list):
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
            llm_options_event_index = index_string_startwith( event_str_list[i :], second_event)
            # print(llm_options_event_index)
            if llm_options_event_index == -1:
                continue
            llm_options_event_index += i
            
            llm_options_event = event_list[llm_options_event_index]
            total_delay += (llm_options_event[0] - timestamp)
            delay_pairs.append((event_list[i], llm_options_event))
    return total_delay, delay_pairs
            
def interpreter_translated_phrases(event_list):
    interpreter_phrase_list = []
    for i, (timestamp, timestamp_str, event) in enumerate(event_list):
        if event.startswith("DHH sign:"):
            interpreter_phrase_list.append(event.split("-INFO-")[-1][9:])
    return interpreter_phrase_list

def dhh_speaking_events_participant_5(event_list):
    dhh_spoken = []
    for i, (timestamp, timestamp_str, event) in enumerate(event_list):
        if event.startswith("Speaking Start:"):
            dhh_spoken.append(event_list[i])
    return dhh_spoken
    
def dhh_speaking_events(event_list):
    dhh_spoken = []
    for i, (timestamp, timestamp_str, event) in enumerate(event_list):
        if event.startswith("Speaking:"):
            dhh_spoken.append(event_list[i])
    return dhh_spoken

def hearing_speaking_events(event_list):
    hearing_list = []
    for i, (timestamp, timestamp_str, event) in enumerate(event_list):
        if event.startswith("Hearing:"):
            hearing_list.append(event_list[i])
    return hearing_list


def number_of_sign_selection_dhh(event_list):
    num = 0
    dhh_sign_events = []
    for timestamp, timestamp_str, event in event_list:
        if event.startswith("DHH sign"):
            num+= 1
            dhh_sign_events.append((timestamp, timestamp_str, event))
    return num, dhh_sign_events



def number_of_pardon_selection_dhh(event_list):
    num = 0
    dhh_sign_events = []
    for timestamp, timestamp_str, event in event_list:
        print(event)
        if event.startswith("DHH: 4. Can"):
            num+= 1
            dhh_sign_events.append((timestamp, timestamp_str, event))
    return num, dhh_sign_events


def number_of_next_options_selection_dhh(event_list):
    num = 0
    dhh_next_events = []
    for timestamp, timestamp_str, event in event_list:
        if event.startswith("DHH: NEXT OPTIONS"):
            num+= 1
            dhh_next_events.append((timestamp, timestamp_str, event))
    return num, dhh_next_events

def get_conversations(event_list, conversation_start_prefix="the tts spoke", conversation_end_prefix="the tts spoke"):
    conversation_start_index = 0
    conversation_end_index = 0
    conversation_list = []
    event_list_str = [event for timestamp, timestamp_str, event in event_list]
    if index_string_startwith(event_list_str, conversation_end_prefix) == -1: return conversation_list
    conversation_end_index = index_string_startwith(event_list_str, conversation_end_prefix)
    conversation_list.append(event_list[conversation_start_index:conversation_end_index + 1])
    while True:
        # conversation_after_start = event_list_str[conversation_start_index + 1: ]
        # if index_string_startwith(conversation_after_start, conversation_end_prefix) == -1: break
        event_list_str_after_last = event_list_str[conversation_end_index + 1:]
        next_start_index = index_string_startwith(event_list_str_after_last, conversation_start_prefix)
        if next_start_index == -1: break
        conversation_start_index = conversation_end_index + 1 + next_start_index
        next_end_index = index_string_startwith(event_list_str_after_last, conversation_end_prefix)
        if next_end_index == -1:
            conversation_list.append(event_list[conversation_start_index: ])
            break
        conversation_end_index = conversation_end_index + 1 + next_end_index
        conversation_list.append(event_list[conversation_start_index: conversation_end_index + 1])
        # next_start_index = index_string_startwith(conversation_after_start,conversation_start_prefix)
        # if next_start_index == -1:break 
        # conversation_start_index += next_start_index + 1
    return conversation_list

def flatten(xss):
    return [x for xs in xss for x in xs]

def conversations_with_and_without_sign(conversation_list, sign_prefix="DHH: SIGN"):
    sign_list = []
    no_sign_list = []
    for conversation in conversation_list:
        if any([event.startswith(sign_prefix) for timestamp,timestamp_str,event in conversation ]):
            sign_list.append(conversation)
        else:
            no_sign_list.append(conversation)
    return sign_list, no_sign_list

def hearing_dhh_sides_conversation(conversation):
    dhh_side_start_index = 0
    for i, (timestamp, timestamp_str, event) in enumerate(conversation):
        if event.startswith("ChatGPT:"):
            dhh_side_start_index = i
            break
    assert len(conversation[:dhh_side_start_index]) + len(conversation[dhh_side_start_index:]) == len(conversation)
    return conversation[:dhh_side_start_index], conversation[dhh_side_start_index:]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--path",
                        help = "path to interaction log file")
    parser.add_argument("-pn",
                        "--participant-number",
                        help = "path to interaction log file", type=int, required=True)

    parser.add_argument("-tn",
                        "--task-number",
                        help =
 "path to interaction log file", type=int, required=True)
    args = parser.parse_args()
    events =  (parse_interaction_log(args.path))
    start_event_index = 0
    if (args.participant_number == 1 or args.participant_number == 2) and args.task_number == 1:
        events_str = [event for timestamp, timestamp_str, event in events]
        start_event_index = index_string_startwith(events_str, "Phase 0")
    elif args.participant_number == 1 and args.task_number == 2:
        start_event_index = 35
    elif args.participant_number ==2 and args.task_number ==2:
        events_str = [event for timestamp, timestamp_str, event in events]
        start_event_index = index_string_startwith(events_str, "Hearing: Hello, hello.")
    elif args.participant_number ==3 and args.task_number ==2:
        start_event_index = 2
    elif args.participant_number ==4 and args.task_number ==1:
        start_event_index = 2
    elif args.participant_number ==4 and args.task_number ==2:
        start_event_index = 8
    elif args.participant_number == 5:
        events_str = [event for timestamp, timestamp_str, event in events]
        start_event_index = index_string_startwith(events_str, "Voice Recording Started")
    elif args.participant_number > 5:
        events_str = [event for timestamp, timestamp_str, event in events]
        start_event_index = index_string_startwith(events_str, "Voice Recording Start")

    events = events[start_event_index: ]
    for i, (timestamp, timestamp_str, event) in enumerate(events):
        if "Yeah, it's fine. What do you like" in event:
            start_event_index = i
            break
        # if  "Hearing: Yeah, sure. What do you" in event:
            # start_event_index = i
            # break
    print("start event index", start_event_index)
    # for event in events:
        # print("____", event, "____")
    delay_from_voice_rec_to_gpt, voice_rec_to_gpt_pairs = (delay_between_2_events(events, "Hearing:", "ChatGPT:"))
    print("Total delay from voice recognition to chatgpt options", delay_from_voice_rec_to_gpt)
    print("Mean delay from voice recognition to chatgpt options",
          np.mean([gpt_option_event[0] - voice_rec_event[0] for  voice_rec_event,gpt_option_event in voice_rec_to_gpt_pairs]))
    print("Std delay from voice recognition to chatgpt options",
          np.std([gpt_option_event[0] - voice_rec_event[0] for  voice_rec_event,gpt_option_event in voice_rec_to_gpt_pairs]))
    # print(voice_rec_to_gpt_pairs)
    # for voice_rec_event,gpt_option_event in voice_rec_to_gpt_pairs:
        # print(voice_rec_event, gpt_option_event)
    
    delay_from_interperter_to_gpt, interperter_gpt_pairs = (delay_between_2_events(events, "DHH sign:", "ChatGPT:"))
    print("Total delay from interpreter pressing enter to chatgpt options", delay_from_interperter_to_gpt)
    print("Mean delay from interpreter pressing enter to chatgpt options",
          np.mean([gpt_option_event[0] - interpreter_event[0] for  interpreter_event,gpt_option_event in interperter_gpt_pairs]))
    print("Std delay from interpreter pressing enter to chatgpt options",
          np.std([gpt_option_event[0] - interpreter_event[0] for  interpreter_event,gpt_option_event in interperter_gpt_pairs]))
    # for interpreter_event,gpt_option_event in interperter_gpt_pairs:
        # print(interpreter_event, gpt_option_event)


    delay_from_dhh_next_to_gpt, dhh_next_gpt_pairs = (delay_between_2_events(events, "DHH: NEXT", "ChatGPT:"))
    print("Total delay from dhh choosing next to chatgpt options", delay_from_dhh_next_to_gpt)
    print("Mean delay from dhh choosing next to chatgpt options",
          np.mean([gpt_option_event[0] - dhh_next_event[0] for  dhh_next_event,gpt_option_event in dhh_next_gpt_pairs]))
    print("Std delay from dhh choosing next to chatgpt options",
          np.std([gpt_option_event[0] - dhh_next_event[0] for  dhh_next_event,gpt_option_event in dhh_next_gpt_pairs]))
    # for dhh_next_event,gpt_option_event in dhh_next_gpt_pairs:
        # print(dhh_next_event, gpt_option_event)
    

    delay_from_voice_recording_stopping_to_voice_rec, voice_recording_stopping_to_voice_rec_pairs = (delay_between_2_events(events, "Voice Recognition Requested", "Hearing:"))
    print("Total delay from voice recording to recognition", delay_from_voice_recording_stopping_to_voice_rec)
    print("Mean delay from voice recording to recognition",
          np.mean([voice_recognition_event[0] - voice_recording_end_event[0] for  voice_recording_end_event,voice_recognition_event in voice_recording_stopping_to_voice_rec_pairs]))
    print("Std delay from voice recording to recognition",
          np.std([voice_recognition_event[0] - voice_recording_end_event[0] for  voice_recording_end_event,voice_recognition_event in voice_recording_stopping_to_voice_rec_pairs]))
    
    # for voice_recording_end_event,voice_recognition_event in voice_recording_stopping_to_voice_rec_pairs:
        # print(voice_recording_end_event,voice_recognition_event)


    interpreter_translated_phrases_list = (interpreter_translated_phrases(events))
    print(interpreter_translated_phrases_list)
    interpreter_words_len = [len(w.split()) for w in interpreter_translated_phrases_list]
    

    if args.participant_number >= 5:
        dhh_speaking_phrase_list = [event[-1][19:] for event in dhh_speaking_events_participant_5(events)]
    else:
        dhh_speaking_phrase_list = [event[-1][13:] for event in dhh_speaking_events(events)]
    print("dhh_speaking_phrase_list", dhh_speaking_phrase_list)

    dhh_speaking_num_word_list = [len(phrase.split()) for phrase in dhh_speaking_phrase_list]
    
    hearing_speaking_phrase_list = [event[-1][9:] for event in hearing_speaking_events(events)]
    print(hearing_speaking_phrase_list)
    
    hearing_speaking_num_word_list = [len(phrase.split()) for phrase in hearing_speaking_phrase_list]
    
    total_raw_time = events[-1][0] - events[0][0]
    total_delay = delay_from_dhh_next_to_gpt + delay_from_interperter_to_gpt + delay_from_voice_rec_to_gpt + delay_from_voice_recording_stopping_to_voice_rec 
    print("TOTAL TIME RAW IS: ", total_raw_time)
    print("TOTAL DELAY IS: ", total_delay)
    print("total time without delay", total_raw_time - total_delay)
    print("total # of saying (hearing)", len(hearing_speaking_phrase_list))
    print("total # of saying (DHH)", len(dhh_speaking_phrase_list))
    print("average on # of words per a sentence (hearing)", np.mean(hearing_speaking_num_word_list))
    print("hearing speaking num word list", hearing_speaking_num_word_list)
    print("standard deviation on # of words per a sentence (hearing)", np.std(hearing_speaking_num_word_list))
    print("average on # of sign words per a sentence (HDD)", np.mean(interpreter_words_len))
    print("standard deviation on # of sign words per a sentence (HDD)", np.std(interpreter_words_len))
    print("max on # of sign words per a sentence (HDD)", max(interpreter_words_len))
    print("min on # of sign words per a sentence (HDD)", min(interpreter_words_len))


    print("#of sign selection (DHH)", number_of_sign_selection_dhh(events))
    if args.participant_number <= 5:
        print("#of next selection (DHH)", number_of_next_options_selection_dhh(events))
    else:
        print("#of can you repeat that selection (DHH)", number_of_pardon_selection_dhh(events))


    conversation_start_prefix, conversation_end_prefix = None, None
    if args.participant_number == 5:
        conversation_start_prefix, conversation_end_prefix = "Voice Recording Started", "Speaking Finished"
    elif args.participant_number > 5:
        conversation_start_prefix, conversation_end_prefix = "Voice Recording Start", "Speaking Finished"
    elif args.participant_number < 5:
        conversation_start_prefix, conversation_end_prefix = "Hearing:", "the tts spoke"


    conversation_list = get_conversations(events, conversation_start_prefix, conversation_end_prefix)
    print(sum([len(conversation) for conversation in conversation_list]))
    print(len(events))
    # assert sum([len(conversation) for conversation in conversation_list]) == len(events)
    if not conversation_list[-1][-1][2].startswith(conversation_end_prefix):
        conversation_list = conversation_list[:-1]
        
    if args.participant_number < 5:
        # for these participants, we didn't log the time from voice recording stop to recognition so we need to add the end of the last conversation to the beginning of the next one, since the end is when tss done speaking which is usually when voice rec picks up.
        for i in range(1, len(conversation_list)):
            conversation_list[i].insert(0, conversation_list[i -1][-1])


    # for convo in conversation_list:
    #     print("______")
    #     for event in convo:print(event)
    #     print("______")
        

            
    convo_sign, convo_no_sign = conversations_with_and_without_sign(conversation_list)


    print("Conversations with sign")
    for convo in convo_sign:
        print("------")
        for event in (convo): print(event)
        print("------")
    print("Conversations without sign")    
    for convo in convo_no_sign:
        print("------")
        for event in (convo): print(event)
        print("------")

    ################ CONVERSATIONS WITH SIGNING
    convo_sign_hearing, convo_sign_dhh = zip(*[hearing_dhh_sides_conversation(convo) for convo in convo_sign])
    assert len(convo_sign_hearing)== len(convo_sign_dhh)
    print("Number of conversations with sign", len(convo_sign))

    convo_sign_duration = [convo[-1][0] - convo[0][0] for convo in convo_sign]
    convo_sign_dhh_duration = [convo[-1][0] - convo[0][0] for convo in convo_sign_dhh]
    convo_sign_hearing_duration = [convo[-1][0] - convo[0][0] for convo in convo_sign_hearing]

    # for convo in convo_sign_dhh:
        # print("--------")
        # for event in convo: print(event)
        # print("--------")
    
    print("Mean time of conversation with sign",np.mean(convo_sign_duration))
    # print("Std time of conversation with sign", np.std(convo_sign_duration))
    print("Mean time of conversation with sign only DHH side",np.mean(convo_sign_dhh_duration))


    all_gpt_options_sign_dhh = []
    for convo in convo_sign_dhh:
        for timestamp, timestamp_str, event in convo:
            if "ChatGPT:" in event:
                options = ast.literal_eval(event[len("ChatGPT:"):])
                all_gpt_options_sign_dhh.extend([(option[3:-2]) for  option in options])

    all_gpt_options_sign_dhh_len = [len(option) for option in all_gpt_options_sign_dhh]
    print("Mean length of gpt options for conversations with signing",np.mean(all_gpt_options_sign_dhh_len))
    
    
    ###############

    ############## CONVERSATIOSN WITH ONLY SELECTION
    selected_option_1_cnt = 0
    selected_option_2_cnt = 0
    selected_option_3_cnt = 0    
    all_gpt_options_no_sign_dhh = []
    all_gpt_options_no_sign_dhh_len = []
    if len(convo_no_sign) > 0:
        convo_no_sign_hearing, convo_no_sign_dhh = zip(*[hearing_dhh_sides_conversation(convo) for convo in convo_no_sign])

        assert len(convo_no_sign_hearing)== len(convo_no_sign_dhh)          

        print("Number of conversations with only selection", len(convo_no_sign))

        convo_no_sign_duration = [convo[-1][0] - convo[0][0] for convo in convo_no_sign]
        convo_no_sign_dhh_duration = [convo[-1][0] - convo[0][0] for convo in convo_no_sign_dhh]
        convo_no_sign_hearing_duration = [convo[-1][0] - convo[0][0] for convo in convo_no_sign_hearing]
        print("no sign hearing", convo_no_sign_hearing_duration)
        # print(convo_no_sign_duration)
        print("Mean time of conversation with only selection", np.mean(convo_no_sign_duration))
        # print("Std time of conversation with only selection", np.std(convo_no_sign_duration))
        print("Mean time of conversation with only selection only DHH side",np.mean(convo_no_sign_dhh_duration))
        print("Mean time of conversation with only selection hearing side",np.mean(convo_no_sign_hearing_duration))
        # print(convo_no_sign_hearing_duration)

        # print(conversation_list[-1])
        # convs = flatten(conversation_list)
        # for i in range(len(events)):
            # print(convs[i])
            # print(events[i])

        cnt = 0
        for convo in convo_no_sign_dhh:
            for timestamp, timestamp_str, event in convo:
                if "ChatGPT:" in event:
                    options = ast.literal_eval(event[len("ChatGPT:"):])
                    all_gpt_options_no_sign_dhh.extend([(option[3:-2]) for  option in options])
        # print(all_gpt_options_no_sign_dhh)

        for convo in convo_no_sign_dhh:
            for timestamp, timestamp_str, event in convo:
                if "DHH: 1" in event:
                    selected_option_1_cnt += 1
                elif  "DHH: 2" in event:
                    selected_option_2_cnt += 1
                elif "DHH: 3" in event:
                    selected_option_3_cnt += 1

                    
        all_gpt_options_no_sign_dhh_len = [len(option) for option in all_gpt_options_no_sign_dhh]
        mean_gpt_option_len_no_sign = np.mean(all_gpt_options_no_sign_dhh_len)
        number_of_gpt_option_len_no_sign = len(all_gpt_options_no_sign_dhh_len)
        print("Mean length of gpt options for conversations with only selection",mean_gpt_option_len_no_sign)


    all_gpt_options = all_gpt_options_sign_dhh + all_gpt_options_no_sign_dhh
    print(all_gpt_options_sign_dhh)
    print(all_gpt_options_no_sign_dhh)
    mean_length_all_gpt_options = np.mean(all_gpt_options_sign_dhh_len + all_gpt_options_no_sign_dhh_len)
    print("mean length of gpt options", mean_length_all_gpt_options)


    print("Number of times DHH selected option 1 in conversations where they only used selection:", selected_option_1_cnt)
    print("Number of times DHH selected option 2 in conversations where they only used selection:", selected_option_2_cnt)
    print("Number of times DHH selected option 3 in conversations where they only used selection:", selected_option_3_cnt)


    if args.participant_number > 5:
        reading_time_no_sign = []
        total_length_options_no_sign = []
        for convo in convo_no_sign_dhh:
            events_in_convo = [event for timestamp, timestamp_str, event in convo]
            for i, (timestamp, timstamp_str, event) in enumerate(convo):
                if "ChatGPT:" in event:
                    selection_event_index = index_string_startwith(events_in_convo[i + 1: ], "DHH selection timestamp") + i + 1
                    selection_timestamp = float(convo[selection_event_index][-1][len("DHH selection timestamp:"):])
                    # print(selection_timestamp)
                    reading_time_no_sign.append(selection_timestamp - timestamp)
                    options = ast.literal_eval(event[len("ChatGPT:"):])
                    total_length_options_no_sign.append(sum([len(option[3:-2]) for  option in options]))
        print(list(zip(reading_time_no_sign, total_length_options_no_sign)))
        


        
