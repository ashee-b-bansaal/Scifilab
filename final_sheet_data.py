import numpy as np
import ast
from copy import deepcopy
import argparse
from calculate_interaction_time import convert_log_time_to_sec, parse_interaction_log, index_string_startwith, delay_between_2_events, flatten, get_conversations,conversations_with_and_without_sign, hearing_dhh_sides_conversation

def get_interactions(event_list, interaction_start_prefix, interaction_end_prefix):
    """
    defintion of interaction is 2 turns from hearing to dhh. Reused function from module but changed name
    """
    return get_conversations(event_list, interaction_start_prefix, interaction_end_prefix)

def get_interaction_with_and_without_sign_list(interaction_list, sign_prefix = "DHH: SIGN"):
    return conversations_with_and_without_sign(interaction_list, sign_prefix)


def print_interaction_list(interaction_list):
    for interaction in interaction_list:
        print("=======START INTERACTION=========")
        for timestamp, timestamp_str, event in interaction:
            print(timestamp, timestamp_str, event)
        print("=======END INTERACTION=========")

        
def calculate_stats(events):
    interaction_start_prefix, interaction_end_prefix = "Voice Recording Start", "Speaking Finished"

    # get all interactions, which comprises of 2 turns
    interaction_list = get_interactions(events, interaction_start_prefix, interaction_end_prefix)

    # assumption is that last interaction is most likely incomplete
    interaction_list = interaction_list[:-1]

    # print_interaction_list(interaction_list)

    
    interaction_with_sign_list, interaction_no_sign_list = get_interaction_with_and_without_sign_list(interaction_list)
    assert len(interaction_with_sign_list) +  len(interaction_no_sign_list) == len(interaction_list)
    hearing_turn_interaction_with_sign, dhh_turn_interaction_with_sign = zip(*[hearing_dhh_sides_conversation(interaction) for interaction in interaction_with_sign_list])
    dhh_turn_interaction_with_sign_duration = [dhh_turn[-1][0] - dhh_turn[0][0] for dhh_turn in dhh_turn_interaction_with_sign]
    
    hearing_turn_interaction_no_sign, dhh_turn_interaction_no_sign = zip(*[hearing_dhh_sides_conversation(interaction) for interaction in interaction_no_sign_list])
    dhh_turn_interaction_no_sign_duration = [dhh_turn[-1][0] - dhh_turn[0][0] for dhh_turn in dhh_turn_interaction_no_sign]


    total_time = interaction_list[-1][-1][0] - interaction_list[0][0][0]
    print("total time", interaction_list[-1][-1][0] - interaction_list[0][0][0])
    # print("start event", interaction_list[0][0])
    # print("end event", interaction_list[-1][-1])
    print("number of all interactions", len(interaction_list))
    print("number of interaction without sign/only select",len(interaction_no_sign_list))
    print("number of interaction with sign",len(interaction_with_sign_list))
    print("average length of time of ddh-turn in interactions without sign/only selection", np.mean(dhh_turn_interaction_no_sign_duration))
    print("std length of time of ddh-turn in interactions without sign/only selection", np.std(dhh_turn_interaction_no_sign_duration))
    print("average length of time of ddh-turn in interactions with sign", np.mean(dhh_turn_interaction_with_sign_duration))
    print("std length of time of ddh-turn in interactions with sign" ,np.std(dhh_turn_interaction_with_sign_duration))    






    
    print(total_time, len(interaction_list), len(interaction_no_sign_list), len(interaction_with_sign_list), np.mean(dhh_turn_interaction_no_sign_duration), np.std(dhh_turn_interaction_no_sign_duration),np.mean(dhh_turn_interaction_with_sign_duration),np.std(dhh_turn_interaction_with_sign_duration), sep=',')
    
    print("-----------")
    print("dhh-select-turn time", dhh_turn_interaction_no_sign_duration)
    print("dhh-sign-turn time", dhh_turn_interaction_with_sign_duration)
    print("-----------")
    return dhh_turn_interaction_no_sign_duration, dhh_turn_interaction_with_sign_duration

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ps",
                        "--path-social",
                        help = "path to interaction log file for social task 2")
    parser.add_argument("-po",
                        "--path-order",
                        help = "path to interaction log file ordering scenario task 1")
    parser.add_argument("-pn",
                        "--participant-number",
                        help = "path to interaction log file", type=int, required=True)

    args = parser.parse_args()

    events_social =  (parse_interaction_log(args.path_social))
    events_social_str = [event for timestamp, timestamp_str, event in events_social]
    social_start_event_index = index_string_startwith(events_social_str, "Voice Recording Start")
    
    events_social = events_social[social_start_event_index:]
    # print("first event social", events_social[0])

    events_order =  (parse_interaction_log(args.path_order))
    events_order_str = [event for timestamp, timestamp_str, event in events_order]
    order_start_event_index = index_string_startwith(events_order_str, "Voice Recording Start")

    events_order = events_order[order_start_event_index:]
    # print("first event order", events_order[0])
    print("========ORDER START========")
    calculate_stats(events_order)

    print("========ORDER END========")
    
    print("========SOCIAL START========")
    calculate_stats(events_social)

    print("========SOCIAL END========")

