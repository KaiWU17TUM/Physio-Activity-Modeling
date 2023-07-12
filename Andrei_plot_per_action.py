import csv
import json

import numpy as np
from matplotlib import pyplot as plt


def group_data(timestamp_file, ecg_data_file):

    #contains the timestamps
    timestamp = {}
    ecg_group_after_time = {}

    # Read egg data from the second CSV file
    # Store the timestamp data in a dict and run over them to group
    with open(timestamp_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            row_list = row[0].split(";")
            label = row_list[2]
            time_start = row_list[0]
            time_end = row_list[1]
            if label in timestamp.keys():
                timestamp[label] = timestamp[label] + [(time_start, time_end)]
            else:
                timestamp[label] = [(time_start, time_end)]

    #print(timestamp)

    ecg_signal = []
    #count = 00

    #effizienz des Codes wichtig ?
    with open(ecg_data_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
        #     if count > 10:
        #         break
            signal = row[1]
            timestamp_in_file = row[2]
            ecg_signal.append((signal, timestamp_in_file))
        #    count += 1


    #9 das sind die labes, kommen nicht sortiert, dadurch passt das nicht mir dem raus lesen, muss für den Key komplett durchlaufen.
    for key in timestamp.keys():
        #run through complete file every time for each label
        curr_pos_in_ecg_signal = 0
        #alle start stop eines labes durchlaufe
        for start, stop in timestamp[key]:
            curr_timestamp = ecg_signal[curr_pos_in_ecg_signal]
            #print(stop.replace(".", "")[:13] + ">=" + curr_timestamp[1] + ">=" + start.replace(".", "")[:13])
            #print(curr_pos_in_ecg_signal)
            #print(int(stop.replace(".", "")[:13]) >= int(curr_timestamp[1]) >= int(start.replace(".", "")[:13]))
            #Do vergelichst immer den ersten wert musst aber mit allen vergelichen.
            # lauf einfach für alle elemente durch optimier später

            for signal_pair in ecg_signal:
                if int(stop.replace(".", "")[:13]) >= int(signal_pair[1]) >= int(start.replace(".", "")[:13]):
                    if key not in ecg_group_after_time.keys():
                        print(key)
                        ecg_group_after_time[key] = [signal_pair[0]]
                    else:
                        ecg_group_after_time[key].append(signal_pair[0])

            # while int(stop.replace(".", "")[:13]) >= int(curr_timestamp[1]) >= int(start.replace(".", "")[:13]):
            #     #print(stop.replace(".", "")[:13] + ">=" + curr_timestamp[1] + ">=" + start.replace(".", "")[:13])
            #     if key not in ecg_group_after_time.keys():
            #         ecg_group_after_time[key] = [curr_timestamp[0]]
            #     else:
            #         ecg_group_after_time[key].append(curr_timestamp[0])
            #     #print(key)
            #     curr_pos_in_ecg_signal += 1
            #     curr_timestamp = ecg_signal[curr_pos_in_ecg_signal]

    return ecg_group_after_time

# def plot_dict(dcit_to_plot):
#     for key, value in dcit_to_plot.items():
#         plt.plot(value, label=key)
#
#     #funktiniert das ?
#     # Individuelle Y-Achsenbegrenzungen einstellen
#     plt.ylim(-2, 25)  # Y-Achsenbereich von -2 bis 25 festlegen
#
#     # Individuelle Legende der Y-Achse festlegen
#     plt.yticks(range(-2, 26, 5))  # Legendenwerte von -2 bis 25 in 5er-Schritten
#
#     # TODO was sind die grenzwerte in denen sich das bewegt ?
#
#     plt.xlabel('Time-Achse')
#     plt.ylabel('Values-Achse')
#     plt.legend()
#     plt.show()

def plot_dict(dict_to_plot):
    plt.figure()  # Create a new plot for each key-value pair
    count = 0
    for key, value in dict_to_plot.items():
        if count == 5:
            break
        count += 1
        #list(map(int, string_list))
        values_as_float = list(map(float, value))[:5000]
        #filtered_x = [val for val in values_as_float if val > 13]
        x = np.arange(len(values_as_float))
        updated_list = [0 if value < 13 else value for value in values_as_float]
        plt.plot(list(map(float, updated_list)), label=key)

    # Individuelle Y-Achsenbegrenzungen einstellen
    plt.ylim(-2, 40)  # Y-Achsenbereich von -2 bis 25 festlegen

    # Individuelle Legende der Y-Achse festlegen
    plt.yticks(range(-2, 40, 5))  # Legendenwerte von -2 bis 25 in 5er-Schritten

    plt.xlabel('X-Achse')
    plt.ylabel('Y-Achse')
    plt.title("Andrei ECG first 100 of class")
    plt.legend()
    plt.savefig(f'Andrei ECG first 5000.png')

    plt.show()

if __name__ == "__main__":
    #Andrei Activities1.csv
    #dict_values = group_data("data/DHM 2/Andrei/Andrei Activities2.csv", "data/DHM 2/Andrei/Fri Aug  2021DRI_WF_ECG1.csv")
    #with open("labeld_ecg.json", "w") as file:
    #    json.dump(dict_values, file)
    # Read dictionary from file
    with open("labeld_ecg.json", "r") as file:
        loaded_data = json.load(file)
    plot_dict(loaded_data)

