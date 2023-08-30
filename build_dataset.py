import csv

def build_csv(lable_file, data_file, name):

    list_of_labels = []

    with open(lable_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            row_list = row[0].split(";")
            label = row_list[2]
            time_start = row_list[0]
            time_end = row_list[1]
            list_of_labels.append((time_end.replace(".", ""), label))
    #print(list_of_labels)

    new_csv_list = []
    index_in_labels = 0
    with open(data_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            time_general = row[0]
            ecg_signal = row[1]
            time_stamp = row[2]
            if index_in_labels >= len(list_of_labels):
                print("Break")
                break
            if time_stamp < list_of_labels[index_in_labels][0]:
                new_csv_list.append((ecg_signal, list_of_labels[index_in_labels][1], time_stamp, time_general, name))
            else:
                index_in_labels += 1

    # Output file name
    output_file = "data/DHM 2/Andrei/transformed_data3.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(new_csv_list)


    print("Data processing complete. Output saved to", output_file)

if __name__ == "__main__":
    print("Hello")
    build_csv("data/DHM 2/Andrei/Andrei Activities3new.csv", "data/DHM 2/Andrei/Fri Aug  2021DRI_WF_ECG1.csv", "Andrei")