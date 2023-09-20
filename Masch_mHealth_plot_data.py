import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # data = pd.read_csv('data/MHEALTHDATASET/activity_hr_data.csv')
    # THat works great for DHM dataset
    # data['time'] = pd.to_datetime(data['time'])
    # data.set_index('time', inplace=True)
    #
    # # Plot the data
    # headers = data.columns[5:]  # Select all headers starting from index 5
    # for header in headers:
    #     plt.plot(data[header], label=header)
    #
    # plt.xlabel('Time')
    # plt.ylabel('Measurement')
    # plt.legend()
    # plt.show()

    # Plot data of MHealthdataset
    #for x in range(120):
    # # Filter the data for subject 1
    # subject_1_data = data[data['subject'] == x]
    #
    # # Extract the necessary columns for visualization
    # time = subject_1_data['time']
    # hr = subject_1_data['hr']
    # hr_lie = subject_1_data['hr_lie']
    # hr_sit = subject_1_data['hr_sit']
    # hr_stand = subject_1_data['hr_stand']
    #
    # # Plot the data
    # plt.plot(time, hr, label='Heart Rate')
    # plt.plot(time, hr_lie, label='Heart Rate (Lying)')
    # plt.plot(time, hr_sit, label='Heart Rate (Sitting)')
    # plt.plot(time, hr_stand, label='Heart Rate (Standing)')
    #
    # # Add labels and legend
    # plt.xlabel('Time')
    # plt.ylabel('Heart Rate')
    # plt.legend()
    #
    # # Display the plot
    # plt.show()

    #DATA actigraph
    data = pd.read_csv('data/MMASH/DataPaper/user_1/Actigraph.csv')
    # Extract the necessary columns for visualization
    axis1 = data['Axis1']
    axis2 = data['Axis2']
    axis3 = data['Axis3']
    steps = data['Steps']

    # Plot the accelerometer data
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(axis1, label='Axis1')
    plt.plot(axis2, label='Axis2')
    plt.plot(axis3, label='Axis3')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title('Accelerometer Data')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(steps, label='Steps')
    plt.xlabel('Time')
    plt.ylabel('Step Count')
    plt.title('Step Count')
    plt.legend()

    plt.tight_layout()
    plt.show()

    #Data activity
    # Read the CSV data into a DataFrame
    data = pd.read_csv('data/MMASH/DataPaper/user_1/Activity.csv')

    # Extract the necessary columns for visualization
    activities = data['Activity']
    start_times = data['Start']
    end_times = data['End']
    days = data['Day']

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(start_times, activities, 'o-', label='Activity')
    plt.xlabel('Time')
    plt.ylabel('Activity')
    plt.title('Activity Timeline')
    plt.legend()

    # Annotate the end times
    for i in range(len(data)):
        plt.annotate(end_times[i], (end_times[i], activities[i]))

    # Add vertical lines to represent the start and end times
    for i in range(len(data)):
        plt.axvline(x=start_times[i], color='gray', linestyle='--')
        plt.axvline(x=end_times[i], color='gray', linestyle='--')

    plt.tight_layout()
    plt.show()

    # Read the CSV data into a DataFrame
    data = pd.read_csv('data/MMASH/DataPaper/user_1/RR.csv')

    # Extract the necessary columns for visualization
    ibi = data['ibi_s']
    time = data['time']

    # Convert time to datetime format
    time = pd.to_datetime(time)

    # Plot the IBI data
    plt.figure(figsize=(12, 6))
    plt.plot(time, ibi, marker='o', linestyle='-', label='IBI')
    plt.xlabel('Time')
    plt.ylabel('IBI (s)')
    plt.title('IBI Variation Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()