import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd

if __name__ == "__main__":

    # data_file = 'data/DHM 2/Fri Aug  2021DRI_WF_ECG1.csv'
    # df = pd.read_csv(data_file, header=None, names=['Dependent', 'Timestamp', 'Overflow',])
    #
    # print(df.head())
    # # Plotting the data
    # plt.plot(df['Timestamp'], df['Dependent'])
    # plt.xlabel('Time')
    # plt.ylabel('Dependent Value')
    # plt.title('Dependent Value vs. Time')
    # plt.xticks(rotation=45)
    # plt.show()

    # Read the data from the file
    filename = 'data/DHM 2/Andrei/Fri Aug  2021DRI_WF_ECG1.csv'
    data = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            data.append([float(values[1]), float(values[2])])

    # Separate the columns
    column_four = [row[0] for row in data[:1000]]  # Include only the first 1000 values
    column_five = [row[1] for row in data[:100]]  # Include only the first 1000 values

    # Generate the x-axis values
    x_axis = list(range(1000))  # Use only the first 1000 values

    # Plot the columns
    plt.plot(x_axis, column_four, color='red', label='Column 1')
    #plt.plot(x_axis, column_five, color='blue', label='Column 2')

    # Set labels and title
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('ECG of Columns 1 first 5000')

    # Show the legend
    plt.legend()

    # Display the plot
    plt.show()
