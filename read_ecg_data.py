import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Read the data from the file
    filename = "data/MHEALTHDATASET/mHealth_subject1.log"
    data = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split('\t')
            data.append([float(values[3]), float(values[4])])

    # Separate the columns
    column_four = [row[0] for row in data[:100]]  # Include only the first 1000 values
    column_five = [row[1] for row in data[:100]]  # Include only the first 1000 values

    # Generate the x-axis values
    x_axis = list(range(100))  # Use only the first 1000 values

    # Plot the columns
    plt.plot(x_axis, column_four, color='red', label='Column 4')
    plt.plot(x_axis, column_five, color='blue', label='Column 5')

    # Set labels and title
    plt.xlabel('Line')
    plt.ylabel('Value')
    plt.title('Plot of Columns 4 and 5 first 100')

    # Show the legend
    plt.legend()

    # Display the plot
    plt.show()