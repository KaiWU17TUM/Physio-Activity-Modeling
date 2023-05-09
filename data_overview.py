import matplotlib.pyplot as plt
import pandas as pd

def test():
    print("hello World")
    # Read in the CSV file
    data = pd.read_csv('data/DHM-action-vitals/activity_hr_data.csv')

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(data['index'], data['clipID'])

    # Add axis labels and a title
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_title('Plot Title')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    test()