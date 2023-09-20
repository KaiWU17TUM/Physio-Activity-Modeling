import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    data = pd.read_csv('../data/DHM-action-vitals/activity_hr_data.csv')
    # #That works great for DHM dataset
    data['index'] = pd.to_datetime(data['index'])
    data.set_index('index', inplace=True)

    # Plot the data
    headers = data.columns[2:]  # Select all headers starting from index 5
    for header in headers:
        plt.plot(data[header], label=header)

    plt.xlabel('Time')
    plt.ylabel('Measurement')
    plt.legend()
    plt.show()
