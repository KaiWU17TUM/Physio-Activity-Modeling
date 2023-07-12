import re

import matplotlib.pyplot as plt
import pandas as pd
import os

def test(filename):
    # Read in the CSV file
    data = pd.read_csv('data/DHM-action-vitals/activity_hr_data.csv')

    # Plot columns using matplotlib
    fig, axs = plt.subplots(4, 3, figsize=(20, 20))
    axs[0, 0].plot(data["hr"])
    axs[0, 0].set_title("HR")
    axs[0, 1].plot(data["rr"])
    axs[0, 1].set_title("RR")
    axs[0, 2].plot(data["spo2"])
    axs[0, 2].set_title("SpO2")
    axs[1, 0].plot(data["nibp_systolic"])
    axs[1, 0].set_title("NIBP Systolic")
    axs[1, 1].plot(data["nibp_diastolic"])
    axs[1, 1].set_title("NIBP Diastolic")
    axs[1, 2].plot(data["nibp_mean"])
    axs[1, 2].set_title("NIBP Mean")
    axs[2, 0].plot(data["hr_lie"])
    axs[2, 0].set_title("HR Lie")
    axs[2, 1].plot(data["hr_sit"])
    axs[2, 1].set_title("HR Sit")
    axs[2, 2].plot(data["hr_stand"])
    axs[2, 2].set_title("HR Stand")
    axs[3, 0].plot(data["hr_lie_std"])
    axs[3, 0].set_title("HR Lie Std")
    axs[3, 1].plot(data["hr_sit_std"])
    axs[3, 1].set_title("HR Sit Std")
    axs[3, 2].plot(data["hr_stand_std"])
    axs[3, 2].set_title("HR Stand Std")
    fig.suptitle(filename)
    plt.show()


if __name__ == "__main__":
    for file in os.listdir('data/DHM-action-vitals/'):
        if file.endswith('.csv'):
            test(file)
        print(file)
    #test('data/DHM-action-vitals/activity_hr_data.csv')
