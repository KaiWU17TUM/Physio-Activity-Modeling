import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def try_out_tsne():
    # Read data from CSV file
    data = pd.read_csv("data/DHM 2/Andrei/transformed_data.csv", header=None, names=["ecg", "l", "time_stempt", "time", "subject"])

    # Separate features and target (if applicable)
    # Replace 'target_column' with the actual name of your target column, or remove this section if you don't have a target
    features = data.drop( ["l", "time_stempt", "time", "subject"], axis=1)
    target = data['l']

    # Perform t-SNE
    tsne = TSNE(n_components=1)  # You can adjust the number of components (dimensions) as needed
    embeddings = tsne.fit_transform(features)

    # Create a scatter plot of the embeddings
    print("plot stated")
    plt.scatter(embeddings[:, 0], embeddings[:, 0], c=target, cmap='viridis')
    plt.colorbar()
    plt.title("t-sne of the ecg directly")
    plt.show()

if __name__ == "__main__":
    try_out_tsne()
