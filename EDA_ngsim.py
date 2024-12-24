import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data

class EDA:
    def __init__(self, data):
        self.data = data

    def summary_statistics(self):
        print("Summary Statistics:")
        print(self.data.describe())

    def plot_histogram(self, column):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.show()

    def plot_correlation_matrix(self):
        plt.figure(figsize=(12, 8))
        numerical_data = self.data.select_dtypes(include=np.number)
        #sns.heatmap(self.data.corr(), annot=True, cmap='Vehicle_ID', vmin=-1, vmax=1)
        sns.heatmap(numerical_data.corr(), annot=True, cmap='viridis', vmin=-1, vmax=1) 
        plt.title('Correlation Matrix')
        plt.show()

# Usage
from google.colab import files
uploaded = files.upload()
data=pd.read_csv('ngsim.csv')
eda = EDA(data)
eda
eda.summary_statistics()
eda.plot_histogram('Vehicle_ID') 
eda.plot_correlation_matrix()
