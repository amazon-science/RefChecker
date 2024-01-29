import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import pickle
import joblib
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

def generate_mask(n):
    num_segments = 6
    segment_length = n // num_segments

    mask = np.tile(np.concatenate([np.ones(segment_length, dtype=bool),
                                   np.zeros(segment_length, dtype=bool)]),
                   n // (2 * segment_length))

    return mask


class KNN:
    # zth: two methods supported: knn and nca (with parameters, and need to specify the dimension, default to 2)
    def __init__(self, k, variance="knn", dim=2):
        self.k = k
        knn = KNeighborsClassifier(n_neighbors=k)
        if variance == "nca":
            nca = NeighborhoodComponentsAnalysis(n_components=dim, random_state=42)
            self.pipe = Pipeline([('nca', nca), ('knn', knn)])
        elif variance == "knn":
            self.pipe = Pipeline([('knn', knn)])

    def train(self, X_train, y_train):
        self.pipe.fit(X_train, y_train)

    def predict(self, input_vector):
        predictions = self.pipe.predict(input_vector)

        return predictions

# zth: when using svm, only need to specify the kernel function
class SVM:
    def __init__(self, kernel='linear'):
        self.kernel = kernel
        self.scaler = StandardScaler()
        self.model = svm.SVC(kernel=self.kernel)

    def train(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, input_vector):
        input_vector = self.scaler.transform(input_vector)
        predictions = self.model.predict(input_vector)
        return predictions

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            saved_data = pickle.load(file)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

class PyTorchClassifier:
    def __init__(self, input_size, hidden_size, output_size, lr=0.0001, alpha=0.0001, epochs=300, batch_size=32, device="cuda"):
        self.device = torch.device(device)

        self.model = NeuralNetwork(input_size, hidden_size, output_size).to(self.device)
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=alpha)

    def train(self, X_train, y_train):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def predict(self, input_vector):
        input_tensor = torch.FloatTensor(input_vector).to(self.device)

        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            return predicted.cpu().numpy()

class AttentionModule(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(AttentionModule, self).__init__()
        self.query_vector = nn.Parameter(torch.rand(hidden_dim, requires_grad=True))
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)

        self.linear_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_matrix):
        key = self.Wk(input_matrix)
        value = self.Wv(input_matrix)
        attention_weights = torch.matmul(self.query_vector, key.transpose(1, 2)) / torch.sqrt(
            torch.tensor(input_matrix.size(-1)))
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(value * attention_weights.unsqueeze(2), dim=1)
        output = self.linear_layer(context_vector)
        return output

class AttentionClassifier:
    def __init__(self, hidden_dim, num_classes, lr=0.0001, alpha=0.0001, epochs=300, batch_size=32,
                 early_stop_patience=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AttentionModule(hidden_dim, num_classes).to(self.device)
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        self.early_stop_counter = 0
        self.best_valid_loss = np.inf

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=alpha)

    def train(self, input_matrix, labels):
        input_tensor = torch.FloatTensor(input_matrix).to(self.device)
        labels_tensor = torch.LongTensor(labels).to(self.device)
        mask = generate_mask(input_tensor.size(0))
        train_input_tensor = input_tensor[mask]
        valid_input_tensor = input_tensor[~mask]
        train_labels_tensor = labels_tensor[mask]
        valid_labels_tensor = labels_tensor[~mask]

        train_dataset = TensorDataset(train_input_tensor, train_labels_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        valid_dataset = TensorDataset(valid_input_tensor, valid_labels_tensor)

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            for inputs, target in train_dataloader:
                inputs, target = inputs.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()

            # Validation
            valid_loss = self.evaluate(valid_dataset)

            print(
                f'Epoch [{epoch + 1}/{self.epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {valid_loss:.4f}')

            # Check for early stopping
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_patience:
                    print(f'Early stopping after {epoch + 1} epochs without improvement on validation loss.')
                    break

    def predict(self, input_matrix):
        input_tensor = torch.FloatTensor(input_matrix).to(self.device)

        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            return predicted.cpu().numpy()

    def evaluate(self, dataset):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, target in DataLoader(dataset, batch_size=self.batch_size):
                inputs, target = inputs.to(self.device), target.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, target)
                total_loss += loss.item() * len(target)

        return total_loss / len(dataset)