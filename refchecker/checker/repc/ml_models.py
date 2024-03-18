import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
try:
    from sklearnex import patch_sklearn
    # zth: sklearn accelerate package, without which svm will execute very slowly
    patch_sklearn()
except:
    print("Warning: scikit-learn-intelex not installed, sklearn acceleration for the RepC checker is not enabled.")
    pass


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

    def pred_prob(self, input_tensor):
        probabilities = self.pipe.predict_proba(input_tensor)
        return torch.FloatTensor(probabilities)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump({'pipe': self.pipe}, file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            saved_data = pickle.load(file)
            self.pipe = saved_data['pipe']

# zth: when using svm, only need to specify the kernel function
class SVM:
    def __init__(self, kernel='linear'):
        self.kernel = kernel
        self.scaler = StandardScaler()
        self.model = svm.SVC(kernel=self.kernel, probability=True)

    def train(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, input_vector):
        input_vector = self.scaler.transform(input_vector)
        predictions = self.model.predict(input_vector)
        return predictions

    def pred_prob(self, input_tensor):
        input_tensor = self.scaler.transform(input_tensor)
        probabilities = self.model.predict_proba(input_tensor)
        probabilities = torch.FloatTensor(probabilities)
        probabilities = F.softmax(probabilities, dim=1)  # Applying softmax to convert decision function values to probabilities
        return probabilities
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
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

class PyTorchClassifier:
    def __init__(self, input_size=4096, hidden_size=1024, output_size=3, lr=0.0001, alpha=0.0001, epochs=300, batch_size=32, patience=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = NeuralNetwork(input_size, hidden_size, output_size).to(self.device)
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.current_patience = 0  # Initialize patience counter

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=alpha)

        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None

    def train(self, X_train, y_train, X_val, y_val):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Validation
            with torch.no_grad():
                self.model.eval()
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)
            print(
                f'Epoch [{epoch + 1}/{self.epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.current_patience = 0
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                self.best_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
            else:
                self.current_patience += 1
                if self.current_patience >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            self.model.train()  # Set the model back to training mode after early stopping
        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
            self.optimizer.load_state_dict(self.best_optimizer_state_dict)

        self.model.eval()
        val_outputs = self.model(X_val_tensor)
        val_loss = self.criterion(val_outputs, y_val_tensor)
        print(f"best val loss: {val_loss}")
    def predict(self, input_matrix, batch_size=128):
        input_tensor = torch.FloatTensor(input_matrix).to(self.device)

        with torch.no_grad():
            self.model.eval()
            all_predictions = []

            if batch_size is None:
                output = self.model(input_tensor)
                _, predicted = torch.max(output, 1)
                all_predictions.extend(predicted.cpu().numpy())
            else:
                for i in range(0, input_tensor.size(0), batch_size):
                    batch_input = input_tensor[i:i + batch_size]
                    output = self.model(batch_input)
                    _, predicted = torch.max(output, 1)
                    all_predictions.extend(predicted.cpu().numpy())

        return np.array(all_predictions)

    def pred_prob(self, input_tensor, batch_size=128):
        input_tensor = torch.FloatTensor(input_tensor).to(self.device)

        with torch.no_grad():
            self.model.eval()
            all_probabilities = []

            if batch_size is None:
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())
            else:
                for i in range(0, input_tensor.size(0), batch_size):
                    batch_input = input_tensor[i:i + batch_size]
                    output = self.model(batch_input)
                    probabilities = F.softmax(output, dim=1)
                    all_probabilities.extend(probabilities.cpu().numpy())

        return torch.FloatTensor(all_probabilities)
    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epochs,
            'loss': self.criterion,
            'patience': self.patience,
            'current_patience': self.current_patience
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs = checkpoint['epoch']
        self.criterion = checkpoint['loss']
        self.patience = checkpoint['patience']
        self.current_patience = checkpoint['current_patience']
class AttentionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(AttentionModule, self).__init__()
        self.query_vector = nn.Parameter(torch.rand(input_dim, requires_grad=True))
        self.Wk = nn.Linear(input_dim, input_dim, bias=False)
        self.Wv = nn.Linear(input_dim, input_dim, bias=False)

        self.linear_layer_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_layer_2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_matrix):
        key = self.Wk(input_matrix)
        value = self.Wv(input_matrix)
        attention_weights = torch.matmul(self.query_vector, key.transpose(1, 2)) / torch.sqrt(
            torch.tensor(input_matrix.size(-1)))
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(value * attention_weights.unsqueeze(2), dim=1)
        output = self.linear_layer_1(context_vector)
        output = self.linear_layer_2(output)
        return output

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

class AttentionClassifier:
    def __init__(self, input_dim=4096, hidden_dim=1024, num_classes=3, lr=0.0001, alpha=0.0001, epochs=300, batch_size=32,
                 patience=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AttentionModule(input_dim, hidden_dim, num_classes).to(self.device)
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.current_patience = 0  # Initialize patience counter

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=alpha)

        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None

    def train(self, X_train, y_train, X_val, y_val):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Validation
            with torch.no_grad():
                self.model.eval()
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)
            print(
                f'Epoch [{epoch + 1}/{self.epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.current_patience = 0
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                self.best_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
            else:
                self.current_patience += 1
                if self.current_patience >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            self.model.train()  # Set the model back to training mode after early stopping

        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
            self.optimizer.load_state_dict(self.best_optimizer_state_dict)

        self.model.eval()
        val_outputs = self.model(X_val_tensor)
        val_loss = self.criterion(val_outputs, y_val_tensor)
        print(f"best val loss: {val_loss}")

    def predict(self, input_matrix, batch_size=128):
        input_tensor = torch.FloatTensor(input_matrix).to(self.device)

        with torch.no_grad():
            self.model.eval()
            all_predictions = []

            if batch_size is None:
                output = self.model(input_tensor)
                _, predicted = torch.max(output, 1)
                all_predictions.extend(predicted.cpu().numpy())
            else:
                for i in range(0, input_tensor.size(0), batch_size):
                    batch_input = input_tensor[i:i + batch_size]
                    output = self.model(batch_input)
                    _, predicted = torch.max(output, 1)
                    all_predictions.extend(predicted.cpu().numpy())

        return np.array(all_predictions)

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

    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epochs,
            'loss': self.criterion,
            'patience': self.patience,
            'current_patience': self.current_patience
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs = checkpoint['epoch']
        self.criterion = checkpoint['loss']
        self.patience = checkpoint['patience']
        self.current_patience = checkpoint['current_patience']

class EnsembleModule(nn.Module):
    def __init__(self, input_size, output_size, expert_paths, expert_type='NN', classifier_type='mlp', skip_num=0, **kwargs):
        super(EnsembleModule, self).__init__()
        self.classifier_type = classifier_type
        self.skip_num = skip_num
        if classifier_type == 'mlp':
            self.gating_network = nn.Linear(input_size, output_size)
        elif classifier_type == 'svm':
            self.classifier = SVM(kernel="rbf")
        elif classifier_type == 'entropy':
            self.temperature = kwargs["temperature"]
            self.topk = kwargs["topk"]

        self.experts = []
        for path in expert_paths:
            if expert_type == 'knn':
                expert = KNN(k=kwargs["k"])
                expert.load(path)
            elif expert_type == 'svm':
                expert = SVM(kernel="rbf")
                expert.load(path)
            elif expert_type == 'nn':
                expert = PyTorchClassifier()
                expert.load(path)
            else:
                raise ValueError("Invalid expert_type. Supported types: 'KNN', 'SVM', 'NN'")
            self.experts.append(expert)
        self.experts = self.experts[skip_num:]

    def entropy(self, p):
        return torch.sum(-torch.where(p > 0, p * p.log2(), p.new([0.0])), dim=-1)
    # def forward(self, x, distribution=None):
    #     feature_selected = x[:, -1, :]
    #     output = self.gating_network_1(feature_selected)
    #     output = self.relu(output)
    #     output = self.gating_network_2(output)
    #     gate_weights = self.softmax(output)
    #     if distribution is not None:
    #         expert_outputs = distribution
    #     else:
    #         expert_outputs = []
    #         for idx, expert in enumerate(self.experts):
    #             expert_output = expert.pred_prob(x[:, idx, :].detach().cpu())
    #             expert_outputs.append(expert_output.unsqueeze(2))
    #         expert_outputs = torch.cat(expert_outputs, dim=2).to(gate_weights.device)
    #     weighted_sum = torch.sum(expert_outputs * gate_weights.unsqueeze(1), dim=2)
    #     return weighted_sum
    def forward(self, x, distribution=None, labels=None):
        if distribution is not None:
            expert_outputs = distribution
        else:
            expert_outputs = []
            for idx, expert in enumerate(self.experts):
                expert_output = expert.pred_prob(x[:, idx+self.skip_num, :].detach().cpu().numpy())
                expert_outputs.append(expert_output.unsqueeze(2))
            expert_outputs = torch.cat(expert_outputs, dim=2).to(x.device)

        if self.classifier_type == 'mlp':
            expert_outputs = expert_outputs.view(expert_outputs.size(0), -1)
            outputs = self.gating_network(expert_outputs)
            return outputs
        elif self.classifier_type == 'svm':
            expert_outputs = expert_outputs.view(expert_outputs.size(0), -1)
            if self.training:
                self.classifier.train(expert_outputs.detach().cpu().numpy(), labels)
            else:
                return self.classifier.predict(expert_outputs.detach().cpu().numpy())
        elif self.classifier_type == 'entropy':
            expert_outputs = expert_outputs.transpose(1, 2)
            entropy = self.entropy(expert_outputs)
            min_value = entropy.min(dim=1, keepdim=True).values
            max_value = entropy.max(dim=1, keepdim=True).values
            normalized_entropy = (entropy - min_value) / (max_value - min_value)
            _, indices = torch.topk(normalized_entropy, self.topk, dim=-1, largest=False)
            result = torch.full_like(entropy, float('inf'))
            result.scatter_(dim=1, index=indices, src=normalized_entropy.gather(dim=1, index=indices))
            weight = F.softmax(-result / self.temperature, dim=-1).unsqueeze(2)
            prob = torch.sum(expert_outputs * weight, dim=1)
            return prob

class EnsembleClassifier:
    def __init__(self, input_size, output_size, num_experts, expert_paths, expert_type='NN', classifier_type='mlp', skip_num=0, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_type = classifier_type
        self.num_experts = num_experts
        self.skip_num = skip_num
        self.output_size = output_size

        self.model = EnsembleModule(input_size, output_size, expert_paths, expert_type, classifier_type, skip_num, **kwargs).to(self.device)

        if classifier_type == 'mlp':
            # Ensemble Classifier Parameters
            self.alpha = kwargs.get('alpha')
            self.epochs = kwargs.get('epochs')
            self.batch_size = kwargs.get('batch_size')
            self.patience = kwargs.get('patience')
            self.current_patience = 0

            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.AdamW(self.model.parameters(), lr=kwargs.get('lr') if kwargs.get('lr') is not None else 1e-3, weight_decay=kwargs.get('alpha') if kwargs.get('alpha') is not None else 1e-5)

            self.best_model_state_dict = None
            self.best_optimizer_state_dict = None
        elif classifier_type == 'entropy':
            self.temperature = kwargs["temperature"]
            self.topk = kwargs["topk"]

    def train(self, X_train, y_train, X_val, y_val):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)

        if X_train_tensor.size(0) > 0:
            train_expert_outputs = []
            for expert_idx in range(self.num_experts):
                expert_output = self.model.experts[expert_idx].pred_prob(X_train_tensor[:, expert_idx+self.skip_num, :].detach().cpu().numpy())
                train_expert_outputs.append(expert_output.unsqueeze(2))
            train_expert_outputs = torch.cat(train_expert_outputs, dim=2).to(self.device)

        val_expert_outputs = []
        for expert_idx in range(self.num_experts):
            expert_output = self.model.experts[expert_idx].pred_prob(
                X_val_tensor[:, expert_idx+self.skip_num, :].detach().cpu().numpy())
            val_expert_outputs.append(expert_output.unsqueeze(2))
        val_expert_outputs = torch.cat(val_expert_outputs, dim=2).to(self.device)

        if self.classifier_type == 'mlp':
            # Convert the expert outputs to TensorDataset
            train_dataset = TensorDataset(X_train_tensor, train_expert_outputs, y_train_tensor)

            # Create DataLoader
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            best_val_loss = float('inf')

            for epoch in range(self.epochs):
                for inputs, input_distribution, labels in train_dataloader:
                    inputs, input_distribution, labels = inputs.to(self.device), input_distribution.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(None, input_distribution)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                # Validation
                with torch.no_grad():
                    self.model.eval()
                    val_outputs = self.model(None, val_expert_outputs)
                    val_loss = self.criterion(val_outputs, y_val_tensor)

                print(
                    f'Epoch [{epoch + 1}/{self.epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.current_patience = 0
                    self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                    self.best_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
                else:
                    self.current_patience += 1
                    if self.current_patience >= self.patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

                self.model.train()

            if self.best_model_state_dict is not None:
                self.model.load_state_dict(self.best_model_state_dict)
                self.optimizer.load_state_dict(self.best_optimizer_state_dict)

            self.model.eval()
            val_outputs = self.model(None, val_expert_outputs)
            val_loss = self.criterion(val_outputs, y_val_tensor)
            print(f"best val loss: {val_loss}")

        elif self.classifier_type == 'svm':
            self.model(None, train_expert_outputs, y_train_tensor.detach().cpu().numpy())
            self.model.eval()
            val_preds = self.model(None, val_expert_outputs)
            # metric = self.evaluation(val_preds, y_val_tensor.detach().cpu().numpy())
            # print(f"val f1: {metric}")

        elif self.classifier_type == 'entropy':
            prob = self.model(None, val_expert_outputs)
            _, predicted = torch.max(prob, -1)
            # metric = self.evaluation(predicted.detach().cpu().numpy(), y_val_tensor.detach().cpu().numpy())
            # print(f"val f1: {metric}")


    def predict(self, input_matrix, batch_size=128):
        input_tensor = torch.FloatTensor(input_matrix)
        if self.classifier_type == 'mlp':
            with torch.no_grad():
                self.model.eval()
                all_predictions = []

                if batch_size is None:
                    input_tensor = input_tensor.to(self.device)
                    output = self.model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                else:
                    for i in range(0, input_tensor.size(0), batch_size):
                        batch_input = input_tensor[i:i + batch_size]
                        batch_input = batch_input.to(self.device)
                        output = self.model(batch_input)
                        _, predicted = torch.max(output, 1)
                        all_predictions.extend(predicted.cpu().numpy())
        elif self.classifier_type == 'svm':
            self.model.eval()
            all_predictions = self.model(input_tensor)

        elif self.classifier_type == 'entropy':
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            all_predictions = predicted.cpu().numpy()

        return np.array(all_predictions)

    def save(self, filename):
        if self.classifier_type == 'mlp':
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.epochs,
                'loss': self.criterion,
                'patience': self.patience,
                'current_patience': self.current_patience
            }, filename)

        elif self.classifier_type == 'svm':
            self.model.classifier.save(filename)

        elif self.classifier_type == 'entropy':
            torch.save({
                'temperature': self.temperature,
                'topk': self.topk,
            }, filename)

    def load(self, filename):
        if self.classifier_type == 'mlp':
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epochs = checkpoint['epoch']
            self.criterion = checkpoint['loss']
            self.patience = checkpoint['patience']
            self.current_patience = checkpoint['current_patience']
        elif self.classifier_type == 'svm':
            self.model.classifier.load(filename)
        elif self.classifier_type == 'entropy':
            checkpoint = torch.load(filename)
            self.temperature = checkpoint['temperature']
            self.topk = checkpoint['topk']
