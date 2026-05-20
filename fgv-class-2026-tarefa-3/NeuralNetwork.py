import numpy as np

# ___________________________________ FUNÇÕES DE ATIVAÇÃO ___________________________________________________________
class ActivationFunction:
    
    def __init__(self, function, derivative = None):
        self.function = function
        self.derivative = derivative
        
def sigmoid(a): return 1/(1 + np.exp(-a))
def sigmoid_prime(a): return sigmoid(a) * (1 - sigmoid(a))

sigmoid_activation = ActivationFunction(sigmoid, sigmoid_prime)

def softmax(a):
    exp_a = np.exp(a)
    return exp_a / exp_a.sum(axis=1, keepdims=True)

def identity(a): return a

#__________________________________ FUNÇÕES DE ERRO _________________________________________________________________

def sse(target, y_prev): return np.sum((target - y_prev)**2) * 0.5

def cross_entropy_multiclass(target, y_prev): 
    n = len(target)
    return - np.sum(np.multiply(target, np.log(y_prev))) / n

# ___________________________________ MULTILAYER PERCEPTRON ___________________________________________________________

class MultilayerPerceptron:

    def __init__(self, hidden_activation: ActivationFunction, output_activation, sizes: list):
        """A lista "sizes" deve conter todos as dimensões das camadas da nossa rede, desde a dimensão de entrada
        até a de saída. Tanto os pesos, quanto os biases estarão na matriz de parâmetros."""

        self.num_layers = len(sizes) - 1
        self.sizes = sizes
        self.hidden_activation = hidden_activation.function
        self.hidden_activation_prime = hidden_activation.derivative
        self.output_activation = output_activation
        self.parameters = [np.random.randn(x + 1,y) for x,y in zip(sizes[:-1], sizes[1:])]
    

    def _initialize_parameters(self):
        self.parameters = [np.random.randn(x + 1,y) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
    

    @staticmethod
    def _add_ones(X):
        """Essa função vai nos permitir adicionar uma coluna de uns aos nossos dados, para conseguirmos 
        multiplicar pelos biases."""
    
        return np.column_stack((np.ones(X.shape[0]), X))
    

    def _forward_pass(self, X):
        """Realiza o forward_pass sem ficar armazendo os valores necessários para o backpropagation.
        
        Parameters
        ----------
            X: np.ndarray
                Conjunto de dados.
                
        Returns
        -------
            y: np.ndarray
                Matriz de predições.
            Zs: list[np.ndarray]
                Lista com as os dados depois das ativações e com a coluna de uns dos biases.
            As: list[np.ndarray]
                Lista com os dados depois das ativações."""
                
        z = self._add_ones(X)
        Zs = [z]
        As = []
        
        #Hidden Layers
        for w in self.parameters[:-1]:
            a = z @ w
            As.append(a)
            
            z = self.hidden_activation(a) 
            z = self._add_ones(z)
            Zs.append(z)

        #Output Layer
        a = z@self.parameters[-1]
        As.append(a)
        y = self.output_activation(a)

        return y, Zs, As
    

    def _fast_forward_pass(self, X):
        """Realiza o forward_pass sem ficar armazendo os valores necessários para o backpropagation.
        
        Parameters
        ----------
            X: np.ndarray
                Conjunto de dados.
                
        Returns
        -------
            y: np.ndarray
                Matriz de predições.
        """
        
        z = self._add_ones(X)
        
        #Hidden Layers
        for w in self.parameters[:-1]:
            a = z @ w
            
            z = self.hidden_activation(a) 
            z = self._add_ones(z)

        #Output Layer
        a = z@self.parameters[-1]
        y = self.output_activation(a)

        return y
        

    @staticmethod
    def _output_derivative(y_prev, target):
        return y_prev - target
    

    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val,
            update_method = "mini_batch",
            batch_size = 32, 
            lr = 0.01, 
            epochs=1, 
            patience = 10,
            weight_decay = 0.0):
        
        """Treina os parâmetros da nossa rede utilizando o método de mini-batch. Os casos do Stochastic Gradient Descent (SGD)
        e do método batch, são os casos especiais em que batch_size = 1 e batch_size = X.shape[0] (número de data points), respectivamente."""

        if update_method == "sgd":
            batch_size = 1
            
        if update_method == "batch":
            batch_size = X_train.shape[0]
            
        n_samples = X_train.shape[0]
        loss_history_train = []
        loss_history_val = [] 

        for epoch in range(epochs):

            for start in range(0, n_samples, batch_size):
                X_batch = X_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]

                delta_w = self.backpropagation(X_batch, y_batch)

                for i, dw in enumerate(delta_w):
                    
                    if weight_decay > 0:
                        regularization = np.zeros_like(self.parameters[i])
                        regularization[1:] = weight_decay * self.parameters[i][1:] 
                        dw = dw + regularization
                    
                    self.parameters[i] -= lr * dw

            y_pred_train = self._fast_forward_pass(X_train)
            y_pred_val = self._fast_forward_pass(X_val)
            loss_history_train.append(self.compute_loss(y_pred_train, y_train))
            loss_history_val.append(self.compute_loss(y_pred_val, y_val))
            
            if epoch > patience:
                if loss_history_val[epoch] > loss_history_val[epoch - 1]:
                    break

        return loss_history_train, loss_history_val

# ___________________________________ MULTILAYER PERCEPTRON (REGRESSÃO) ___________________________________________________________

class MultilayerPerceptronRegression(MultilayerPerceptron):
    
    def __init__(self, hidden_activation, output_activation, sizes):
        super().__init__(hidden_activation, output_activation, sizes)
    
    def backpropagation(self, X, y):
        n = X.shape[0]
        delta_w = [np.zeros_like(w) for w in self.parameters]

        y_pred, Zs, As = self._forward_pass(X)

        n_out = y_pred.shape[1]
        if n_out == 1:
            y_enc = y.reshape(-1, 1)          

        delta = self._output_derivative(y_pred, y_enc)  
        delta_w[-1] = Zs[-1].T @ delta / n

        for l in range(2, self.num_layers):
            delta = (delta @ self.parameters[-l + 1][1:].T) * self.hidden_activation_prime(As[-l])
            
            delta_w[-l] = Zs[-l].T @ delta / n

        return delta_w
    
    def predict(self, X):
        y_pred = self._fast_forward_pass(X)
        
        if y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
            
        return y_pred
    
    @staticmethod
    def sse(target, y_prev): return np.sum((target - y_prev)**2) * 0.5

    @staticmethod
    def rmse(target, y_prev): return np.sqrt(np.mean((target - y_prev)**2))

    def compute_loss(self, y_prev, target):
        return self.rmse(y_prev, target)
    
# ___________________________________ MULTILAYER PERCEPTRON (CLASSIFICAÇÃO) ___________________________________________________________

class MultilayerPerceptronClassification(MultilayerPerceptron):

    def __init__(self, hidden_activation, output_activation, sizes):
        super().__init__(hidden_activation, output_activation, sizes)
        self.num_classes = sizes[-1]
    
    @staticmethod
    def _one_hot(y, num_classes):
        """Converte labels inteiros em matriz com uns nas classes correspondentes"""
        
        Y = np.zeros((len(y), num_classes))
        Y[np.arange(len(y)), y] = 1
        return Y
        

    @staticmethod
    def _output_derivative(y_prev, target):
        return y_prev - target
    
    def backpropagation(self, X, y):
        n      = X.shape[0]
        delta_w = [np.zeros_like(w) for w in self.parameters]

        y_pred, Zs, As = self._forward_pass(X)

        if y.ndim == 1:
            y_target  = self._one_hot(y, self.num_classes)         

        delta = self._output_derivative(y_pred, y_target)  
        delta_w[-1] = Zs[-1].T @ delta / n

        for l in range(2, self.num_layers):
            delta = (delta @ self.parameters[-l + 1][1:].T) * self.hidden_activation_prime(As[-l])
            
            delta_w[-l] = Zs[-l].T @ delta / n

        return delta_w
    
    def predict(self, X):
        return self._fast_forward_pass(X)
    
    def predict_classes(self, X):
        """Retorna o índice da classe com maior probabilidade."""
        
        y_pred = self._fast_forward_pass(X)
        return np.argmax(y_pred, axis=1)
    
    def cross_entropy_multiclass(self, target, y_prev):
        t = self._one_hot(target, self.num_classes)
        n = len(target)
        return - np.sum(np.multiply(t, np.log(y_prev))) 
    
    def compute_loss(self, y_prev, target):
        return self.cross_entropy_multiclass(y_prev, target)