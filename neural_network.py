"""
Neural network implementation from scratch using NumPy
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


class ActivationFunction:
    """Activation functions"""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)  # جلوگیری از overflow
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return x
    
    @staticmethod
    def linear_derivative(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class DenseLayer:
    """لایه Dense با قابلیت dropout و batch normalization"""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = 'relu',
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # He initialization
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros(output_size) if use_bias else None
        
        # تنظیم تابع فعال‌سازی
        self.activation_name = activation
        if activation == 'relu':
            self.activation = ActivationFunction.relu
            self.activation_derivative = ActivationFunction.relu_derivative
        elif activation == 'tanh':
            self.activation = ActivationFunction.tanh
            self.activation_derivative = ActivationFunction.tanh_derivative
        elif activation == 'sigmoid':
            self.activation = ActivationFunction.sigmoid
            self.activation_derivative = ActivationFunction.sigmoid_derivative
        elif activation == 'linear':
            self.activation = ActivationFunction.linear
            self.activation_derivative = ActivationFunction.linear_derivative
        else:
            raise ValueError(f"Activation function '{activation}' not supported")
        
        # Batch normalization parameters
        if use_batch_norm:
            self.gamma = np.ones(output_size)
            self.beta = np.zeros(output_size)
            self.running_mean = np.zeros(output_size)
            self.running_var = np.ones(output_size)
            self.epsilon = 1e-8
            self.momentum = 0.9
        
        # برای ذخیره مقادیر در forward pass
        self.last_input = None
        self.last_output = None
        self.last_dropout_mask = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass"""
        self.last_input = x.copy()
        
        # محاسبه خروجی لایه
        output = np.dot(x, self.weights)
        
        if self.use_bias:
            output += self.bias
        
        # Batch normalization
        if self.use_batch_norm:
            if training:
                batch_mean = np.mean(output, axis=0)
                batch_var = np.var(output, axis=0)
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
                normalized = (output - batch_mean) / np.sqrt(batch_var + self.epsilon)
            else:
                normalized = (output - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            # Store normalized for gradient computation
            self.last_normalized = normalized.copy()
            output = self.gamma * normalized + self.beta
        
        # فعال‌سازی
        activated = self.activation(output)
        self.last_output = activated.copy()
        
        # Dropout
        if training and self.dropout_rate > 0:
            self.last_dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, activated.shape) / (1 - self.dropout_rate)
            activated = activated * self.last_dropout_mask
        
        return activated
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass"""
        # Dropout gradient
        if self.last_dropout_mask is not None:
            grad_output = grad_output * self.last_dropout_mask
        
        # Gradient از تابع فعال‌سازی
        grad_activation = grad_output * self.activation_derivative(self.last_output)
        
        # Batch normalization gradient
        if self.use_batch_norm:
            # محاسبه gradient برای batch norm
            # During forward: output = gamma * normalized + beta
            # where normalized = (x - mean) / sqrt(var + eps)
            # 
            # Gradient w.r.t. gamma: dL/dgamma = sum(grad_activation * normalized)
            # Gradient w.r.t. beta: dL/dbeta = sum(grad_activation)
            # Gradient w.r.t. normalized: dL/dnormalized = grad_activation * gamma
            # 
            # For simplicity, we compute gradients for gamma and beta,
            # but use simplified gradient flow for the input (just multiply by gamma)
            if not hasattr(self, 'last_normalized'):
                # Fallback: if normalized wasn't stored, just pass through with gamma
                grad_normalized = grad_activation * self.gamma
            else:
                # Compute gradients for gamma and beta
                if len(grad_activation.shape) > 1:
                    # Batch case
                    self.grad_gamma = np.sum(grad_activation * self.last_normalized, axis=0)
                    self.grad_beta = np.sum(grad_activation, axis=0)
                else:
                    # Single sample case
                    self.grad_gamma = grad_activation * self.last_normalized
                    self.grad_beta = grad_activation
                # Gradient w.r.t. normalized input (simplified - full computation would be more complex)
                grad_normalized = grad_activation * self.gamma
            grad_activation = grad_normalized
        
        # Gradient نسبت به وزن‌ها و bias
        grad_weights = np.dot(self.last_input.T, grad_activation)
        grad_bias = np.sum(grad_activation, axis=0) if self.use_bias else None
        grad_input = np.dot(grad_activation, self.weights.T)
        
        # ذخیره gradient‌ها برای به‌روزرسانی
        self.grad_weights = grad_weights
        self.grad_bias = grad_bias
        
        return grad_input
    
    def get_parameters(self) -> List[np.ndarray]:
        """بازگرداندن پارامترهای قابل آموزش"""
        params = [self.weights]
        if self.use_bias:
            params.append(self.bias)
        if self.use_batch_norm:
            params.extend([self.gamma, self.beta])
        return params
    
    def get_gradients(self) -> List[np.ndarray]:
        """بازگرداندن gradient‌ها"""
        if not hasattr(self, 'grad_weights'):
            # اگر backward صدا نشده باشد
            self.grad_weights = np.zeros_like(self.weights)
            self.grad_bias = np.zeros_like(self.bias) if self.use_bias else None
        
        grads = [self.grad_weights]
        if self.use_bias:
            grads.append(self.grad_bias)
        if self.use_batch_norm:
            # برای batch norm، gradient‌های gamma و beta را محاسبه می‌کنیم
            if hasattr(self, 'grad_gamma') and hasattr(self, 'grad_beta'):
                grads.extend([self.grad_gamma, self.grad_beta])
            else:
                # Fallback if gradients weren't computed
                grads.extend([None, None])
        return grads


class AdamOptimizer:
    """بهینه‌ساز Adam از صفر"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}  # moment estimates
        self.v = {}  # velocity estimates
    
    def update(self, params: List[np.ndarray], grads: List[np.ndarray], param_ids: List[int]):
        """به‌روزرسانی پارامترها با Adam"""
        self.t += 1
        
        for param, grad, param_id in zip(params, grads, param_ids):
            if grad is None:
                continue
            
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param)
                self.v[param_id] = np.zeros_like(param)
            
            # به‌روزرسانی moment estimates
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
            
            # به‌روزرسانی پارامتر
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class NeuralNetwork:
    """شبکه عصبی کامل با قابلیت‌های پیشرفته"""
    
    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str] = None,
        dropout_rates: List[float] = None,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001
    ):
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 2) + ['linear']
        if dropout_rates is None:
            dropout_rates = [0.0] * (len(layer_sizes) - 1)
        
        assert len(activations) == len(layer_sizes) - 1
        assert len(dropout_rates) == len(layer_sizes) - 1
        
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(
                layer_sizes[i],
                layer_sizes[i + 1],
                activation=activations[i],
                dropout_rate=dropout_rates[i],
                use_batch_norm=use_batch_norm if i < len(layer_sizes) - 2 else False
            )
            self.layers.append(layer)
        
        self.optimizer = AdamOptimizer(learning_rate=learning_rate)
        self.param_counter = 0
        self.param_ids = {}
        self._register_parameters()
    
    def _register_parameters(self):
        """ثبت شناسه‌های پارامترها"""
        for layer_idx, layer in enumerate(self.layers):
            params = layer.get_parameters()
            for param_idx, param in enumerate(params):
                param_id = self.param_counter
                self.param_ids[(layer_idx, param_idx)] = param_id
                self.param_counter += 1
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass در تمام لایه‌ها"""
        output = x
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output
    
    def backward(self, grad_output: np.ndarray):
        """Backward pass در تمام لایه‌ها"""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update(self):
        """به‌روزرسانی وزن‌ها با Adam"""
        all_params = []
        all_grads = []
        all_ids = []
        
        for layer_idx, layer in enumerate(self.layers):
            params = layer.get_parameters()
            grads = layer.get_gradients()
            for param_idx, (param, grad) in enumerate(zip(params, grads)):
                all_params.append(param)
                all_grads.append(grad)
                all_ids.append(self.param_ids[(layer_idx, param_idx)])
        
        self.optimizer.update(all_params, all_grads, all_ids)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """پیش‌بینی (بدون training mode)"""
        return self.forward(x, training=False)
    
    def get_num_parameters(self) -> int:
        """تعداد کل پارامترهای شبکه"""
        total = 0
        for layer in self.layers:
            params = layer.get_parameters()
            for param in params:
                total += param.size
        return total

