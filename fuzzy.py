import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class FuzzyNeuralNetwork:
    def __init__(self):
        # Fuzzy parameters
        self.pixel_range = np.linspace(0, 255, 256)
        
        # Membership function parameters
        self.mf_params = {
            'very_dark': [0, 0, 63],    
            'dark': [32, 96, 159],      
            'medium': [128, 160, 191],  
            'light': [160, 224, 287],   
            'very_light': [192, 255, 255]
        }
        
        # Neural Network Model
        self.model = None
        self.build_model()

    def build_model(self):
        """Build Keras model with improved architecture"""
        self.model = Sequential([
            # Input layer
            Dense(256, input_shape=(784 * 5,), activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output layer
            Dense(10, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def triangular_mf_vectorized(self, x, params):
        """Vectorized triangular membership function"""
        a, b, c = params
        x = np.asarray(x)
        
        membership = np.zeros_like(x, dtype=float)
        
        mask1 = (x > a) & (x <= b)
        if b != a:
            membership[mask1] = (x[mask1] - a) / (b - a)
        
        mask2 = (x > b) & (x <= c)
        if c != b:
            membership[mask2] = (c - x[mask2]) / (c - b)
        
        membership[x == b] = 1.0
        
        return np.clip(membership, 0, 1)

    def fuzzify_image_vectorized(self, image):
        """Vectorized image fuzzification"""
        if len(image.shape) > 1:
            image = image.reshape(-1)
        
        image = image * 255
        fuzzy_values = np.zeros((len(image), 5))
        
        for i, (_, params) in enumerate(self.mf_params.items()):
            fuzzy_values[:, i] = self.triangular_mf_vectorized(image, params)
        
        return fuzzy_values.flatten()

    def prepare_data(self, X):
        """Optimized data preparation with batch processing"""
        if len(X.shape) == 2:
            m = X.shape[0]
            batch_size = 100
            X_fuzzy = np.zeros((m, 784 * 5))
            
            for i in range(0, m, batch_size):
                end_idx = min(i + batch_size, m)
                batch = X[i:end_idx]
                X_fuzzy[i:end_idx] = np.vstack([
                    self.fuzzify_image_vectorized(img) for img in batch
                ])
                if (i + batch_size) % 1000 == 0:
                    print(f"Processed {i + batch_size}/{m} images")
            return X_fuzzy
        else:
            return self.fuzzify_image_vectorized(X).reshape(1, -1)

    def train_network(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """Train the neural network using Keras"""
        print("Preparing fuzzy features...")
        X_fuzzy = self.prepare_data(X)
        
        # Create model directory if it doesn't exist
        os.makedirs('model_checkpoints', exist_ok=True)
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                'model_checkpoints/best_model.keras',  # Updated extension
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        print("\nTraining neural network...")
        history = self.model.fit(
            X_fuzzy, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def predict(self, X):
        """Make predictions using the trained model"""
        X_fuzzy = self.prepare_data(X)
        predictions = self.model.predict(X_fuzzy, verbose=0)  # Added verbose=0 to reduce output
        return np.argmax(predictions, axis=1)

    def save_model(self, filename='fuzzy_nn_model.keras'):  # Updated extension
        """Save the Keras model"""
        self.model.save(filename)
        print("Model saved successfully!")

    def load_model(self, filename='fuzzy_nn_model.keras'):  # Updated extension
        """Load the Keras model"""
        try:
            self.model = load_model(filename)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def train_model():
    """Train model with improved parameters"""
    print("Loading MNIST dataset...")
    try:
        mnist = fetch_openml('mnist_784')
        train_size = 10000  # Increased training size
        X = mnist.data.to_numpy()[:train_size]
        y = mnist.target.astype(int)[:train_size]
        
        # Normalize pixel values
        X = X / 255.0
        
        # Create and train model
        model = FuzzyNeuralNetwork()
        history = model.train_network(
            X, y,
            validation_split=0.2,
            epochs=20,
            batch_size=32
        )
        
        # Save model
        model.save_model()
        
        # Print final metrics
        val_accuracy = max(history.history['val_accuracy']) * 100
        print(f"\nBest Validation Accuracy: {val_accuracy:.2f}%")
        
        return model
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def main():
    model = FuzzyNeuralNetwork()
    
    while True:
        print("\nFuzzy Neural Network Digit Recognition")
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Predict from image")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            model = train_model()
            
        elif choice == '2':
            if not model.load_model():
                print("No saved model found. Please train a new model first.")
                
        elif choice == '3':
            if model.model is None:
                print("Please train or load a model first.")
                continue
                
            image_path = input("Enter image path: ")
            try:
                img = Image.open(image_path).convert('L')
                img = img.resize((28, 28))
                img_array = np.array(img).reshape(1, 784) / 255.0
                prediction = model.predict(img_array)[0]
                print(f"Predicted digit: {prediction}")
            except Exception as e:
                print(f"Error processing image: {e}")
                
        elif choice == '4':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
