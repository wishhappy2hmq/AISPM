import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


class TimeSeriesPredictor:
    def __init__(self, n_features=3, test_size=0.2, random_state=42):
        self.n_features = n_features
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False

    def load_and_prepare_data(self, file_path, sheet_name=0):
        self.df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(self.df.head())
        if self.df.shape[1] <= self.n_features:
            raise ValueError(f"Data has only {self.df.shape[1]} columns, but at least {self.n_features + 1} columns are required")

        return self.df

    def create_features_target(self):
        self.X = self.df.iloc[:, :self.n_features].values
        self.y = self.df.iloc[:, self.n_features].values
        self.feature_names = self.df.columns[:self.n_features].tolist()
        self.target_name = self.df.columns[self.n_features]

        return self.X, self.y

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess_data(self):
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        return self.X_train_scaled, self.X_test_scaled

    def train_model(self, model=None):
        if model is None:
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            self.model = model
        self.model.fit(self.X_train_scaled, self.y_train)
        self.is_fitted = True
        self.y_train_pred = self.model.predict(self.X_train_scaled)

        return self.model

    def predict(self, X=None):
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Please call train_model() first")

        if X is None:
            self.y_pred = self.model.predict(self.X_test_scaled)
            return self.y_pred
        else:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)

    def evaluate_model(self):
        if not hasattr(self, 'y_pred'):
            self.y_pred = self.predict()
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }

        return metrics

    def visualize_input_vs_prediction(self, num_samples=10, figsize=(15, 10)):
        if not hasattr(self, 'y_pred'):
            self.y_pred = self.predict()
        display_indices = np.random.choice(len(self.X_test), min(num_samples, len(self.X_test)), replace=False)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        for i, idx in enumerate(display_indices):
            input_data = self.X_test[idx]
            actual_target = self.y_test[idx]
            predicted_target = self.y_pred[idx]
            x_positions = range(1, self.n_features + 1)
            axes[0, 0].plot(x_positions, input_data, 'o-', alpha=0.7,
                            label=f'Sample {i + 1}' if i < 3 else "")
            axes[0, 0].plot(self.n_features + 1, actual_target, 's',
                            color='green', markersize=8, alpha=0.8)
            axes[0, 0].plot(self.n_features + 1, predicted_target, '^',
                            color='red', markersize=8, alpha=0.8)

        axes[0, 0].axvline(x=self.n_features + 0.5, color='gray', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Time Steps/Features')
        axes[0, 0].set_ylabel('Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        detailed_idx = display_indices[0]
        detailed_input = self.X_test[detailed_idx]
        detailed_actual = self.y_test[detailed_idx]
        detailed_pred = self.y_pred[detailed_idx]
        time_points = list(range(1, self.n_features + 2))
        values = list(detailed_input) + [detailed_actual]
        predicted_values = list(detailed_input) + [detailed_pred]

        axes[0, 1].plot(time_points[:-1], values[:-1], 'bo-', linewidth=2,
                        markersize=8, label='Input Data')
        axes[0, 1].plot(time_points[-2:], [values[-2], values[-1]], 'go-',
                        linewidth=2, markersize=8, label='Actual Target')
        axes[0, 1].plot(time_points[-2:], [values[-2], detailed_pred], 'ro--',
                        linewidth=2, markersize=8, label='Predicted Target')
        axes[0, 1].axvline(x=self.n_features + 0.5, color='gray', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Values')
        axes[0, 1].set_title(f'Sample Detailed Analysis\nActual: {detailed_actual:.2f}, Predicted: {detailed_pred:.2f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        time_series_length = min(100, len(self.y_test))
        time_index = range(time_series_length)

        axes[1, 0].plot(time_index, self.y_test[:time_series_length],
                        'g-', linewidth=2, label='Actual', alpha=0.8)
        axes[1, 0].plot(time_index, self.y_pred[:time_series_length],
                        'r--', linewidth=2, label='Predicted', alpha=0.8)
        axes[1, 0].set_xlabel('Samples')
        axes[1, 0].set_ylabel('Target Values')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        errors = self.y_test - self.y_pred
        axes[1, 1].hist(errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Prediction Errors')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

    def visualize_continuous_prediction(self, num_sequences=3, figsize=(15, 5)):
        if not hasattr(self, 'y_pred'):
            self.y_pred = self.predict()

        fig, axes = plt.subplots(1, num_sequences, figsize=figsize)
        if num_sequences == 1:
            axes = [axes]

        fig.suptitle('Continuous Input-Prediction Sequence Visualization', fontsize=16, fontweight='bold')

        for i in range(num_sequences):
            # Select a random sample
            idx = np.random.randint(len(self.X_test))

            # Input sequence
            input_sequence = self.X_test[idx]
            actual_value = self.y_test[idx]
            predicted_value = self.y_pred[idx]

            # Create complete time series (input + prediction)
            time_points = list(range(1, self.n_features + 2))
            complete_sequence = list(input_sequence) + [actual_value]
            prediction_sequence = list(input_sequence) + [predicted_value]

            # Plot curves
            axes[i].plot(time_points, complete_sequence, 'go-',
                         linewidth=2, markersize=6, label='Input Data + Actual')
            axes[i].plot(time_points, prediction_sequence, 'ro--',
                         linewidth=2, markersize=6, label='Input Data + Predicted')

            # Mark turning point
            axes[i].axvline(x=self.n_features + 0.5, color='gray',
                            linestyle='--', alpha=0.5, label='Prediction Point')

            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Values')
            axes[i].set_title(f'Sequence {i + 1}\nError: {abs(actual_value - predicted_value):.3f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig


def main():
    excel_path = 'little_01.xlsx'
    predictor = TimeSeriesPredictor(n_features=3, test_size=0.4)
    predictor.load_and_prepare_data(excel_path)
    X, y = predictor.create_features_target()
    X_train, X_test, y_train, y_test = predictor.split_data()
    X_train_scaled, X_test_scaled = predictor.preprocess_data()

    predictor.train_model()
    y_pred = predictor.predict()
    metrics = predictor.evaluate_model()
    predictor.visualize_input_vs_prediction(num_samples=8)
    predictor.visualize_continuous_prediction(num_sequences=3)


if __name__ == "__main__":
    main()