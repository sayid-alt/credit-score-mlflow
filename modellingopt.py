import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("credit_scoring_tutorial_optimization")

data = pd.read_csv("train_pca.csv")
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Credit_Score", axis=1),
    data["Credit_Score"],
    random_state=42,
    test_size=0.2
)

input_example = X_train[0:5]

n_estimators_range = np.linspace(10, 1000, 5, dtype=int)  # 5 evenly spaced values
max_depth_range = np.linspace(1, 50, 5, dtype=int) 

best_accuracy = 0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"n_estimators_{n_estimators}_max_depth_{max_depth}"):
            mlflow.autolog()

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            
            model.fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                print(f"Best Accuracy: {best_accuracy}")
                print(f"Best Hyperparameters: {best_params}")

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    input_example=input_example,
                )

