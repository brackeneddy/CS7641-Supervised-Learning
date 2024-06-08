from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np

# Load wholesale data from UCI
wholesale_customers_data = fetch_ucirepo(id=292)
wholesale_customers = wholesale_customers_data.data.features

# Load car data from UCI
car_evaluation_data = fetch_ucirepo(id=19)
car_evaluation = pd.concat(
    [car_evaluation_data.data.features, car_evaluation_data.data.targets], axis=1
)

# Preprocess wholesale
wholesale_customers["Total Spend"] = wholesale_customers[
    ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
].sum(axis=1)
wholesale_customers["Category"] = pd.qcut(
    wholesale_customers["Total Spend"], q=3, labels=["Low", "Medium", "High"]
)

label_encode_wholesale = LabelEncoder()
wholesale_customers["Category"] = label_encode_wholesale.fit_transform(
    wholesale_customers["Category"]
)

x_wholesale = wholesale_customers.drop(columns=["Total Spend", "Category"])
y_wholesale = wholesale_customers["Category"]

scaler_wholesale = StandardScaler()

x_wholesale_scaled = scaler_wholesale.fit_transform(x_wholesale)
x_wholesale_train, x_wholesale_test, y_wholesale_train, y_wholesale_test = (
    train_test_split(x_wholesale_scaled, y_wholesale, test_size=0.3, random_state=42)
)

# Preprocess car
label_encode_car = LabelEncoder()

for column in ["buying", "maint", "doors", "persons", "lug_boot", "safety"]:
    car_evaluation[column] = label_encode_car.fit_transform(car_evaluation[column])

car_evaluation["class"] = label_encode_car.fit_transform(car_evaluation["class"])

X_car = car_evaluation.drop(columns=["class"])
y_car = car_evaluation["class"]

scaler_car = StandardScaler()

X_car_scaled = scaler_car.fit_transform(X_car)
X_car_train, X_car_test, y_car_train, y_car_test = train_test_split(
    X_car_scaled, y_car, test_size=0.3, random_state=42
)


# Plot learning curve
def plot_learning_curve(estimator, title, x, y, cv=5, n_jobs=None):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )

    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )

    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.legend(loc="best")
    plt.show()


# Plot validation curve
def plot_validation_curve(
    estimator, title, X, y, param_name, param_range, cv=5, scoring="accuracy"
):
    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )

    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )

    plt.plot(param_range, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        param_range, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
    plt.show()


# Grid Search for Neural Network
param_grid_nn = {
    "hidden_layer_sizes": [(50,), (100,), (200,)],
    "alpha": [0.0001, 0.001, 0.01],
    "max_iter": [1000, 1500, 2000],
}

nn_grid_search = GridSearchCV(
    MLPClassifier(), param_grid_nn, cv=5, scoring="accuracy", n_jobs=-1
)

nn_grid_search.fit(x_wholesale_train, y_wholesale_train)
print("Best NN parameters (Wholesale Customers):", nn_grid_search.best_params_)
best_nn_wholesale = nn_grid_search.best_estimator_

# Grid Search for SVM
param_grid_svm = {
    "C": np.logspace(-3, 2, 6),
    "gamma": np.logspace(-4, 1, 6),
    "kernel": ["linear", "rbf"],
}

svm_grid_search = GridSearchCV(
    SVC(), param_grid_svm, cv=5, scoring="accuracy", n_jobs=-1
)

svm_grid_search.fit(x_wholesale_train, y_wholesale_train)
print("Best SVM parameters (Wholesale Customers):", svm_grid_search.best_params_)
best_svm_wholesale = svm_grid_search.best_estimator_

# Grid Search for k-NN
param_grid_knn = {"n_neighbors": np.arange(1, 11), "metric": ["euclidean", "manhattan"]}
knn_grid_search = GridSearchCV(
    KNeighborsClassifier(), param_grid_knn, cv=5, scoring="accuracy", n_jobs=-1
)

knn_grid_search.fit(x_wholesale_train, y_wholesale_train)
print("Best k-NN parameters (Wholesale Customers):", knn_grid_search.best_params_)
best_knn_wholesale = knn_grid_search.best_estimator_

# Grid Search for AdaBoost with pruning
param_grid_ada = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 1.0],
    "estimator__max_depth": [1, 2, 3],
    "estimator__min_samples_split": [2, 5, 10],
    "estimator__min_samples_leaf": [1, 2, 4],
}

ada_estimator = DecisionTreeClassifier()
ada_grid_search = GridSearchCV(
    AdaBoostClassifier(estimator=ada_estimator),
    param_grid_ada,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

ada_grid_search.fit(x_wholesale_train, y_wholesale_train)
print("Best AdaBoost parameters (Wholesale Customers):", ada_grid_search.best_params_)
best_ada_wholesale = ada_grid_search.best_estimator_

# Evaluate all best models using learning curves
plot_learning_curve(
    best_nn_wholesale,
    "Learning Curve (Best NN, Wholesale Customers)",
    x_wholesale_train,
    y_wholesale_train,
)

plot_learning_curve(
    best_svm_wholesale,
    "Learning Curve (Best SVM, Wholesale Customers)",
    x_wholesale_train,
    y_wholesale_train,
)

plot_learning_curve(
    best_knn_wholesale,
    "Learning Curve (Best k-NN, Wholesale Customers)",
    x_wholesale_train,
    y_wholesale_train,
)

plot_learning_curve(
    best_ada_wholesale,
    "Learning Curve (Best AdaBoost, Wholesale Customers)",
    x_wholesale_train,
    y_wholesale_train,
)

# Repeat Grid Search for Car Evaluation dataset
nn_grid_search.fit(X_car_train, y_car_train)
print("Best NN parameters (Car Evaluation):", nn_grid_search.best_params_)
best_nn_car = nn_grid_search.best_estimator_

svm_grid_search.fit(X_car_train, y_car_train)
print("Best SVM parameters (Car Evaluation):", svm_grid_search.best_params_)
best_svm_car = svm_grid_search.best_estimator_

knn_grid_search.fit(X_car_train, y_car_train)
print("Best k-NN parameters (Car Evaluation):", knn_grid_search.best_params_)
best_knn_car = knn_grid_search.best_estimator_

ada_grid_search.fit(X_car_train, y_car_train)
print("Best AdaBoost parameters (Car Evaluation):", ada_grid_search.best_params_)
best_ada_car = ada_grid_search.best_estimator_

# Evaluate all best models using learning curves for Car Evaluation dataset
plot_learning_curve(
    best_nn_car, "Learning Curve (Best NN, Car Evaluation)", X_car_train, y_car_train
)

plot_learning_curve(
    best_svm_car, "Learning Curve (Best SVM, Car Evaluation)", X_car_train, y_car_train
)

plot_learning_curve(
    best_knn_car, "Learning Curve (Best k-NN, Car Evaluation)", X_car_train, y_car_train
)

plot_learning_curve(
    best_ada_car,
    "Learning Curve (Best AdaBoost, Car Evaluation)",
    X_car_train,
    y_car_train,
)

# Plot validation curves for Neural Network (hidden_layer_sizes)
param_range_nn = [10, 50, 100, 200]

plot_validation_curve(
    MLPClassifier(max_iter=10000),
    "Validation Curve (NN, hidden_layer_sizes, Wholesale Customers)",
    x_wholesale_train,
    y_wholesale_train,
    param_name="hidden_layer_sizes",
    param_range=param_range_nn,
)

plot_validation_curve(
    MLPClassifier(max_iter=10000),
    "Validation Curve (NN, hidden_layer_sizes, Car Evaluation)",
    X_car_train,
    y_car_train,
    param_name="hidden_layer_sizes",
    param_range=param_range_nn,
)

# Plot validation curves for SVM (C parameter)
param_range_svm_c = np.logspace(-3, 2, 6)

plot_validation_curve(
    SVC(kernel="linear"),
    "Validation Curve (SVM, C parameter, Wholesale Customers)",
    x_wholesale_train,
    y_wholesale_train,
    param_name="C",
    param_range=param_range_svm_c,
)

plot_validation_curve(
    SVC(kernel="linear"),
    "Validation Curve (SVM, C parameter, Car Evaluation)",
    X_car_train,
    y_car_train,
    param_name="C",
    param_range=param_range_svm_c,
)

# Plot validation curves for SVM (gamma parameter, RBF kernel)
param_range_svm_gamma = np.logspace(-4, 1, 6)

plot_validation_curve(
    SVC(kernel="rbf"),
    "Validation Curve (SVM, gamma parameter, Wholesale Customers)",
    x_wholesale_train,
    y_wholesale_train,
    param_name="gamma",
    param_range=param_range_svm_gamma,
)

plot_validation_curve(
    SVC(kernel="rbf"),
    "Validation Curve (SVM, gamma parameter, Car Evaluation)",
    X_car_train,
    y_car_train,
    param_name="gamma",
    param_range=param_range_svm_gamma,
)

# Plot validation curves for k-NN (n_neighbors)
param_range_knn = np.arange(1, 11)

plot_validation_curve(
    KNeighborsClassifier(),
    "Validation Curve (k-NN, n_neighbors, Wholesale Customers)",
    x_wholesale_train,
    y_wholesale_train,
    param_name="n_neighbors",
    param_range=param_range_knn,
)

plot_validation_curve(
    KNeighborsClassifier(),
    "Validation Curve (k-NN, n_neighbors, Car Evaluation)",
    X_car_train,
    y_car_train,
    param_name="n_neighbors",
    param_range=param_range_knn,
)

# Classification reports and confusion matrices
models = {
    "NN, Wholesale Customers": best_nn_wholesale,
    "NN, Car Evaluation": best_nn_car,
    "SVM, Wholesale Customers": best_svm_wholesale,
    "SVM, Car Evaluation": best_svm_car,
    "k-NN, Wholesale Customers": best_knn_wholesale,
    "k-NN, Car Evaluation": best_knn_car,
    "AdaBoost, Wholesale Customers": best_ada_wholesale,
    "AdaBoost, Car Evaluation": best_ada_car,
}

datasets = {
    "Wholesale Customers": (x_wholesale_test, y_wholesale_test),
    "Car Evaluation": (X_car_test, y_car_test),
}

for model_name, model in models.items():
    dataset_name = model_name.split(", ")[1]
    X_test, y_test = datasets[dataset_name]
    predictions = model.predict(X_test)

    print(f"{model_name} Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=1))
    print(f"Confusion Matrix ({model_name}):\n", confusion_matrix(y_test, predictions))
