import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from django.shortcuts import render

def knn(request):
    # Load the fruit dataset
    dataset = pd.read_csv('fruit_dataset.csv')

    X = dataset[['weight']]
    y = dataset['label']

    if request.method == 'POST':
        k_value = int(request.POST.get('k_value'))

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the KNN classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
        knn_classifier.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        predictions = knn_classifier.predict(X_test_scaled)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        context = {
            'k_value': k_value,
            'accuracy': accuracy * 100,
        }

        return render(request, 'result.html', context)

    return render(request, 'knn.html')
