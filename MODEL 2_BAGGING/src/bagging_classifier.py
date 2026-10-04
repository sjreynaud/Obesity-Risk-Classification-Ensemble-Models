# Step 9: Build and Train the Bagging Ensemble Model

bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

bag_model.fit(X_train, y_train)
