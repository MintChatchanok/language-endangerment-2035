# Random Forest Classifier
# Train Classifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=endangerment_mapping.keys()))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=endangerment_mapping.keys(),
            yticklabels=endangerment_mapping.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# If the significant misclassification occurs
# Please do Hyperparameter Tuning with GridSearchCV

# XGBoost Classifier
# XGBoost Classifier 
xgb_clf = XGBClassifier(
    objective='multi:softmax',
    num_class=5,
    eval_metric='mlogloss',
    use_label_encoder=False,
    max_depth=8,
    n_estimators=150,
    learning_rate=0.1,
    random_state=42
)

xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=endangerment_mapping.keys()))

# XGBoost Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=endangerment_mapping.keys(),
            yticklabels=endangerment_mapping.keys())
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Neural Network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot Encode Target
y_categorical = to_categorical(y, num_classes=5)

# Train-test Split
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Compute Class Weights 
y_train_labels = np.argmax(y_train_nn, axis=1)  # convert back to single-label form for weighting

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)

class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)

# Build the Model
model = Sequential()
model.add(Dense(128, input_shape=(X_train_nn.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))  # 5-class classification

# Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train_nn,
    y_train_nn,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate the NN model
# Predict
y_pred_nn = model.predict(X_test_nn)
y_pred_labels = np.argmax(y_pred_nn, axis=1)
y_true_labels = np.argmax(y_test_nn, axis=1)

# Classification Report
print("Neural Network Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=endangerment_mapping.keys()))

# Confusion Matrix
cm_nn = confusion_matrix(y_true_labels, y_pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_nn, annot=True, fmt="d", cmap="Oranges",
            xticklabels=endangerment_mapping.keys(),
            yticklabels=endangerment_mapping.keys())
plt.title("Neural Network Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
