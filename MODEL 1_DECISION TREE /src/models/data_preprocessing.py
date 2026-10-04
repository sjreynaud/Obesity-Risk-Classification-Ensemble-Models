# Step 7: Encode Categorical Variables and Prepare Features

combined = pd.concat([train.drop("NObeyesdad", axis=1), test])

categorical_cols = combined.select_dtypes(include=["object"]).columns

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    encoders[col] = le

X_train_full = combined.iloc[:len(train)]
X_test = combined.iloc[len(train):]

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(train["NObeyesdad"])

# Step 8: Split Data into Training and Validation SetsTrain/Test Split

X_train, X_valid, y_train, y_valid = train_test_split(
X_train_full,
y,
test_size=0.20,
random_state=42,
stratify=y
)
