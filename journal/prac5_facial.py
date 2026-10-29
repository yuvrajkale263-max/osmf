import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 1. Load the Olivetti Faces dataset
faces = fetch_olivetti_faces()
X = faces.data       # Each face contains 4096 pixel values (64 x 64)
y = faces.target     # Person IDs: 0 to 39

print("Total images:", X.shape[0], "| People:", len(set(y)))

# Display sample faces
fig, ax = plt.subplots(1, 5, figsize=(10, 2.5))

for i in range(5):
    ax[i].imshow(faces.images[i * 10], cmap="gray")
    ax[i].set_title(f"Person {i}")
    ax[i].axis("off")

plt.show()

# 2. Separate authorized users and impostors
# Persons 0-4 are authorized; persons 5-39 are impostors
X_auth, y_auth = X[y < 5], y[y < 5]
X_imp = X[y >= 5]

# Split authorized faces into enrollment and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_auth,
    y_auth,
    test_size=0.3,
    stratify=y_auth,
    random_state=42
)

print(
    "Enrolled images:", len(X_train),
    "| Authorized test images:", len(X_test),
    "| Impostor images:", len(X_imp)
)

# 3. Reduce face dimensions using PCA
pca = PCA(n_components=20, random_state=42)

Xtr = pca.fit_transform(X_train)
Xte = pca.transform(X_test)
Ximp = pca.transform(X_imp)

# 4. Train the KNN face-identification model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Xtr, y_train)

print(
    "Identification accuracy on authorized users:",
    knn.score(Xte, y_test)
)

# 5. Calculate an authentication threshold
# The second neighbor is the closest OTHER enrolled face.
d_train, _ = knn.kneighbors(Xtr, n_neighbors=2)

threshold = np.percentile(d_train[:, 1], 95)
print("Threshold:", round(threshold, 2))

# 6. Calculate distances for authorized users and impostors
d_auth = knn.kneighbors(Xte, n_neighbors=1)[0][:, 0]
d_imp = knn.kneighbors(Ximp, n_neighbors=1)[0][:, 0]

# 7. Evaluate false rejection and false acceptance rates
print("\nThreshold | FRR (real user rejected) | FAR (impostor accepted)")

for m in [0.6, 0.8, 1.0, 1.2, 1.5]:
    t = threshold * m

    frr = np.mean(d_auth > t) * 100
    far = np.mean(d_imp <= t) * 100

    print(f"{t:8.2f} | {frr:24.1f}% | {far:23.1f}%")

# 8. Authenticate a face
def authenticate(face):
    p = pca.transform([face])
    dist = knn.kneighbors(p, n_neighbors=1)[0][0][0]

    if dist <= threshold:
        user_id = knn.predict(p)[0]
        return f"ACCEPTED (User {user_id})"

    return "REJECTED"


# 9. Display one authorized face and one impostor face
fig, ax = plt.subplots(1, 2, figsize=(7, 3))

examples = [
    ("Authorized", X_test[0]),
    ("Impostor", X_imp[0])
]

for a, (name, face) in zip(ax, examples):
    a.imshow(face.reshape(64, 64), cmap="gray")
    a.set_title(f"{name}\n{authenticate(face)}")
    a.axis("off")

plt.tight_layout()
plt.show()