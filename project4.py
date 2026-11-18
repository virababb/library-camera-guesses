import numpy as np

NAMES = ["person", "chair", "book"]
OUTPUT = np.array([
    [0.90, 0.90, 0.05, 0.05],
    [0.80, 0.10, 0.80, 0.10],
    [0.60, 0.20, 0.10, 0.70],
    [0.50, 0.60, 0.30, 0.10],
    [0.10, 0.90, 0.05, 0.05],
    [0.70, 0.30, 0.20, 0.90],
])
MIN_SCORE = 0.5

scores = OUTPUT[:, 0:1] * OUTPUT[:, 1:]

print("PART 1: Scores")
print("Shape:", scores.shape)
print("Guess 4, person:", round(float(scores[4, 0]), 2))

class_ids = scores.argmax(axis=1)
confidences = scores.max(axis=1)

print("\nPART 2: Best name")
for i in range(6):
    print("Guess", i, "->", NAMES[class_ids[i]], f"{confidences[i]:.2f}")

kept = np.where(confidences >= MIN_SCORE)[0]

print("\nPART 3: Kept")
print("Kept guesses:", [int(i) for i in kept])
print("Kept names:", [NAMES[class_ids[i]] for i in kept])

print("\nPART 4: Three thresholds")
for threshold in [0.2, 0.5, 0.8]:
    count = np.sum(confidences >= threshold)
    print(f"Threshold {threshold} -> {count} kept")