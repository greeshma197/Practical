sc
Practical 1
Aim: Write a program to implement logical gates AND, OR and NOT with McCulloch-Pitts.

Code :
def McCullochPitts(inputs, weights, threshold):

summation = sum([i * w for i, w in zip(inputs, weights)]) if summation >= threshold:
return 1 else:
return 0



print("McCulloch-Pitts Logic Gates")



# AND Gate print("\nAND Gate:")
for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:

print(f"Input: {x}, Output: {McCullochPitts(x, [1, 1], 2)}")



# OR Gate print("\nOR Gate:")
for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:

print(f"Input: {x}, Output: {McCullochPitts(x, [1, 1], 1)}")



# NOT Gate print("\nNOT Gate:")
 
for x in [0, 1]:

print(f"Input: {x}, Output: {McCullochPitts([x], [-1], 0)}")




Output :



 
Practical 2
Aim: Write a program to implement Hebb‟s rule.

Code :
import numpy as np
# Define the Hebbian Network class HebbNetwork:
def  init (self, input_size): # Initialize weights to zero
self.weights = np.zeros(input_size)


def train(self, X, Y): """
Train the network using Hebb's rule.


Parameters:
X: 2D numpy array of input patterns (bipolar values) Y: 1D numpy array of target outputs (bipolar values) """
for x, y in zip(X, Y):
self.weights += x * y # Apply He def train(self, X, Y):
"""
Train the network using Hebb's rule.


Parameters:
X: 2D numpy array of input patterns (bipolar values) Y: 1D numpy array of target outputs (bipolar values) """
for x, y in zip(X, Y):
 
self.weights += x * y # Apply Hebbian learning rule def predict(self, x):
"""
Predict the output for a given input using sign activation.


Parameters:
x: input pattern


Returns:
1 or -1 """
net_input = np.dot(self.weights, x) return 1 if net_input >= 0 else -1
class HebbNetwork:
def  init (self, input_size): self.weights = [0] * input_size self.bias = 0

def train(self, X, y):
# Hebbian learning: w = w + x*y for xi, yi in zip(X, y):
self.weights = [w + x * yi for w, x in zip(self.weights, xi)] self.bias += yi

def predict(self, x):
activation = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias return 1 if activation > 0 else 0
# Input patterns for 2-input OR gate (bipolar) X = np.array([
[-1, -1],  # 0 OR 0
 
[-1, 1],  # 0 OR 1
[ 1, -1],  # 1 OR 0
[ 1, 1]	# 1 OR 1
])


# Target outputs in bipolar form
Y = np.array([-1, 1, 1, 1]) # Output of OR gate in bipolar # Create and train the network
net = HebbNetwork(input_size=2) net.train(X, Y)

# Test the model
print("Testing OR Gate with Hebb's Rule:") for x, y in zip(X, Y):
pred = net.predict(x)
print(f"Input: {x}, Predicted: {pred}, Actual: {y}")


Output :

 
#AND Gate
import numpy as np


class HebbNetwork:
def  init (self, input_size): # Initialize weights to zero
self.weights = np.zeros(input_size)


def train(self, X, Y): """
Train the network using Hebbian learning rule X: input patterns (2D array)
Y: target outputs (1D array) """
for x, y in zip(X, Y):
self.weights += x * y # Hebb Rule: w = w + x * y


def predict(self, x): """
Predict output using sign activation function x: input pattern
"""
net_input = np.dot(self.weights, x) return 1 if net_input >= 0 else -1



# Bipolar representation for 2-input AND gate X = np.array([
[-1, -1], # 0 AND 0 = 0 → -1
[-1, 1], # 0 AND 1 = 0 → -1
 
[ 1, -1], # 1 AND 0 = 0 → -1
[ 1, 1]  # 1 AND 1 = 1 → 1
])


Y = np.array([-1, -1, -1, 1]) # Expected outputs in bipolar form


# Create and train the network
net = HebbNetwork(input_size=2) net.train(X, Y)

# Test the trained network
print("Testing AND Gate with Hebb's Rule:") for x, y in zip(X, Y):
prediction = net.predict(x)
print(f"Input: {x}, Predicted: {prediction}, Expected: {y}")


Output :

 
import numpy as np
import matplotlib.pyplot as plt


# Bipolar input patterns for AND gate X = np.array([
[-1, -1], # 0 AND 0 → -1
[-1, 1], # 0 AND 1 → -1
[ 1, -1], # 1 AND 0 → -1
[ 1, 1]  # 1 AND 1 → 1
])
import matplotlib.pyplot as plt
# Target outputs in bipolar format Y = np.array([-1, -1, -1, 1])

# Initialize weights weights = np.zeros(2)
weight_history = [weights.copy()] # Track weight updates # Hebb learning rule
for x, y in zip(X, Y): delta_w = x * y weights += delta_w
weight_history.append(weights.copy()) # Save updated weights


# Convert weight history to NumPy array for plotting weight_history = np.array(weight_history)
# Plotting the weight evolution plt.figure(figsize=(10, 6))
plt.plot(weight_history[:, 0], marker='o', label='Weight 1 (w1)')
plt.plot(weight_history[:, 1], marker='s', label='Weight 2 (w2)') plt.title('Weight Evolution During Hebbian Training (AND Gate)')
 
plt.xlabel('Training Step') plt.ylabel('Weight Value') plt.grid(True) plt.legend()
plt.xticks(ticks=range(len(weight_history)), labels=[f"Step {i}" for i in range(len(weight_history))])
plt.tight_layout() plt.show()
OUTPUT :

 
Practical 3
Aim : Implement Kohonen Self organizing map.

Code :
from minisom import MiniSom
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


wine = load_wine() x = wine.data
y = wine.target
label_names = wine.target_names


scaler = MinMaxScaler() x_scaled = scaler.fit_transform(x)

som = MiniSom(x=10, y=10, input_len=13, sigma=1.0, learning_rate=0.5) som.random_weights_init(x_scaled)
som.train_random(x_scaled, 200)


plt.figure(figsize=(10, 9)) plt.pcolor(som.distance_map().T, cmap='bone_r') plt.colorbar(label='distance')

colors = ['r', 'g', 'b']
label_color = [colors[label] for label in y] for i, x in enumerate(x_scaled):
 
w = som.winner(x)
plt.text(w[0] + 0.5, w[1] + 0.5, str(y[i]), color=label_color[i], fontdict={'weight': 'bold', 'size': 9})
legend = [Patch(color=c, label=label_names[i]) for i, c in enumerate(colors)] plt.legend(handles=legend, loc='upper right')
plt.title("Kohonen Self-Organizing Map (Wine Dataset)") plt.grid()
plt.show()
Output :

 
Practical 4
Aim : Solve the Hamming network given the exemplar vectors


Code :
import numpy as np
def hamming_network(input_vector, exemplar_vectors, max_iters=10): input_vector = np.array(input_vector)
exemplar_vectors = np.array(exemplar_vectors)

# Step 1: Calculate Layer 1 output (similarity score)
scores = np.dot(exemplar_vectors, input_vector.T) # Dot product scores += input_vector.size # Add bias (length of vector)

print("Initial similarity scores (Layer 1):", scores)

# Step 2: Apply competitive inhibition (Layer 2 / MaxNet) y = scores.copy()
epsilon = 0.1 # Inhibition factor n = len(y)

# Create inhibition matrix inhibition_matrix = np.full((n, n), -epsilon) np.fill_diagonal(inhibition_matrix, 1.0)

print("\nCompetitive layer iterations:") for i in range(max_iters):
y_new = np.dot(inhibition_matrix, y)
y_new = np.maximum(y_new, 0) # Clamp negatives to 0

print(f"Iteration {i+1}: {y_new}") if np.allclose(y, y_new):
print("Converged.") break
y = y_new
 
winner_index = np.argmax(y) return winner_index, y
# === Example === exemplars = [
[1, -1, 1, -1],	# X1
[-1, -1, 1, 1],	# X2
[1, 1, 1, 1]	# X3
]

input_vector = [1, -1, 1, 1] # Input to classify

winner, activations = hamming_network(input_vector, exemplars) print(f"\nInput vector is closest to exemplar X{winner + 1}")
Output :

 
Practical 5
Aim: Write a program for implementing BAM network.


Code :
import numpy as np

x = np.array([1, -1, 1])
y = np.array([1, -1]) w = np.outer(x, y)

def recall_bam(x):
y_out = np.sign(np.dot(x, w))
x_out = np.sign(np.dot(y_out, w.T)) return x_out, y_out

x_out, y_out = recall_bam(x) print("\nBAM Network") print("Recalled Y:", y_out) print("Recalled X:", x_out)

Output :

 
Practical 6
Aim: Implement a program to find the winning neuron using MaxNet

Code :

import numpy as np
def maxnet(activations, epsilon=0.1, max_iter=100): x = np.array(activations, dtype=float)
n = len(x)

for iteration in range(max_iter): x_new = np.zeros_like(x)
for i in range(n):
inhibitory_sum = np.sum(x) - x[i] x_new[i] = x[i] - epsilon * inhibitory_sum

# Apply ReLU to prevent negative values x_new = np.maximum(x_new, 0)

# Stop when only one neuron is active if np.count_nonzero(x_new) == 1:
break
x = x_new.copy()
winner_index = np.argmax(x) return winner_index, x
# Example usage
activations = [0.3, 0.9, 0.6, 0.2]
winner, final_output = maxnet(activations)

print("Final neuron activations:", final_output)
print(f"Winning neuron index: {winner} (activation: {final_output[winner]:.2f})")




 
Practical 7
Aim: Implement De-Morgan‟s Law.

Code :

# Define fuzzy set operations def fuzzy_union(A, B):
return {x: max(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_intersection(A, B):
return {x: min(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_complement(A): return {x: 1 - A[x] for x in A}
# De Morgan's Laws implementation def de_morgan_laws(A, B):
# Law 1: ~(A ∨ B) = ~A ∧ ~B
lhs1 = fuzzy_complement(fuzzy_union(A, B))
rhs1 = fuzzy_intersection(fuzzy_complement(A), fuzzy_complement(B))

# Law 2: ~(A ∧ B) = ~A ∨ ~B
lhs2 = fuzzy_complement(fuzzy_intersection(A, B))
rhs2 = fuzzy_union(fuzzy_complement(A), fuzzy_complement(B))

return lhs1, rhs1, lhs2, rhs2 # Example fuzzy sets
A = {'x1': 0.3, 'x2': 0.7, 'x3': 1.0}
B = {'x1': 0.5, 'x2': 0.4, 'x3': 0.8}
# Apply De Morgan's Laws
lhs1, rhs1, lhs2, rhs2 = de_morgan_laws(A, B)

# Display results
print("Law 1: ~(A ∨ B) = ~A ∧ ~B") print("LHS:", lhs1)
print("RHS:", rhs1)

print("\nLaw 2: ~(A ∧ B) = ~A ∨ ~B") print("LHS:", lhs2)
print("RHS:", rhs2)
 
Output :

 
Practical 8
Aim: Implement Union, Intersection, Complement and Difference operations on fuzzy sets.
Code :
# Fuzzy Set Operations
# Define fuzzy sets A and B

A = {1: 0.3, 2: 0.7, 3: 1.0, 4: 0.6}

B = {2: 0.8, 3: 0.4, 4: 0.9, 5: 1.0}



# 1. Union: μ_A∪B(x) = max(μ_A(x), μ_B(x)) def fuzzy_union(A, B):
result = {}

all_keys = set(A.keys()).union(B.keys()) for key in all_keys:
result[key] = max(A.get(key, 0), B.get(key, 0)) return result

# 2. Intersection: μ_A∩B(x) = min(μ_A(x), μ_B(x)) def fuzzy_intersection(A, B):
result = {}

all_keys = set(A.keys()).union(B.keys()) for key in all_keys:
result[key] = min(A.get(key, 0), B.get(key, 0)) return result
 
# 3. Complement: μ_A'(x) = 1 - μ_A(x) def fuzzy_complement(A):
result = {key: 1 - value for key, value in A.items()} return result

# 4. Difference: A - B = μ_A(x) - μ_A∩B(x) def fuzzy_difference(A, B):
intersection = fuzzy_intersection(A, B) result = {}
for key in A:

result[key] = A.get(key, 0) - intersection.get(key, 0) return result
print("Fuzzy Set A:", A) print("Fuzzy Set B:", B) print("\n Union (A ∪ B):") print(fuzzy_union(A, B)) print("\n Intersection (A ∩ B):") print(fuzzy_intersection(A, B)) print("\n Complement (~A):") print(fuzzy_complement(A)) print("\n Difference (A - B):") print(fuzzy_difference(A, B))
 
Output :

 
Practical 9
Aim: Create fuzzy relation by Cartesian product of any two fuzzy sets

Code :
# Fuzzy Relation via Cartesian Product
A = {1: 0.3, 2: 0.7, 3: 1.0}
B = {'a': 0.4, 'b': 0.8, 'c': 1.0}


def cartesian_product(A, B):
return {(x, y): min(A[x], B[y]) for x in A for y in B}


relation = cartesian_product(A, B) print("\nFuzzy Relation (A x B):") for k,v in relation.items():
print(f"{k}: {v}")


Output :

 
import pandas as pd
# Convert relation to matrix (DataFrame)
def relation_to_matrix(relation, A_keys, B_keys):
matrix = pd.DataFrame(index=A_keys, columns=B_keys) for (x, y), value in relation.items():
matrix.loc[x, y] = value return matrix.astype(float)
relation_matrix = relation_to_matrix(relation, list(A.keys()), list(B.keys())) print("\nFuzzy Relation Matrix:\n")
print(relation_matrix)


Output :

 
Practical 10
Aim: Perform max-min composition on any two fuzzy relations.

Code :
# Max-Min Composition import numpy as np import pandas as pd

# Define fuzzy relations as matrices # Relation R1: A → B
R1 = np.array([
[0.3, 0.7, 1.0],  # A1 to B1, B2, B3
[0.6, 0.4, 0.8]	# A2 to B1, B2, B3
])

# Relation R2: B → C R2 = np.array([
[0.9, 0.2],  # B1 to C1, C2
[0.5, 1.0],  # B2 to C1, C2
[0.4, 0.7]	# B3 to C1, C2
])

# Max-min composition
def max_min_composition(R1, R2): A_rows, B_cols = R1.shape B_rows, C_cols = R2.shape
assert B_cols == B_rows, "Incompatible matrix dimensions for composition" R3 = np.zeros((A_rows, C_cols))
for i in range(A_rows):	# Iterate over A for j in range(C_cols): # Iterate over C
min_vals = []
for k in range(B_cols): # Iterate over B min_vals.append(min(R1[i][k], R2[k][j]))
R3[i][j] = max(min_vals)

return R3
# Perform composition
R3 = max_min_composition(R1, R2)

# Display result
df = pd.DataFrame(R3, index=["A1", "A2"], columns=["C1", "C2"]) print("Max-Min Composition (A → C):")
print(df)

Output :
 
 
