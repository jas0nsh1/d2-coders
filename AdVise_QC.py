import numpy as np
from sklearn.cluster import KMeans
import cv2
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit.primitives import Sampler
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.utils import algorithm_globals
from qiskit.compiler import transpile, assemble
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE

from qiskit.primitives import Estimator

ad_categories = ["Games", "Entertainment", "Art/Design", "Finance", "Travel"]

# 🔹 Function to Extract Dominant Colors from Ad Image
def extract_dominant_colors(image_path, num_colors=3):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"⚠️ Image '{image_path}' not found.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return np.array(colors) if len(colors) > 0 else np.array([[255, 255, 255]])  # Default to white

# 🔹 Function to Adjust Weights Based on Image Colors
def adjust_weights_by_color(colors):
    base_weights = {
        "Engagement Rate (ER)": [0.40, 0.20, 0.35, 0.10, 0.15],
        "Video Completion Rate (VCR)": [0.20, 0.40, 0.25, 0.05, 0.10],
        "Ad Click-Through Rate (CTR)": [0.30, 0.25, 0.30, 0.15, 0.20],
        "Cost Per Lead (CPL)": [0.05, 0.10, 0.05, 0.50, 0.25],
        "Return on Ad Spend (ROAS)": [0.05, 0.05, 0.05, 0.20, 0.30]
    }

    avg_color = np.mean(colors, axis=0)
    red, green, blue = avg_color

    if red > blue and red > green:
        print("🔴 Red-dominant Ad: Boosting 'Games' and 'Entertainment'")
        base_weights["Engagement Rate (ER)"][0] += 0.2
        base_weights["Video Completion Rate (VCR)"][1] += 0.2
    elif blue > red and blue > green:
        print("🔵 Blue-dominant Ad: Boosting 'Finance' and 'Travel'")
        base_weights["Cost Per Lead (CPL)"][3] += 0.2
        base_weights["Return on Ad Spend (ROAS)"][4] += 0.2
    elif green > red and green > blue:
        print("🟢 Green-dominant Ad: Boosting 'Art/Design'")
        base_weights["Ad Click-Through Rate (CTR)"][2] += 0.2

    return {key: np.array(value) / sum(value) for key, value in base_weights.items()}  

# 🎯 Load Image and Adjust Weights
image_path = "township-ad.jpg"
dominant_colors = extract_dominant_colors(image_path)
print("🎨 Dominant Colors (RGB):", dominant_colors)

weights_normalized = adjust_weights_by_color(dominant_colors)

# 🎯 Define Quadratic Program
qp = QuadraticProgram()
for category in ad_categories:
    qp.binary_var(category)

objective_coeffs = np.mean(list(weights_normalized.values()), axis=0)
qp.maximize(linear=objective_coeffs.tolist())

# ✅ Add Constraint: Allow Maximum 3 Categories
qp.linear_constraint(
    linear={category: 1 for category in ad_categories},
    sense="LE",
    rhs=3,
    name="category_limit"
)

# 🎯 Convert QP to QUBO for Quantum Computing
conv = QuadraticProgramToQubo()
qubo = conv.convert(qp)
operator, offset = qubo.to_ising()

# 🎯 Solve Using QAOA
sampler = Sampler()
optimizer = COBYLA(maxiter=100)
qaoa = QAOA(
    sampler=sampler,
    optimizer=optimizer,
    reps=3,
    initial_state=RealAmplitudes(num_qubits=operator.num_qubits)
)

optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qp)

# ✅ Extract Best Performing Ad Category
best_category = max(result.variables_dict, key=result.variables_dict.get)
print(f"\n🏆 Best Performing Ad Category: {best_category}")

# 🔹 Function to Evaluate Ad Performance (0-100 Scale)
def evaluate_ad_performance(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"⚠️ Image '{image_path}' not found.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    er_score = min(100, max(0, (brightness / 255) * 100))

    contrast = gray.std()
    ctr_score = min(100, max(0, (contrast / 128) * 100))

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges) / 255
    cpl_score = min(100, max(0, (1 - edge_density) * 100))

    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(pixels)
    unique_colors = len(np.unique(kmeans.labels_))
    vcr_score = min(100, max(0, (unique_colors / 10) * 100))

    roas_score = (er_score + ctr_score + vcr_score - cpl_score) / 3

    return {
        "Engagement Rate (ER)": round(er_score, 2),
        "Click-Through Rate (CTR)": round(ctr_score, 2),
        "Video Completion Rate (VCR)": round(vcr_score, 2),
        "Cost Per Lead (CPL)": round(cpl_score, 2),
        "Return on Ad Spend (ROAS)": round(roas_score, 2),
    }

# 🎯 Quantum Normalize Scores
def quantum_normalize(scores):
    normalized_scores = {}

    estimator = Estimator()
    optimizer = SPSA(maxiter=100)
    
    for metric, score in scores.items():
        cost_operator = SparsePauliOp.from_list([("Z", score / 100)])

        ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

        vqe = VQE(estimator, ansatz, optimizer=optimizer)
        result = vqe.compute_minimum_eigenvalue(cost_operator)
        
        optimal_score = abs(result.eigenvalue.real) * 10
        optimal_score = min(100, max(0, optimal_score))

        normalized_scores[metric] = round(optimal_score, 2)

    return normalized_scores

# ✅ Run Evaluation
ad_performance_scores = evaluate_ad_performance(image_path)
quantum_scores = quantum_normalize(ad_performance_scores)

# ✅ Print Final Evaluation
print(f"\n🎯 Final Quantum Ad Evaluation for {best_category}:\n")
for metric, score in quantum_scores.items():
    print(f" - {metric}: {score}/100")
