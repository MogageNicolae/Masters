"""
2D HP Protein Folding - Optimized Backofen 98 CSP Solution
- CSP with Lattice Parity & Pyramidal Domains (Backofen Formalization)
- Reinforcement Learning (Q-Learning)
- Fixed Greedy Algorithm
- Integrated Visualization
"""

import numpy as np
import random
import time
import csv
from typing import List, Tuple, Dict, Set, Callable, Optional
from collections import defaultdict
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

# ============================================================================
# GENERIC CSP FRAMEWORK (Optimized)
# ============================================================================

class Variable:
    """Represents a CSP variable with name and domain"""
    def __init__(self, name: str, domain: List):
        self.name = name
        self.domain = domain.copy()  # List of possible values (tuples)

    def __repr__(self):
        return f"Var({self.name}, |D|={len(self.domain)})"


class Constraint:
    """Represents a constraint with variables and predicate function"""
    def __init__(self, variables: List[str], predicate: Callable, name: str = ""):
        self.variables = variables
        self.predicate = predicate
        self.name = name

    def is_satisfied(self, assignment: Dict[str, any]) -> bool:
        # Only check if all involved variables are assigned
        if not all(var in assignment for var in self.variables):
            return True
        values = [assignment[var] for var in self.variables]
        return self.predicate(*values)


class CSP:
    """Generic Constraint Satisfaction Problem"""
    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.constraints: List[Constraint] = []
        self.assignment: Dict[str, any] = {}
        self.var_constraints = defaultdict(list) # Index for faster lookup

    def add_variable(self, variable: Variable):
        self.variables[variable.name] = variable

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)
        for var in constraint.variables:
            self.var_constraints[var].append(constraint)

    def is_consistent(self, var_name: str, value: any) -> bool:
        """Check if assigning value to var_name violates any constraints with ALREADY assigned variables"""
        self.assignment[var_name] = value

        valid = True
        for constraint in self.var_constraints[var_name]:
            # Only check constraints where all other variables are already assigned
            if all(v in self.assignment for v in constraint.variables):
                if not constraint.is_satisfied(self.assignment):
                    valid = False
                    break

        del self.assignment[var_name]
        return valid


class CSPSolver:
    """CSP Solver with MRV and Backtracking"""
    def __init__(self, csp: CSP, max_time: float = 30):
        self.csp = csp
        self.max_time = max_time
        self.start_time = None
        self.nodes_explored = 0
        self.best_solution = None
        self.best_objective = float('inf')

    def solve(self, objective_func: Callable = None) -> Optional[Dict[str, any]]:
        self.start_time = time.time()
        self.nodes_explored = 0
        self.best_solution = None
        self.best_objective = float('inf')
        self.objective_func = objective_func

        # Pre-sort variables by domain size to help MRV
        print(f"  CSP Solver: {len(self.csp.variables)} variables")

        if self._backtrack():
            return self.best_solution
        return self.best_solution

    def _backtrack(self) -> bool:
        if time.time() - self.start_time > self.max_time:
            return False

        self.nodes_explored += 1
        if self.nodes_explored % 10000 == 0:
            print(f"    Nodes: {self.nodes_explored}...")

        # 1. Check if assignment is complete
        if len(self.csp.assignment) == len(self.csp.variables):
            solution = self.csp.assignment.copy()
            if self.objective_func:
                obj_val = self.objective_func(solution)
                if obj_val < self.best_objective:
                    self.best_objective = obj_val
                    self.best_solution = solution
                    print(f"    New Best: {obj_val} (Time: {time.time()-self.start_time:.2f}s)")
                return False # Continue searching for better
            else:
                self.best_solution = solution
                return True # Found a valid one (SAT problem)

        # 2. MRV Heuristic: Select unassigned variable with fewest legal values
        var_name = self._select_variable_MRV()
        if not var_name:
            return False

        variable = self.csp.variables[var_name]

        # 3. LCV Heuristic (sort values to try to stay close to origin/compact)
        # Sort by distance to origin to encourage compactness
        sorted_domain = sorted(variable.domain, key=lambda p: abs(p[0])+abs(p[1]))

        for value in sorted_domain:
            if self.csp.is_consistent(var_name, value):
                self.csp.assignment[var_name] = value

                # Simple Forward Checking (Prune neighbors)
                # In full Backofen, we would prune domains of unassigned variables here.
                # For python speed, we rely on tight initial domains and MRV.

                if self._backtrack():
                    return True # Stop if we found a SAT solution (not OPT)

                del self.csp.assignment[var_name]

        return False

    def _select_variable_MRV(self) -> str:
        best_var = None
        min_len = float('inf')

        for name, var in self.csp.variables.items():
            if name not in self.csp.assignment:
                # Calculate valid domain size dynamically
                valid_count = 0
                for val in var.domain:
                    # Quick check against currently assigned neighbors
                    if self.csp.is_consistent(name, val):
                        valid_count += 1

                if valid_count < min_len:
                    min_len = valid_count
                    best_var = name
                if min_len == 0: return name # Fail fast

        return best_var

# ============================================================================
# HP PROTEIN CLASSES
# ============================================================================

class HPProtein:
    def __init__(self, sequence: str):
        self.sequence = sequence
        self.length = len(sequence)

    def is_hydrophobic(self, index: int) -> bool:
        return self.sequence[index] == 'H'

class Conformation:
    def __init__(self, positions: List[Tuple[int, int]], sequence: str):
        self.positions = positions
        self.sequence = sequence

    def calculate_energy(self) -> int:
        energy = 0
        h_positions = [(i, pos) for i, pos in enumerate(self.positions)
                       if self.sequence[i] == 'H']

        # H-H contacts (Manhattan dist = 1)
        for i, (idx1, pos1) in enumerate(h_positions):
            for j in range(i + 1, len(h_positions)):
                idx2, pos2 = h_positions[j]
                if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1:
                    if abs(idx1 - idx2) > 1: # Not sequential neighbors
                        energy -= 1
        return energy

    def get_hh_contacts(self) -> List[Tuple[int, int]]:
        contacts = []
        h_positions = [(i, pos) for i, pos in enumerate(self.positions) if self.sequence[i] == 'H']
        for i, (idx1, pos1) in enumerate(h_positions):
            for j in range(i + 1, len(h_positions)):
                idx2, pos2 = h_positions[j]
                if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1 and abs(idx1 - idx2) > 1:
                    contacts.append((idx1, idx2))
        return contacts

class HPProteinCSP:
    """
    Backofen 98 Formalization:
    - Variables: Coordinates (x, y)
    - Domains: Restricted by Parity and Distance from origin
    - Constraints: SAW, Connectivity, Symmetry Breaking
    """
    MOVES = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    def __init__(self, protein: HPProtein):
        self.protein = protein
        self.csp = CSP()
        self._build_csp()

    def _build_csp(self):
        print(f"\n{'='*70}")
        print("CSP FORMALIZATION (Backofen 98)")
        print(f"{'='*70}")
        print(f"Sequence: {self.protein.sequence}")

        # 1. Symmetry Breaking: Fix first two residues
        # x0 is origin
        var0 = Variable("x_0", [(0, 0)])
        self.csp.add_variable(var0)

        # x1 must be (1,0) - Removes rotations
        var1 = Variable("x_1", [(1, 0)])
        self.csp.add_variable(var1)

        print("  Symmetry Breaking: Fixed x_0=(0,0), x_1=(1,0)")

        # 2. Add Variables with Parity & Pyramidal Domains
        for i in range(2, self.protein.length):
            domain = self._generate_backofen_domain(i)
            var = Variable(f"x_{i}", domain)
            self.csp.add_variable(var)

        print(f"  Variables x_2..x_{self.protein.length-1} added with Parity Domains")

        # 3. Add Constraints
        self._add_constraints()

    def _generate_backofen_domain(self, index: int) -> List[Tuple[int, int]]:
        """
        Generate domain based on Backofen's Lattice Parity and Pyramid.
        1. Max distance from origin is 'index' (Pyramid).
        2. (x+y) must have same parity as 'index' (Checkerboard).
        """
        domain = []
        # Tighter bounds: cannot be further than index
        # Also, for compactness, we can heuristically bound it closer than N
        # But for correctness, we use index.
        max_dist = index

        for x in range(-max_dist, max_dist + 1):
            for y in range(-max_dist, max_dist + 1):
                # Distance constraint (Connectivity from origin)
                if abs(x) + abs(y) <= max_dist:
                    # Parity Constraint (Backofen's Key Pruner)
                    # On a grid, if you take 'i' steps, x+y must have parity of 'i'
                    if (x + y) % 2 == index % 2:
                        domain.append((x, y))
        return domain

    def _add_constraints(self):
        # C1: Adjacency (Chain Connectivity)
        for i in range(self.protein.length - 1):
            self.csp.add_constraint(Constraint(
                [f"x_{i}", f"x_{i+1}"],
                lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1,
                name=f"Adj_{i}_{i+1}"
            ))

        # C2: Self-Avoiding Walk (Global AllDiff broken down)
        # Note: In efficient solvers, we use a global AllDiff. 
        # Here we add binary constraints but rely on MRV to check them lazily.
        for i in range(self.protein.length):
            for j in range(i + 1, self.protein.length):
                # Optimization: Only add constraint if domains overlap
                # (handled dynamically by solver, but good to know)
                self.csp.add_constraint(Constraint(
                    [f"x_{i}", f"x_{j}"],
                    lambda p1, p2: p1 != p2,
                    name=f"Diff_{i}_{j}"
                ))

    def solve(self, max_time: float = 30) -> Optional[Conformation]:
        def objective(assignment: Dict[str, Tuple[int, int]]) -> int:
            positions = [assignment[f"x_{i}"] for i in range(self.protein.length)]
            return Conformation(positions, self.protein.sequence).calculate_energy()

        solver = CSPSolver(self.csp, max_time=max_time)
        best_assignment = solver.solve(objective_func=objective)

        if best_assignment:
            positions = [best_assignment[f"x_{i}"] for i in range(self.protein.length)]
            return Conformation(positions, self.protein.sequence)
        return None

# ============================================================================
# REINFORCEMENT LEARNING & GREEDY (Unchanged for Context)
# ============================================================================

class ReinforcementLearningSolver:
    MOVES = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    def __init__(self, protein: HPProtein, episodes: int = 5000): # Reduced for speed in demo
        self.protein = protein
        self.episodes = episodes
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.learning_rate = 0.1
        self.discount = 0.95
        self.epsilon = 0.1

    def solve(self) -> Conformation:
        best_conf = None
        best_energy = float('inf')
        for ep in range(self.episodes):
            conf, _ = self._episode()
            if conf and len(conf.positions) == self.protein.length:
                energy = conf.calculate_energy()
                if energy < best_energy:
                    best_energy = energy
                    best_conf = conf
            if ep % 500 == 0: self.epsilon *= 0.95
        return best_conf

    def _episode(self):
        positions = [(0, 0)]
        used = {(0, 0)}
        for i in range(1, self.protein.length):
            state = self._state(positions, i)
            if random.random() < self.epsilon: action = random.randint(0, 3)
            else: action = np.argmax(self.q_table[state])

            new_pos = (positions[-1][0] + self.MOVES[action][0], positions[-1][1] + self.MOVES[action][1])
            if new_pos in used:
                self.q_table[state][action] += self.learning_rate * (-10 - self.q_table[state][action])
                return None, 0

            positions.append(new_pos)
            used.add(new_pos)

            # Simple reward: H-H contact
            reward = 0
            if self.protein.is_hydrophobic(i):
                for j, p in enumerate(positions[:-1]):
                    if self.protein.is_hydrophobic(j) and abs(p[0]-new_pos[0])+abs(p[1]-new_pos[1])==1:
                        if abs(i-j)>1: reward += 1

            next_state = self._state(positions, i+1)
            self.q_table[state][action] += self.learning_rate * (reward + self.discount * np.max(self.q_table[next_state]) - self.q_table[state][action])

        return Conformation(positions, self.protein.sequence), 0

    def _state(self, positions, index):
        if len(positions) < 2: return (index, 0, 0)
        rel_x = positions[-1][0] - positions[-2][0]
        rel_y = positions[-1][1] - positions[-2][1]
        return (index, rel_x, rel_y)


class ImprovedGreedySolver:
    MOVES = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    def __init__(self, protein: HPProtein):
        self.protein = protein

    def solve(self) -> Conformation:
        best_conf = None
        best_energy = float('inf')
        for attempt in range(20): # Multiple restarts
            positions = [(0, 0)]
            used = {(0, 0)}

            for i in range(1, self.protein.length):
                candidates = []
                for move in self.MOVES:
                    new_pos = (positions[-1][0] + move[0], positions[-1][1] + move[1])
                    if new_pos not in used:
                        score = 0
                        # Heuristic: Neighbors
                        available_neighbors = 0
                        for m2 in self.MOVES:
                            if (new_pos[0]+m2[0], new_pos[1]+m2[1]) not in used: available_neighbors += 1
                        score += available_neighbors * 0.1

                        # Heuristic: H-contact
                        if self.protein.is_hydrophobic(i):
                            for j, p in enumerate(positions):
                                if self.protein.is_hydrophobic(j) and abs(p[0]-new_pos[0])+abs(p[1]-new_pos[1])==1:
                                    if abs(i-j) > 1: score += 5
                        candidates.append((score, new_pos))

                if not candidates: break
                candidates.sort(key=lambda x: x[0], reverse=True)
                # Softmax-like selection
                top = candidates[:2]
                selected = top[random.randint(0, len(top)-1)][1]
                positions.append(selected)
                used.add(selected)

            if len(positions) == self.protein.length:
                conf = Conformation(positions, self.protein.sequence)
                e = conf.calculate_energy()
                if e < best_energy:
                    best_energy = e
                    best_conf = conf
        return best_conf

# ============================================================================
# PLOTTING
# ============================================================================

def visualize_conformation(conf: Conformation, title: str, filename: str):
    if not conf: return
    plt.figure(figsize=(6, 6))
    xs, ys = zip(*conf.positions)
    plt.plot(xs, ys, 'k-', alpha=0.3, zorder=1)

    for i, (x, y) in enumerate(conf.positions):
        color = 'red' if conf.sequence[i] == 'H' else 'lightblue'
        plt.scatter(x, y, c=color, s=200, zorder=2, edgecolors='black')
        plt.text(x, y, str(i), ha='center', va='center', fontsize=8, zorder=3)

    contacts = conf.get_hh_contacts()
    for idx1, idx2 in contacts:
        p1, p2 = conf.positions[idx1], conf.positions[idx2]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', lw=2, alpha=0.7)

    plt.title(f"{title}\nEnergy: {conf.calculate_energy()}")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def run_benchmark(sequences, max_time=30):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for seq_idx, (length, seq_str, optimal) in enumerate(sequences, 1):
        print(f"\nSEQUENCE {seq_idx}: {seq_str} (L={length})")
        protein = HPProtein(seq_str)
        res = {'sequence': seq_str, 'length': length, 'optimal': optimal}

        # 1. Backofen CSP
        t0 = time.time()
        csp_solver = HPProteinCSP(protein)
        csp_conf = csp_solver.solve(max_time=max_time)
        res['csp_time'] = time.time() - t0
        res['csp_energy'] = csp_conf.calculate_energy() if csp_conf else 0
        print(f"  CSP: {res['csp_energy']} (Time: {res['csp_time']:.2f}s)")

        # 2. RL
        t0 = time.time()
        rl_solver = ReinforcementLearningSolver(protein)
        rl_conf = rl_solver.solve()
        res['rl_time'] = time.time() - t0
        res['rl_energy'] = rl_conf.calculate_energy() if rl_conf else 0
        print(f"  RL:  {res['rl_energy']} (Time: {res['rl_time']:.2f}s)")

        # 3. Greedy
        t0 = time.time()
        greedy_solver = ImprovedGreedySolver(protein)
        greedy_conf = greedy_solver.solve()
        res['greedy_time'] = time.time() - t0
        res['greedy_energy'] = greedy_conf.calculate_energy() if greedy_conf else 0
        print(f"  Grd: {res['greedy_energy']} (Time: {res['greedy_time']:.2f}s)")

        # Save Visuals
        if csp_conf: visualize_conformation(csp_conf, f"CSP: {seq_str[:10]}...", f"csp_{seq_idx}.png")
        if rl_conf: visualize_conformation(rl_conf, f"RL: {seq_str[:10]}...", f"rl_{seq_idx}.png")
        if greedy_conf: visualize_conformation(greedy_conf, f"Greedy: {seq_str[:10]}...", f"greedy_{seq_idx}.png")

        results.append(res)

    # Save Results CSV
    with open(f"results_{timestamp}.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to results_{timestamp}.csv")

if __name__ == "__main__":
    test_sequences = [
        (20, "HPHPPHHPHPPHPHHPPHPH", -9),
        (24, "HHPPHPPHPPHPPHPPHPPHPPHH", -9),
        # You can add larger ones here, but CSP is exponential!
        (25, "PPHPPHHPPPPHHPPPPHHPPPPHH", -8),
    ]
    run_benchmark(test_sequences, max_time=45)