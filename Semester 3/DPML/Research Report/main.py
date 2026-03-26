"""
2D HP Protein Folding - Complete Integrated Solution
- CSP with Variables, Domains, Constraints
- Reinforcement Learning (Q-Learning)
- Fixed Greedy Algorithm
- Integrated Visualization
- Results saved to CSV and PNG with timestamps
"""

import numpy as np
import random
import time
import csv
from typing import List, Tuple, Dict, Set, Callable, Optional
from collections import defaultdict, deque
from datetime import datetime
import os

# ============================================================================
# GENERIC CSP FRAMEWORK
# ============================================================================

class Variable:
    """Represents a CSP variable with name and domain"""
    def __init__(self, name: str, domain: List):
        self.name = name
        self.domain = domain.copy()
        self.value = None

    def __repr__(self):
        return f"Var({self.name}, |D|={len(self.domain)})"


class Constraint:
    """Represents a constraint with variables and predicate function"""
    def __init__(self, variables: List[str], predicate: Callable):
        self.variables = variables
        self.predicate = predicate

    def is_satisfied(self, assignment: Dict[str, any]) -> bool:
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

    def add_variable(self, variable: Variable):
        self.variables[variable.name] = variable

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)

    def get_constraints_for_variable(self, var_name: str) -> List[Constraint]:
        return [c for c in self.constraints if var_name in c.variables]

    def is_consistent(self, var_name: str, value: any) -> bool:
        old_value = self.assignment.get(var_name)
        self.assignment[var_name] = value

        for constraint in self.get_constraints_for_variable(var_name):
            if not constraint.is_satisfied(self.assignment):
                if old_value is None:
                    del self.assignment[var_name]
                else:
                    self.assignment[var_name] = old_value
                return False

        if old_value is None:
            del self.assignment[var_name]
        else:
            self.assignment[var_name] = old_value

        return True


class CSPSolver:
    """CSP Solver with Backtracking, Forward Checking, and MRV"""

    def __init__(self, csp: CSP, max_time: float = 30):
        self.csp = csp
        self.max_time = max_time
        self.start_time = None
        self.nodes_explored = 0
        self.best_solution = None
        self.best_objective = float('inf')
        self.solutions_found = []
        self.objective_func = None

    def solve(self, objective_func: Callable = None) -> Optional[Dict[str, any]]:
        self.start_time = time.time()
        self.nodes_explored = 0
        self.best_solution = None
        self.best_objective = float('inf')
        self.solutions_found = []
        self.objective_func = objective_func

        print(f"  CSP Solver: {len(self.csp.variables)} variables, "
              f"{len(self.csp.constraints)} constraints")

        self._backtrack()

        print(f"  Nodes explored: {self.nodes_explored}")
        if self.best_solution:
            print(f"  Best objective: {self.best_objective}")

        return self.best_solution

    def _backtrack(self) -> bool:
        if time.time() - self.start_time > self.max_time:
            return False

        self.nodes_explored += 1

        if len(self.csp.assignment) == len(self.csp.variables):
            solution = self.csp.assignment.copy()

            if self.objective_func:
                obj_val = self.objective_func(solution)

                if obj_val < self.best_objective:
                    self.best_objective = obj_val
                    self.best_solution = solution
                    self.solutions_found.append((obj_val, time.time() - self.start_time))
                    print(f"    Solution: objective {obj_val}, "
                          f"nodes {self.nodes_explored}, "
                          f"time {time.time() - self.start_time:.1f}s")
            else:
                self.best_solution = solution

            return True

        var_name = self._select_variable_MRV()
        if var_name is None:
            return False

        for value in self.csp.variables[var_name].domain.copy():
            if self.csp.is_consistent(var_name, value):
                self.csp.assignment[var_name] = value
                removed = self._forward_check(var_name, value)

                self._backtrack()

                self._restore_domains(removed)
                del self.csp.assignment[var_name]

        return False

    def _select_variable_MRV(self) -> Optional[str]:
        unassigned = [v for v in self.csp.variables if v not in self.csp.assignment]
        if not unassigned:
            return None
        return min(unassigned, key=lambda v: len(self.csp.variables[v].domain))

    def _forward_check(self, var_name: str, value: any) -> Dict:
        removed = defaultdict(list)

        for constraint in self.csp.get_constraints_for_variable(var_name):
            for other_var in constraint.variables:
                if other_var != var_name and other_var not in self.csp.assignment:
                    to_remove = []
                    for other_value in self.csp.variables[other_var].domain:
                        old_assign = self.csp.assignment.copy()
                        self.csp.assignment[other_var] = other_value

                        if not constraint.is_satisfied(self.csp.assignment):
                            to_remove.append(other_value)

                        self.csp.assignment = old_assign

                    for val in to_remove:
                        if val in self.csp.variables[other_var].domain:
                            self.csp.variables[other_var].domain.remove(val)
                            removed[other_var].append(val)

        return removed

    def _restore_domains(self, removed: Dict):
        for var_name, values in removed.items():
            self.csp.variables[var_name].domain.extend(values)


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

        for i, (idx1, pos1) in enumerate(h_positions):
            for j in range(i + 1, len(h_positions)):
                idx2, pos2 = h_positions[j]
                if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1:
                    if abs(idx1 - idx2) > 1:
                        energy -= 1
        return energy

    def get_hh_contacts(self) -> List[Tuple[int, int]]:
        """Get list of H-H contact pairs"""
        contacts = []
        h_positions = [(i, pos) for i, pos in enumerate(self.positions)
                       if self.sequence[i] == 'H']

        for i, (idx1, pos1) in enumerate(h_positions):
            for j in range(i + 1, len(h_positions)):
                idx2, pos2 = h_positions[j]
                if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1:
                    if abs(idx1 - idx2) > 1:
                        contacts.append((idx1, idx2))
        return contacts


class HPProteinCSP:
    """HP Protein Folding as CSP"""

    MOVES = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    def __init__(self, protein: HPProtein):
        self.protein = protein
        self.csp = CSP()
        self._build_csp()

    def _build_csp(self):
        print(f"\n{'='*70}")
        print("CSP FORMALIZATION")
        print(f"{'='*70}")
        print(f"Sequence: {self.protein.sequence}")
        print(f"Length: {self.protein.length}")

        print(f"\nVARIABLES: X = {{x_0, x_1, ..., x_{self.protein.length-1}}}")

        var0 = Variable("x_0", [(0, 0)])
        self.csp.add_variable(var0)
        print(f"  x_0: D(x_0) = {{(0,0)}} [origin]")

        for i in range(1, self.protein.length):
            domain = self._generate_domain()
            var = Variable(f"x_{i}", domain)
            self.csp.add_variable(var)
        print(f"  x_1...x_{self.protein.length-1}: domains pruned by constraints")

        print(f"\nCONSTRAINTS:")
        self._add_constraints()

        print(f"Total: {len(self.csp.constraints)} constraints")
        print(f"{'='*70}\n")

    def _generate_domain(self) -> List[Tuple[int, int]]:
        n = self.protein.length
        return [(r, c) for r in range(-n, n+1) for c in range(-n, n+1)]

    def _add_constraints(self):
        # C1: Adjacency
        for i in range(self.protein.length - 1):
            self.csp.add_constraint(Constraint(
                [f"x_{i}", f"x_{i+1}"],
                lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1
            ))
        print(f"  C1: {self.protein.length-1} adjacency |x_i - x_{{i+1}}|_1 = 1")

        # C2: Non-overlap
        count = 0
        for i in range(self.protein.length):
            for j in range(i + 1, self.protein.length):
                self.csp.add_constraint(Constraint(
                    [f"x_{i}", f"x_{j}"],
                    lambda p1, p2: p1 != p2
                ))
                count += 1
        print(f"  C2: {count} non-overlap x_i ≠ x_j")

        # C3: Valid moves
        for i in range(self.protein.length - 1):
            self.csp.add_constraint(Constraint(
                [f"x_{i}", f"x_{i+1}"],
                lambda p1, p2: (p2[0] - p1[0], p2[1] - p1[1]) in self.MOVES
            ))
        print(f"  C3: {self.protein.length-1} valid moves ∈ {{U,D,L,R}}")

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
# REINFORCEMENT LEARNING
# ============================================================================

class ReinforcementLearningSolver:
    MOVES = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    def __init__(self, protein: HPProtein, episodes: int = 10000):
        self.protein = protein
        self.episodes = episodes
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.learning_rate = 0.1
        self.discount = 0.95
        self.epsilon = 0.1

    def solve(self) -> Conformation:
        print(f"  Training RL ({self.episodes} episodes)...")
        best_conf = None
        best_energy = float('inf')

        for ep in range(self.episodes):
            conf, _ = self._episode()
            if conf and len(conf.positions) == self.protein.length:
                energy = conf.calculate_energy()
                if energy < best_energy:
                    best_energy = energy
                    best_conf = conf
                    if ep % 1000 == 0 or energy <= -7:
                        print(f"    Episode {ep}: energy {energy}")
            if ep % 1000 == 0:
                self.epsilon *= 0.95

        print(f"  Best: {best_energy}")
        return best_conf or self._greedy_conf()

    def _episode(self):
        positions = [(0, 0)]
        used = {(0, 0)}
        reward_sum = 0

        for i in range(1, self.protein.length):
            state = self._state(positions, i)
            action = random.randint(0, 3) if random.random() < self.epsilon else np.argmax(self.q_table[state])

            new_pos = (positions[-1][0] + self.MOVES[action][0],
                       positions[-1][1] + self.MOVES[action][1])

            if new_pos in used:
                self.q_table[state][action] += self.learning_rate * (-10 - self.q_table[state][action])
                return None, reward_sum

            positions.append(new_pos)
            used.add(new_pos)
            reward = self._reward(positions, i)
            reward_sum += reward

            next_state = self._state(positions, i+1)
            self.q_table[state][action] += self.learning_rate * (
                    reward + self.discount * np.max(self.q_table[next_state]) - self.q_table[state][action]
            )

        return Conformation(positions, self.protein.sequence), reward_sum

    def _state(self, positions, index):
        if len(positions) < 3:
            return (index, 0, 0, 0)
        last = positions[-1]
        rel = []
        for i in range(min(3, len(positions))):
            p = positions[-(i+1)]
            rel.extend([p[0]-last[0], p[1]-last[1]])
        while len(rel) < 6:
            rel.append(0)
        return tuple([index] + rel[:6])

    def _reward(self, positions, index):
        r = 0
        if self.protein.is_hydrophobic(index):
            last = positions[-1]
            for i, p in enumerate(positions[:-1]):
                if self.protein.is_hydrophobic(i):
                    d = abs(last[0]-p[0]) + abs(last[1]-p[1])
                    if d == 1 and abs(i-index) > 1:
                        r += 5
                    elif d == 2:
                        r += 0.5
        xs, ys = [p[0] for p in positions], [p[1] for p in positions]
        r -= ((max(xs)-min(xs)) + (max(ys)-min(ys))) * 0.01
        return r

    def _greedy_conf(self):
        positions = [(0, 0)]
        used = {(0, 0)}
        for i in range(1, self.protein.length):
            state = self._state(positions, i)
            for action in np.argsort(self.q_table[state])[::-1]:
                new_pos = (positions[-1][0] + self.MOVES[action][0],
                           positions[-1][1] + self.MOVES[action][1])
                if new_pos not in used:
                    positions.append(new_pos)
                    used.add(new_pos)
                    break
        return Conformation(positions, self.protein.sequence)


# ============================================================================
# IMPROVED GREEDY WITH LOOKAHEAD
# ============================================================================

class ImprovedGreedySolver:
    """Improved greedy with 2-step lookahead and better heuristics"""

    MOVES = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    def __init__(self, protein: HPProtein, lookahead: int = 2):
        self.protein = protein
        self.lookahead = lookahead

    def solve(self) -> Conformation:
        best_conf = None
        best_energy = float('inf')

        # Try multiple random restarts
        for attempt in range(5):
            positions = [(0, 0)]
            used = {(0, 0)}

            for i in range(1, self.protein.length):
                # Get valid moves
                valid_moves = []
                for move in self.MOVES:
                    new_pos = (positions[-1][0] + move[0], positions[-1][1] + move[1])
                    if new_pos not in used:
                        valid_moves.append(new_pos)

                if not valid_moves:
                    break

                # Evaluate with lookahead
                best_pos = None
                best_score = float('-inf')

                for pos in valid_moves:
                    score = self._evaluate_with_lookahead(
                        positions + [pos],
                        used | {pos},
                        i,
                        self.lookahead
                    )

                    # Add randomness for diversity
                    score += random.uniform(-0.1, 0.1) if attempt > 0 else 0

                    if score > best_score:
                        best_score = score
                        best_pos = pos

                if best_pos:
                    positions.append(best_pos)
                    used.add(best_pos)

            if len(positions) == self.protein.length:
                conf = Conformation(positions, self.protein.sequence)
                energy = conf.calculate_energy()
                if energy < best_energy:
                    best_energy = energy
                    best_conf = conf

        return best_conf or Conformation([(0, i) for i in range(self.protein.length)],
                                         self.protein.sequence)

    def _evaluate_with_lookahead(self, positions, used, index, depth):
        if depth == 0 or index >= self.protein.length:
            return self._evaluate_position(positions[-1], positions, index)

        # Current position score
        score = self._evaluate_position(positions[-1], positions, index)

        # Look ahead
        if index < self.protein.length - 1:
            future_scores = []
            for move in self.MOVES:
                next_pos = (positions[-1][0] + move[0], positions[-1][1] + move[1])
                if next_pos not in used:
                    future_score = self._evaluate_with_lookahead(
                        positions + [next_pos],
                        used | {next_pos},
                        index + 1,
                        depth - 1
                    )
                    future_scores.append(future_score)

            if future_scores:
                score += 0.7 * max(future_scores)

        return score

    def _evaluate_position(self, pos, positions, current_idx):
        score = 0

        # H-H contact potential
        if current_idx < self.protein.length and self.protein.is_hydrophobic(current_idx):
            for i, placed_pos in enumerate(positions[:-1]):
                if self.protein.is_hydrophobic(i):
                    dist = abs(pos[0] - placed_pos[0]) + abs(pos[1] - placed_pos[1])
                    if dist == 1 and abs(i - current_idx) > 1:
                        score += 10  # Actual H-H contact
                    elif dist == 2:
                        score += 3  # Close to H
                    elif dist == 3:
                        score += 0.5

        # Compactness
        if len(positions) > 2:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            dist_to_center = abs(pos[0] - center_x) + abs(pos[1] - center_y)
            score -= dist_to_center * 0.3

        # Available neighbors (avoid dead ends)
        available = sum(1 for move in self.MOVES
                        if (pos[0] + move[0], pos[1] + move[1]) not in positions)
        score += available * 0.5

        return score


# ============================================================================
# VISUALIZATION WITH MATPLOTLIB
# ============================================================================

def setup_matplotlib():
    """Setup matplotlib for non-interactive backend"""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    return plt

plt = setup_matplotlib()


def visualize_conformation_plot(conf: Conformation, title: str, ax):
    """Visualize a conformation on matplotlib axis"""
    if not conf or not conf.positions:
        ax.text(0.5, 0.5, 'No solution', ha='center', va='center')
        ax.set_title(title)
        return

    positions = conf.positions
    sequence = conf.sequence

    # Extract coordinates
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    # Plot backbone
    ax.plot(xs, ys, 'k-', linewidth=2, alpha=0.3, zorder=1)

    # Plot amino acids
    for i, (x, y) in enumerate(positions):
        if sequence[i] == 'H':
            ax.scatter(x, y, c='red', s=300, edgecolors='darkred',
                       linewidths=2, zorder=3, label='H' if i == 0 else '')
        else:
            ax.scatter(x, y, c='lightblue', s=300, marker='s',
                       edgecolors='blue', linewidths=2, zorder=3,
                       label='P' if sequence[:i+1].count('P') == 1 else '')

        # Add index labels
        ax.text(x, y, str(i), ha='center', va='center', fontsize=8, zorder=4)

    # Plot H-H contacts
    contacts = conf.get_hh_contacts()
    for idx1, idx2 in contacts:
        pos1, pos2 = positions[idx1], positions[idx2]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                'g--', linewidth=2, alpha=0.6, zorder=2)

    # Formatting
    energy = conf.calculate_energy()
    ax.set_title(f'{title}\nEnergy: {energy} ({len(contacts)} H-H contacts)',
                 fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    # Set axis limits with padding
    margin = 2
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


def save_comparison_plot(csp_conf, rl_conf, greedy_conf, sequence, optimal, filename):
    """Save comparison plot of all three methods"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    visualize_conformation_plot(csp_conf, f'CSP (Gap: {abs(csp_conf.calculate_energy() - optimal)})', axes[0])
    visualize_conformation_plot(rl_conf, f'RL (Gap: {abs(rl_conf.calculate_energy() - optimal)})', axes[1])
    visualize_conformation_plot(greedy_conf, f'Greedy (Gap: {abs(greedy_conf.calculate_energy() - optimal)})', axes[2])

    fig.suptitle(f'Sequence: {sequence[:30]}... (Length: {len(sequence)}, Optimal: {optimal})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def save_summary_plots(results, timestamp):
    """Save summary bar charts"""
    sequences = [f"Seq{i+1}" for i in range(len(results))]

    # Energy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sequences))
    width = 0.2

    optimal = [r['optimal_energy'] for r in results]
    csp = [r['csp_energy'] for r in results]
    rl = [r['rl_energy'] for r in results]
    greedy = [r['greedy_energy'] for r in results]

    ax.bar(x - 1.5*width, optimal, width, label='Optimal', color='gold')
    ax.bar(x - 0.5*width, csp, width, label='CSP', color='blue')
    ax.bar(x + 0.5*width, rl, width, label='RL', color='green')
    ax.bar(x + 1.5*width, greedy, width, label='Greedy', color='red')

    ax.set_xlabel('Sequence')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Comparison Across Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(sequences)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    filename = f'energy_comparison_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

    # Gap comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    csp_gaps = [abs(r['csp_energy'] - r['optimal_energy']) for r in results]
    rl_gaps = [abs(r['rl_energy'] - r['optimal_energy']) for r in results]
    greedy_gaps = [abs(r['greedy_energy'] - r['optimal_energy']) for r in results]

    ax.bar(x - width, csp_gaps, width, label='CSP', color='blue')
    ax.bar(x, rl_gaps, width, label='RL', color='green')
    ax.bar(x + width, greedy_gaps, width, label='Greedy', color='red')

    ax.set_xlabel('Sequence')
    ax.set_ylabel('Gap from Optimal')
    ax.set_title('Optimality Gap Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(sequences)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    filename = f'gap_comparison_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# ============================================================================
# CSV EXPORT
# ============================================================================

def save_results_to_csv(results, timestamp):
    """Save results to CSV file"""
    filename = f'results_{timestamp}.csv'

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Sequence', 'Length', 'H_count', 'P_count', 'Optimal_Energy',
            'CSP_Energy', 'CSP_Gap', 'CSP_Time',
            'RL_Energy', 'RL_Gap', 'RL_Time',
            'Greedy_Energy', 'Greedy_Gap', 'Greedy_Time'
        ])

        # Data rows
        for r in results:
            writer.writerow([
                r['sequence'],
                r['length'],
                r['H_count'],
                r['P_count'],
                r['optimal_energy'],
                r['csp_energy'],
                abs(r['csp_energy'] - r['optimal_energy']),
                f"{r['csp_time']:.3f}",
                r['rl_energy'],
                abs(r['rl_energy'] - r['optimal_energy']),
                f"{r['rl_time']:.3f}",
                r['greedy_energy'],
                abs(r['greedy_energy'] - r['optimal_energy']),
                f"{r['greedy_time']:.4f}"
            ])

    print(f"\n  Saved: {filename}")


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(sequences: List[Tuple[int, str, int]], max_time: int = 30):
    """Run complete benchmark with visualization and CSV export"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("2D HP PROTEIN FOLDING - COMPLETE BENCHMARK")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)

    results = []

    for seq_idx, (length, seq_str, optimal_energy) in enumerate(sequences, 1):
        print(f"\n{'=' * 80}")
        print(f"SEQUENCE {seq_idx}: {seq_str}")
        print(f"Length: {length}, Optimal: {optimal_energy}")
        print(f"{'=' * 80}\n")

        protein = HPProtein(seq_str)
        result = {
            'sequence': seq_str,
            'length': length,
            'optimal_energy': optimal_energy,
            'H_count': seq_str.count('H'),
            'P_count': seq_str.count('P')
        }

        # 1. CSP
        print("--- CSP Solver ---")
        hp_csp = HPProteinCSP(protein)
        t0 = time.time()
        csp_conf = hp_csp.solve(max_time=max_time)
        csp_time = time.time() - t0

        if csp_conf:
            csp_energy = csp_conf.calculate_energy()
            result['csp_energy'] = csp_energy
            result['csp_time'] = csp_time
            gap = abs(csp_energy - optimal_energy)
            print(f"CSP: Energy {csp_energy} (gap {gap}), Time {csp_time:.2f}s\n")

        # 2. RL
        # print("--- Reinforcement Learning ---")
        # rl = ReinforcementLearningSolver(protein, 100000)
        # t0 = time.time()
        # rl_conf = rl.solve()
        # rl_time = time.time() - t0
        # rl_energy = rl_conf.calculate_energy()
        # result['rl_energy'] = rl_energy
        # result['rl_time'] = rl_time
        # gap = abs(rl_energy - optimal_energy)
        # print(f"RL: Energy {rl_energy} (gap {gap}), Time {rl_time:.2f}s\n")
        # 
        # # 3. Improved Greedy
        # print("--- Improved Greedy ---")
        # greedy = ImprovedGreedySolver(protein, lookahead=2)
        # t0 = time.time()
        # greedy_conf = greedy.solve()
        # greedy_time = time.time() - t0
        # greedy_energy = greedy_conf.calculate_energy()
        # result['greedy_energy'] = greedy_energy
        # result['greedy_time'] = greedy_time
        # gap = abs(greedy_energy - optimal_energy)
        # print(f"Greedy: Energy {greedy_energy} (gap {gap}), Time {greedy_time:.4f}s\n")
        # 
        # # Save comparison plot
        # plot_filename = f'comparison_seq{seq_idx}_{timestamp}.png'
        # save_comparison_plot(csp_conf, rl_conf, greedy_conf, seq_str, optimal_energy, plot_filename)

        results.append(result)

    # Save summary plots
    print("\n" + "=" * 80)
    print("Saving summary plots...")
    print("=" * 80)
    save_summary_plots(results, timestamp)

    # Save CSV
    print("\n" + "=" * 80)
    print("Saving results to CSV...")
    print("=" * 80)
    save_results_to_csv(results, timestamp)

    # Print summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Seq':<5} {'Len':<5} {'Opt':<5} {'CSP':<5} {'RL':<5} {'Greedy':<7}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['length']:<5} {r['optimal_energy']:<5} "
              f"{r['csp_energy']:<5} {r['rl_energy']:<5} {r['greedy_energy']:<7}")

    print("\n" + "=" * 80)
    print(f"All results saved with timestamp: {timestamp}")
    print("Files created:")
    print(f"  - results_{timestamp}.csv")
    print(f"  - energy_comparison_{timestamp}.png")
    print(f"  - gap_comparison_{timestamp}.png")
    for i in range(len(sequences)):
        print(f"  - comparison_seq{i+1}_{timestamp}.png")
    print("=" * 80 + "\n")

    return results, timestamp


if __name__ == "__main__":
    # Test sequences
    sequences = [
        (20, "HPHPPHHPHPPHPHHPPHPH", -9),
        (24, "HHPPHPPHPPHPPHPPHPPHPPHH", -9),
        (25, "PPHPPHHPPPPHHPPPPHHPPPPHH", -8),
        (36, "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP", -14),
        (48, "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH", -23),
        (50, "PPHPPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH", -21),
        (60, "PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHHPHHPHP", -36),
        (64, "HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHPPHPHHPHHHHHHHHHHHH", -42),
        (85, "HHHHPPPPHHHHHHHHHHHHPPPPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHPPHHPPHHPPHPH", -53),
        (100, "PPPHHPPHHHHPPHHHPHHPHHPHHHHPPPPPPPPHHHHHHPPHHHHHHPPPPPPPPPHPHHPHHHHHHHHHHHPPHHHPHHPHPPHPHHHPPPPPPHHH", -50)
    ]

    # Run with configurable time limit
    # Change max_time here to run with different time limits
    results, timestamp = run_benchmark(sequences, max_time=30)

    print("\n🎉 Benchmark complete!")
    print(f"Check files with timestamp {timestamp} for results")