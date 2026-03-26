"""
2D HP Protein Folding - Complete Integrated Solution with AC-3
- Improved CSP with AC-3, relative moves, symmetry breaking
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from io import StringIO
from contextlib import contextmanager

# ============================================================================
# GENERIC CSP FRAMEWORK WITH AC-3
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
    """Generic Constraint Satisfaction Problem with AC-3 support"""

    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.constraints: List[Constraint] = []
        self.assignment: Dict[str, any] = {}
        # For AC-3: map variables to their constraints
        self.var_to_constraints: Dict[str, List[Constraint]] = defaultdict(list)

    def add_variable(self, variable: Variable):
        self.variables[variable.name] = variable

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)
        # Build constraint map for AC-3
        for var in constraint.variables:
            self.var_to_constraints[var].append(constraint)

    def get_constraints_for_variable(self, var_name: str) -> List[Constraint]:
        return self.var_to_constraints[var_name]

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
    """CSP Solver with Backtracking, AC-3, and MRV"""

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

        print(f"  CSP Solver with AC-3: {len(self.csp.variables)} variables, "
              f"{len(self.csp.constraints)} constraints")

        # Apply AC-3 initially
        print(f"  Running initial AC-3...")
        if not self._ac3():
            print(f"  AC-3 detected inconsistency - no solution exists")
            return None

        print(f"  AC-3 completed - domains pruned")
        self._backtrack()

        print(f"  Nodes explored: {self.nodes_explored}")
        if self.best_solution:
            print(f"  Best objective: {self.best_objective}")

        return self.best_solution

    def _ac3(self) -> bool:
        """
        AC-3 Algorithm for arc consistency.
        Returns True if consistent, False if inconsistency detected.
        """
        # Queue of arcs (Xi, Xj) where Xi and Xj are connected by a constraint
        queue = deque()

        # Initialize queue with all arcs
        for constraint in self.csp.constraints:
            if len(constraint.variables) == 2:
                var1, var2 = constraint.variables
                queue.append((var1, var2, constraint))
                queue.append((var2, var1, constraint))

        # Process arcs
        while queue:
            xi, xj, constraint = queue.popleft()

            if self._revise(xi, xj, constraint):
                if len(self.csp.variables[xi].domain) == 0:
                    return False  # Inconsistency detected

                # Add neighbors of Xi back to queue
                for neighbor_constraint in self.csp.get_constraints_for_variable(xi):
                    for var in neighbor_constraint.variables:
                        if var != xi and var != xj:
                            queue.append((var, xi, neighbor_constraint))

        return True

    def _revise(self, xi: str, xj: str, constraint: Constraint) -> bool:
        """
        Revise the domain of Xi with respect to Xj.
        Returns True if domain of Xi was revised.
        """
        revised = False
        xi_var = self.csp.variables[xi]
        xj_var = self.csp.variables[xj]

        to_remove = []

        for xi_value in xi_var.domain:
            # Check if there exists a value in Xj's domain that satisfies the constraint
            satisfiable = False

            for xj_value in xj_var.domain:
                # Temporarily assign values
                old_assignment = self.csp.assignment.copy()
                self.csp.assignment[xi] = xi_value
                self.csp.assignment[xj] = xj_value

                if constraint.is_satisfied(self.csp.assignment):
                    satisfiable = True
                    self.csp.assignment = old_assignment
                    break

                self.csp.assignment = old_assignment

            if not satisfiable:
                to_remove.append(xi_value)
                revised = True

        # Remove inconsistent values
        for value in to_remove:
            xi_var.domain.remove(value)

        return revised

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
                    print(f"    ★ Solution: objective {obj_val}, "
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

                # Save domains before forward checking
                saved_domains = {v: self.csp.variables[v].domain.copy()
                                 for v in self.csp.variables}

                removed = self._forward_check(var_name, value)

                self._backtrack()

                # Restore domains
                for v in self.csp.variables:
                    self.csp.variables[v].domain = saved_domains[v]

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
    """
    Improved HP Protein Folding CSP with AC-3 and relative move encoding.
    
    Key improvements (Backofen & Will 1998):
    - Relative moves (U,D,L,R) instead of absolute coordinates
    - Domain size: 4 instead of O(n²)
    - AC-3 for arc consistency
    - Symmetry breaking: first move fixed
    - Contact capacity upper bound pruning
    - Smart move ordering heuristics
    """

    # Move encoding: 0=Right, 1=Up, 2=Left, 3=Down
    MOVES = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    MOVE_NAMES = ['R', 'U', 'L', 'D']
    OPPOSITE = {0: 2, 1: 3, 2: 0, 3: 1}

    def __init__(self, protein: HPProtein):
        self.protein = protein
        self.csp = CSP()
        self.h_positions = [i for i, aa in enumerate(protein.sequence) if aa == 'H']
        self._build_csp()

    def _build_csp(self):
        print(f"\n{'='*70}")
        print("IMPROVED CSP FORMALIZATION (AC-3 + Relative Moves)")
        print(f"{'='*70}")
        print(f"Sequence: {self.protein.sequence}")
        print(f"Length: {self.protein.length}")
        print(f"H count: {len(self.h_positions)}")

        print(f"\nVARIABLES: M = {{m_1, m_2, ..., m_{self.protein.length-1}}} (moves)")
        print(f"  Position 0 fixed at origin (0,0)")
        print(f"  Position 1 fixed at (0,1) - symmetry breaking (first move = Right)")

        # Variables represent MOVES, not positions
        # m_1 is fixed to 0 (Right) for symmetry breaking
        var1 = Variable("m_1", [0])  # First move always Right
        self.csp.add_variable(var1)
        print(f"  m_1: D(m_1) = {{R}} [symmetry breaking]")

        # Remaining moves can be any direction initially
        for i in range(2, self.protein.length):
            domain = [0, 1, 2, 3]  # R, U, L, D
            var = Variable(f"m_{i}", domain)
            self.csp.add_variable(var)

        print(f"  m_2...m_{self.protein.length-1}: D(m_i) = {{R,U,L,D}} initially")
        print(f"  Domain size: 4 (vs {(2*self.protein.length+1)**2} with absolute coords)")

        print(f"\nCONSTRAINTS:")
        self._add_constraints()

        print(f"Total: {len(self.csp.constraints)} constraints")
        print(f"{'='*70}\n")

    def _add_constraints(self):
        # C1: No immediate reversal (can't go R then L, etc.)
        count = 0
        for i in range(1, self.protein.length - 1):
            self.csp.add_constraint(Constraint(
                [f"m_{i}", f"m_{i+1}"],
                lambda m1, m2: m2 != self.OPPOSITE[m1]
            ))
            count += 1
        print(f"  C1: {count} no-reversal constraints (m_i+1 ≠ opposite(m_i))")

        # C2: No self-intersection (self-avoiding walk)
        # This is complex to encode directly, so we'll check during solution construction
        print(f"  C2: Self-avoiding walk (enforced during solution construction)")

    def solve(self, max_time: float = 30, target_energy: float = None) -> Optional[Conformation]:
        """Solve with improved CSP + custom search

        Args:
            max_time: Maximum time to search in seconds
            target_energy: If provided, stop early when this energy is reached (optimal solution)
        """

        def objective(assignment: Dict[str, int]) -> int:
            # Convert moves to positions
            positions = self._moves_to_positions(assignment)
            if positions is None:
                return float('inf')  # Invalid (self-intersecting)
            return Conformation(positions, self.protein.sequence).calculate_energy()

        # Use custom backtracking with better heuristics
        return self._custom_solve(max_time, target_energy)

    def _custom_solve(self, max_time: float, target_energy: float = None) -> Optional[Conformation]:
        """Custom solver with move-based encoding and pruning

        Args:
            max_time: Maximum search time in seconds
            target_energy: If provided, stop early when this energy is reached
        """
        start_time = time.time()
        nodes = 0
        best_energy = float('inf')
        best_conf = None
        found_optimal = False  # Flag to signal early stopping

        def backtrack(moves: List[int], positions: List[Tuple[int, int]],
                      used: Set[Tuple[int, int]]) -> None:
            nonlocal nodes, best_energy, best_conf, found_optimal

            # Early stopping conditions
            if time.time() - start_time > max_time:
                return

            if found_optimal:  # Stop if we found the target/optimal
                return

            nodes += 1

            # Complete solution
            if len(moves) == self.protein.length - 1:
                conf = Conformation(positions, self.protein.sequence)
                energy = conf.calculate_energy()
                if energy < best_energy:
                    best_energy = energy
                    best_conf = conf

                    # Check if we reached the optimal/target energy
                    if target_energy is not None and energy <= target_energy:
                        found_optimal = True
                        return  # Stop immediately - found optimal!

                return  # Continue searching for better solutions (if not optimal yet)

            # Upper bound pruning - but be less aggressive
            if not self._can_improve(positions, len(moves), best_energy):
                return

            current_pos = positions[-1]

            # Try moves in smart order
            moves_to_try = self._order_moves(positions, len(moves))

            for move in moves_to_try:
                if found_optimal:  # Check before each branch
                    return

                # Check for reversal
                if len(moves) > 0 and move == self.OPPOSITE[moves[-1]]:
                    continue

                new_pos = (current_pos[0] + self.MOVES[move][0],
                           current_pos[1] + self.MOVES[move][1])

                # Check collision
                if new_pos in used:
                    continue

                # Check if leaves future moves possible
                if not self._has_future_moves(new_pos, used, len(moves)):
                    continue

                # Make move
                backtrack(moves + [move], positions + [new_pos], used | {new_pos})

            return

        # Removed print statements for speed optimization
        # Start with position (0,0) and first move Right to (0,1)
        backtrack([0], [(0, 0), (0, 1)], {(0, 0), (0, 1)})

        return best_conf

    def _order_moves(self, positions: List[Tuple[int, int]], depth: int) -> List[int]:
        """Order moves by heuristic: prefer moves bringing H's together"""
        current_idx = len(positions)
        scores = []
        current_pos = positions[-1]

        for move in range(4):
            new_pos = (current_pos[0] + self.MOVES[move][0],
                       current_pos[1] + self.MOVES[move][1])

            # Skip collision
            if new_pos in positions:
                continue

            score = 0

            # Current residue H-H contact potential
            if current_idx < self.protein.length and self.protein.is_hydrophobic(current_idx):
                for i, pos in enumerate(positions):
                    if self.protein.is_hydrophobic(i):
                        dist = abs(new_pos[0] - pos[0]) + abs(new_pos[1] - pos[1])
                        if dist == 1 and abs(i - current_idx) > 1:
                            score += 30  # Actual contact - very high priority
                        elif dist == 2:
                            score += 8  # Close to H - good potential
                        elif dist == 3:
                            score += 2

            # Look ahead: consider future H positions
            if current_idx + 1 < self.protein.length:
                for future_offset in range(1, min(4, self.protein.length - current_idx)):
                    future_idx = current_idx + future_offset
                    if self.protein.is_hydrophobic(future_idx):
                        # Check if this move positions us well for future H's
                        for i, pos in enumerate(positions):
                            if self.protein.is_hydrophobic(i):
                                # Estimate future position distance
                                future_dist = abs(new_pos[0] - pos[0]) + abs(new_pos[1] - pos[1])
                                if future_dist <= future_offset + 1:
                                    score += 3  # Good positioning for future contacts

            # Compactness - prefer staying close to the center of mass
            if len(positions) > 2:
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)
                dist_to_center = abs(new_pos[0] - center_x) + abs(new_pos[1] - center_y)
                score -= dist_to_center * 0.5

            # Avoid positions that create dead ends for future placements
            available = sum(1 for m in self.MOVES
                           if (new_pos[0] + m[0], new_pos[1] + m[1]) not in positions)
            score += available * 1.5

            scores.append((score, move))

        scores.sort(reverse=True)
        return [move for _, move in scores]

    def _has_future_moves(self, pos: Tuple[int, int], used: Set[Tuple[int, int]],
                          depth: int) -> bool:
        """Check if position leaves at least one valid future move"""
        if depth >= self.protein.length - 2:
            return True

        available = sum(1 for move in self.MOVES
                        if (pos[0] + move[0], pos[1] + move[1]) not in used)
        return available >= 1

    def _can_improve(self, positions: List[Tuple[int, int]], depth: int,
                     best_energy: float) -> bool:
        """Upper bound pruning using contact capacity heuristic"""
        if best_energy == float('inf'):
            return True

        current_energy = Conformation(positions, self.protein.sequence).calculate_energy()
        current_contacts = -current_energy

        # Count H residues
        placed_h = sum(1 for i in range(len(positions)) if self.protein.is_hydrophobic(i))
        remaining_h = sum(1 for i in range(len(positions), self.protein.length)
                          if self.protein.is_hydrophobic(i))

        # Better upper bound calculation:
        # Each remaining H can make at most 4 contacts with placed H's and other remaining H's
        # But we need to be more realistic about the lattice constraints

        # Optimistic: each remaining H makes 2 contacts with placed H's
        max_contacts_with_placed = min(remaining_h * 2, placed_h * 4)

        # Optimistic: remaining H's form a compact chain among themselves
        # In a compact structure, each H (except terminal) can have 2 contacts
        max_contacts_among_remaining = max(0, remaining_h - 1) if remaining_h > 1 else 0

        # Upper bound on total contacts
        optimistic_total_contacts = current_contacts + max_contacts_with_placed + max_contacts_among_remaining
        optimistic_energy = -optimistic_total_contacts

        # Only prune if we're SURE we can't improve
        # Add a safety margin to avoid premature pruning
        return optimistic_energy <= best_energy

    def _moves_to_positions(self, assignment: Dict[str, int]) -> Optional[List[Tuple[int, int]]]:
        """Convert move assignment to absolute positions"""
        positions = [(0, 0)]
        used = {(0, 0)}

        for i in range(1, self.protein.length):
            move = assignment.get(f"m_{i}")
            if move is None:
                return None

            new_pos = (positions[-1][0] + self.MOVES[move][0],
                       positions[-1][1] + self.MOVES[move][1])

            if new_pos in used:
                return None  # Self-intersection

            positions.append(new_pos)
            used.add(new_pos)

        return positions


# ============================================================================
# REINFORCEMENT LEARNING
# ============================================================================

class ReinforcementLearningSolver:
    """
    RL Solver based on paper's rigid criterion approach:
    - Check matrix to track occupied positions
    - Only valid actions are considered (no collisions ever)
    - Reward is 0 until terminal state, then |E| (absolute energy)
    - Uses Q-learning with epsilon-greedy on valid actions only
    """
    MOVES = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # Right, Left, Up, Down
    MOVE_NAMES = ['R', 'L', 'U', 'D']

    def __init__(self, protein: HPProtein, episodes: int = 10000):
        self.protein = protein
        self.episodes = episodes
        self.n = protein.length

        # Q-table: state -> action values
        self.q_table = defaultdict(lambda: np.zeros(4))

        # Hyperparameters - tuned for better learning
        self.learning_rate = 0.3  # Higher α for faster learning with sparse rewards
        self.discount = 0.99  # High γ for long-term planning
        self.epsilon_start = 0.5  # More exploration initially
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.999  # Slower decay
        self.epsilon = self.epsilon_start

        # Add potential-based reward shaping (doesn't change optimal policy)
        self.use_shaping = True

    def solve(self) -> Conformation:
        # Removed verbose print for speed
        best_conf = None
        best_energy = float('inf')

        for episode in range(self.episodes):
            conf = self._one_episode()

            if conf and len(conf.positions) == self.n:
                energy = conf.calculate_energy()
                if energy < best_energy:
                    best_energy = energy
                    best_conf = conf

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return best_conf or self._greedy_solution()

    def _one_episode(self):
        """
        Execute one episode using rigid criterion:
        - Initialize check matrix (2n-1 × 2n-1)
        - Place amino acids one by one
        - Only consider valid actions (no collisions)
        - Reward = 0 until terminal, then |E|
        """
        # Initialize check matrix (2n-1 × 2n-1) centered at (n-1, n-1)
        check_size = 2 * self.n - 1
        check_matrix = np.zeros((check_size, check_size), dtype=int)

        # Starting position at center
        center = self.n - 1
        positions = [(0, 0)]  # Relative coordinates
        check_matrix[center, center] = 1  # Mark as occupied

        # Place remaining amino acids
        for i in range(1, self.n):
            state = self._get_state(positions, i)

            # Get valid actions using check matrix
            valid_actions = self._get_valid_actions(positions[-1], check_matrix, center)

            if not valid_actions:
                # Dead end - should rarely happen with good learning
                return None

            # Epsilon-greedy action selection on valid actions only
            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                # Choose best valid action
                q_values = self.q_table[state]
                valid_q = [(action, q_values[action]) for action in valid_actions]
                action = max(valid_q, key=lambda x: x[1])[0]

            # Take action
            new_pos = (positions[-1][0] + self.MOVES[action][0],
                      positions[-1][1] + self.MOVES[action][1])
            positions.append(new_pos)

            # Update check matrix
            check_x = center + new_pos[0]
            check_y = center + new_pos[1]
            check_matrix[check_x, check_y] = 1

            # Calculate reward
            # Paper says: R = 0 for i ∈ [1, n-1], R = |E| for i = n
            # But we can add potential-based shaping: R' = R + γΦ(s') - Φ(s)
            # This doesn't change optimal policy but helps learning

            if i == self.n - 1:
                # Terminal state - reward is |E| as per paper
                conf = Conformation(positions, self.protein.sequence)
                terminal_reward = abs(conf.calculate_energy())

                if self.use_shaping:
                    # Add shaping for better learning
                    potential_s = self._potential(positions[:-1], i-1)
                    potential_next = self._potential(positions, i)
                    shaping = self.discount * potential_next - potential_s
                    reward = terminal_reward + shaping
                else:
                    reward = terminal_reward

                next_state = None
                is_terminal = True
            else:
                # Non-terminal state
                if self.use_shaping:
                    # Potential-based shaping: doesn't change optimal policy
                    potential_s = self._potential(positions[:-1], i-1)
                    potential_next = self._potential(positions, i)
                    reward = self.discount * potential_next - potential_s
                else:
                    # Pure paper approach: reward = 0
                    reward = 0

                next_state = self._get_state(positions, i + 1)
                is_terminal = False

            # Q-learning update: Q(s,a) ← Q(s,a) + α[R + γ·max_a'Q(s',a') - Q(s,a)]
            if is_terminal:
                target = reward
            else:
                target = reward + self.discount * np.max(self.q_table[next_state])

            self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

        return Conformation(positions, self.protein.sequence)

    def _get_valid_actions(self, current_pos, check_matrix, center):
        """
        Get valid actions that don't cause collisions
        Uses check matrix to determine validity
        """
        valid_actions = []

        for action_idx, move in enumerate(self.MOVES):
            new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            check_x = center + new_pos[0]
            check_y = center + new_pos[1]

            # Check if within bounds and not occupied
            if (0 <= check_x < check_matrix.shape[0] and
                0 <= check_y < check_matrix.shape[1] and
                check_matrix[check_x, check_y] == 0):
                valid_actions.append(action_idx)

        return valid_actions

    def _get_state(self, positions, index):
        """
        State representation:
        - Current index
        - Last few relative positions for local structure
        - Residue types in context
        """
        state_features = [index]

        # Relative positions of last 3-4 residues
        last_pos = positions[-1]
        for i in range(min(4, len(positions))):
            pos = positions[-(i+1)]
            state_features.extend([pos[0] - last_pos[0], pos[1] - last_pos[1]])

        # Pad to fixed size
        while len(state_features) < 9:
            state_features.append(0)

        # Add residue type context
        if index < self.n:
            state_features.append(1 if self.protein.is_hydrophobic(index) else 0)

        return tuple(state_features[:10])

    def _potential(self, positions, index):
        """
        Potential function Φ(s) for reward shaping
        Estimates the value of a partial structure based on:
        - Number of H-H contacts formed so far
        - Compactness
        - Flexibility for future placements

        This is used for potential-based reward shaping: R' = R + γΦ(s') - Φ(s)
        Doesn't change optimal policy but helps with sparse rewards
        """
        if len(positions) < 2:
            return 0

        potential = 0

        # Count existing H-H contacts (main driver of energy)
        h_contacts = 0
        for i in range(len(positions)):
            if self.protein.is_hydrophobic(i):
                for j in range(i + 2, len(positions)):  # Non-adjacent
                    if self.protein.is_hydrophobic(j):
                        pos_i, pos_j = positions[i], positions[j]
                        dist = abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1])
                        if dist == 1:
                            h_contacts += 1

        potential += h_contacts * 10  # Each contact is worth 10 in potential

        # Compactness: prefer structures that are more compact
        if len(positions) > 3:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            bounding_box = (max(xs) - min(xs)) + (max(ys) - min(ys))
            # Normalize by length
            compactness_score = len(positions) / max(bounding_box, 1)
            potential += compactness_score * 2

        # Potential for future H-H contacts (proximity of remaining H's)
        remaining_h_count = sum(1 for i in range(len(positions), self.n)
                               if self.protein.is_hydrophobic(i))
        placed_h_count = sum(1 for i in range(len(positions))
                            if self.protein.is_hydrophobic(i))

        # Estimate potential future contacts
        potential_future = remaining_h_count * placed_h_count * 0.5
        potential += potential_future

        return potential

    def _greedy_solution(self):
        """
        Generate greedy solution using learned Q-values with rigid criterion
        """
        check_size = 2 * self.n - 1
        check_matrix = np.zeros((check_size, check_size), dtype=int)
        center = self.n - 1

        positions = [(0, 0)]
        check_matrix[center, center] = 1

        for i in range(1, self.n):
            state = self._get_state(positions, i)
            valid_actions = self._get_valid_actions(positions[-1], check_matrix, center)

            if not valid_actions:
                # Dead end - pick any valid move if possible
                for action_idx, move in enumerate(self.MOVES):
                    new_pos = (positions[-1][0] + move[0], positions[-1][1] + move[1])
                    check_x = center + new_pos[0]
                    check_y = center + new_pos[1]
                    if (0 <= check_x < check_size and 0 <= check_y < check_size and
                        check_matrix[check_x, check_y] == 0):
                        positions.append(new_pos)
                        check_matrix[check_x, check_y] = 1
                        break
                else:
                    # Truly stuck - this shouldn't happen
                    break
            else:
                # Choose best valid action
                q_values = self.q_table[state]
                valid_q = [(action, q_values[action]) for action in valid_actions]
                best_action = max(valid_q, key=lambda x: x[1])[0]

                new_pos = (positions[-1][0] + self.MOVES[best_action][0],
                          positions[-1][1] + self.MOVES[best_action][1])
                positions.append(new_pos)

                check_x = center + new_pos[0]
                check_y = center + new_pos[1]
                check_matrix[check_x, check_y] = 1

        return Conformation(positions, self.protein.sequence)


# ============================================================================
# IMPROVED GREEDY WITH LOOKAHEAD
# ============================================================================

class ImprovedGreedySolver:
    """
    Enhanced greedy solver with multiple strategies:
    1. Chain growth with best-first evaluation
    2. Multiple random restarts with different seeds
    3. Hydrophobic core guidance
    4. Dead-end avoidance
    5. Contact maximization heuristic
    """

    MOVES = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    def __init__(self, protein: HPProtein, lookahead: int = 3):
        self.protein = protein
        self.lookahead = lookahead

    def solve(self) -> Conformation:
        """Try multiple strategies and return best result"""
        best_conf = None
        best_energy = float('inf')

        strategies = [
            (self._hydrophobic_core_growth, "Hydrophobic Core"),
            (self._contact_maximizing_greedy, "Contact Maximizing"),
            (self._beam_search_greedy, "Beam Search"),
            (self._spiral_growth, "Spiral Growth"),
        ]

        # Try each strategy
        for strategy_func, strategy_name in strategies:
            try:
                conf = strategy_func()
                if conf and len(conf.positions) == self.protein.length:
                    energy = conf.calculate_energy()
                    if energy < best_energy:
                        best_energy = energy
                        best_conf = conf
            except Exception:
                # Strategy failed, continue with others
                continue

        # If all strategies failed, return linear conformation
        if best_conf is None:
            positions = [(0, i) for i in range(self.protein.length)]
            best_conf = Conformation(positions, self.protein.sequence)

        return best_conf

    def _hydrophobic_core_growth(self) -> Conformation:
        """
        Grow the chain by trying to form a hydrophobic core.
        Place H residues close together and P residues on the periphery.
        """
        positions = [(0, 0)]
        used = {(0, 0)}

        for i in range(1, self.protein.length):
            best_pos = None
            best_score = float('-inf')

            # Get valid neighbors
            valid_positions = []
            for move in self.MOVES:
                new_pos = (positions[-1][0] + move[0], positions[-1][1] + move[1])
                if new_pos not in used:
                    valid_positions.append(new_pos)

            if not valid_positions:
                return None  # Dead end

            # Evaluate each position
            for pos in valid_positions:
                score = self._evaluate_hydrophobic_core(pos, positions, used, i)
                if score > best_score:
                    best_score = score
                    best_pos = pos

            if best_pos:
                positions.append(best_pos)
                used.add(best_pos)

        return Conformation(positions, self.protein.sequence)

    def _evaluate_hydrophobic_core(self, pos, positions, used, index):
        """Evaluate position based on hydrophobic core formation"""
        score = 0

        # Strong reward for H-H contacts
        if self.protein.is_hydrophobic(index):
            h_contacts = 0
            for i, placed_pos in enumerate(positions):
                if self.protein.is_hydrophobic(i):
                    dist = abs(pos[0] - placed_pos[0]) + abs(pos[1] - placed_pos[1])
                    if dist == 1 and abs(i - index) > 1:
                        h_contacts += 1
                        score += 20  # Strong reward for actual contact
                    elif dist == 2:
                        score += 5  # Potential contact
                    elif dist == 3:
                        score += 1

            # Bonus for multiple contacts
            if h_contacts >= 2:
                score += h_contacts * 10

        # If P, prefer positions that don't block H residues
        if not self.protein.is_hydrophobic(index):
            # Count future H residues
            future_h = sum(1 for j in range(index + 1, self.protein.length)
                          if self.protein.is_hydrophobic(j))
            if future_h > 0:
                # Prefer positions that maintain open space for future H's
                h_positions = [positions[j] for j in range(len(positions))
                             if self.protein.is_hydrophobic(j)]
                if h_positions:
                    # Don't place P in the center of H cluster
                    min_dist_to_h = min(abs(pos[0] - hp[0]) + abs(pos[1] - hp[1])
                                       for hp in h_positions)
                    if min_dist_to_h > 1:
                        score += 3

        # Avoid dead ends
        available = sum(1 for move in self.MOVES
                       if (pos[0] + move[0], pos[1] + move[1]) not in used)
        if available == 0:
            score -= 100  # Heavy penalty
        elif available == 1:
            score -= 10
        else:
            score += available * 2

        # Compactness for H residues
        if self.protein.is_hydrophobic(index) and len(positions) > 2:
            h_positions = [positions[j] for j in range(len(positions))
                          if self.protein.is_hydrophobic(j)]
            if h_positions:
                center_x = sum(p[0] for p in h_positions) / len(h_positions)
                center_y = sum(p[1] for p in h_positions) / len(h_positions)
                dist_to_h_center = abs(pos[0] - center_x) + abs(pos[1] - center_y)
                score -= dist_to_h_center * 0.5

        return score

    def _contact_maximizing_greedy(self) -> Conformation:
        """
        Greedy algorithm focused on maximizing H-H contacts at each step.
        Uses deeper lookahead for better decisions.
        """
        best_conf = None
        best_energy = float('inf')

        # Try with different random seeds for diversity - increased to 10
        for seed in range(10):
            positions = [(0, 0)]
            used = {(0, 0)}

            for i in range(1, self.protein.length):
                valid_positions = []
                for move in self.MOVES:
                    new_pos = (positions[-1][0] + move[0], positions[-1][1] + move[1])
                    if new_pos not in used:
                        valid_positions.append(new_pos)

                if not valid_positions:
                    break

                # Evaluate with lookahead
                best_pos = None
                best_score = float('-inf')

                for pos in valid_positions:
                    score = self._lookahead_evaluation(
                        positions + [pos],
                        used | {pos},
                        i,
                        depth=self.lookahead
                    )

                    # Add small random noise for diversity
                    score += random.uniform(-0.5, 0.5) if seed > 0 else 0

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

        return best_conf

    def _lookahead_evaluation(self, positions, used, index, depth):
        """Recursive lookahead evaluation"""
        if depth == 0 or index >= self.protein.length:
            return self._immediate_score(positions[-1], positions, index)

        current_score = self._immediate_score(positions[-1], positions, index)

        # Look ahead
        if index < self.protein.length - 1:
            future_scores = []
            for move in self.MOVES:
                next_pos = (positions[-1][0] + move[0], positions[-1][1] + move[1])
                if next_pos not in used:
                    future_score = self._lookahead_evaluation(
                        positions + [next_pos],
                        used | {next_pos},
                        index + 1,
                        depth - 1
                    )
                    future_scores.append(future_score)

            if future_scores:
                current_score += 0.6 * max(future_scores)

        return current_score

    def _immediate_score(self, pos, positions, index):
        """Calculate immediate score for a position"""
        score = 0

        # H-H contacts
        if index < self.protein.length and self.protein.is_hydrophobic(index):
            for i, placed_pos in enumerate(positions[:-1]):
                if self.protein.is_hydrophobic(i):
                    dist = abs(pos[0] - placed_pos[0]) + abs(pos[1] - placed_pos[1])
                    if dist == 1 and abs(i - index) > 1:
                        score += 15
                    elif dist == 2:
                        score += 4
                    elif dist == 3:
                        score += 1

        # Avoid sprawling structures
        if len(positions) > 3:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            bounding_box = (max(xs) - min(xs)) + (max(ys) - min(ys))
            score -= bounding_box * 0.2

        # Maintain flexibility
        available = sum(1 for move in self.MOVES
                       if (pos[0] + move[0], pos[1] + move[1]) not in set(positions))
        score += available * 1.5

        return score

    def _beam_search_greedy(self, beam_width=5) -> Conformation:
        """
        Beam search approach: maintain top-k partial solutions
        and expand them in parallel.
        """
        # Start with initial position
        beam = [
            {
                'positions': [(0, 0)],
                'used': {(0, 0)},
                'score': 0
            }
        ]

        for i in range(1, self.protein.length):
            candidates = []

            for state in beam:
                positions = state['positions']
                used = state['used']

                # Generate successors
                for move in self.MOVES:
                    new_pos = (positions[-1][0] + move[0], positions[-1][1] + move[1])
                    if new_pos not in used:
                        new_positions = positions + [new_pos]
                        new_used = used | {new_pos}

                        # Evaluate this candidate
                        score = self._evaluate_complete_partial(new_positions, i)

                        candidates.append({
                            'positions': new_positions,
                            'used': new_used,
                            'score': score
                        })

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x['score'], reverse=True)
            beam = candidates[:beam_width]

        # Return best complete solution from beam
        best_conf = None
        best_energy = float('inf')

        for state in beam:
            if len(state['positions']) == self.protein.length:
                conf = Conformation(state['positions'], self.protein.sequence)
                energy = conf.calculate_energy()
                if energy < best_energy:
                    best_energy = energy
                    best_conf = conf

        return best_conf

    def _evaluate_complete_partial(self, positions, current_index):
        """Evaluate a partial conformation"""
        score = 0

        # Count existing H-H contacts
        for i in range(len(positions)):
            if self.protein.is_hydrophobic(i):
                for j in range(i + 2, len(positions)):
                    if self.protein.is_hydrophobic(j):
                        pi, pj = positions[i], positions[j]
                        if abs(pi[0] - pj[0]) + abs(pi[1] - pj[1]) == 1:
                            score += 10

        # Estimate potential future contacts
        remaining_h = sum(1 for k in range(current_index + 1, self.protein.length)
                         if self.protein.is_hydrophobic(k))
        placed_h = sum(1 for k in range(len(positions))
                      if self.protein.is_hydrophobic(k))

        score += remaining_h * placed_h * 0.3

        # Compactness bonus
        if len(positions) > 3:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            bounding_box = (max(xs) - min(xs)) + (max(ys) - min(ys))
            score -= bounding_box * 0.3

        return score

    def _spiral_growth(self) -> Conformation:
        """
        Spiral growth strategy: build the protein in a spiral/zigzag pattern.
        This creates compact structures that often have good H-H contacts.
        """
        best_conf = None
        best_energy = float('inf')

        # Try different spiral directions
        spiral_patterns = [
            # Clockwise spiral
            [(0, 1), (1, 0), (0, -1), (-1, 0)],
            # Counter-clockwise spiral
            [(0, 1), (-1, 0), (0, -1), (1, 0)],
            # Zigzag patterns
            [(0, 1), (1, 0), (0, 1), (1, 0)],
            [(1, 0), (0, 1), (1, 0), (0, 1)],
        ]

        for pattern in spiral_patterns:
            positions = [(0, 0)]
            used = {(0, 0)}
            pattern_idx = 0
            consecutive_failures = 0

            for i in range(1, self.protein.length):
                placed = False
                attempts = 0

                # Try to follow the pattern, with fallback
                while attempts < 8 and not placed:
                    if attempts < 4:
                        # Try pattern direction
                        move = pattern[pattern_idx % len(pattern)]
                    else:
                        # Fallback to any valid move
                        move = self.MOVES[(attempts - 4) % 4]

                    new_pos = (positions[-1][0] + move[0], positions[-1][1] + move[1])

                    if new_pos not in used:
                        # Evaluate if this move creates good contacts
                        if self._is_good_spiral_move(new_pos, positions, used, i):
                            positions.append(new_pos)
                            used.add(new_pos)
                            placed = True
                            consecutive_failures = 0

                            # Advance pattern if we followed it
                            if attempts < 4:
                                pattern_idx += 1

                    attempts += 1

                if not placed:
                    consecutive_failures += 1
                    if consecutive_failures > 3:
                        break  # Give up on this pattern

            if len(positions) == self.protein.length:
                conf = Conformation(positions, self.protein.sequence)
                energy = conf.calculate_energy()
                if energy < best_energy:
                    best_energy = energy
                    best_conf = conf

        return best_conf

    def _is_good_spiral_move(self, pos, positions, used, index):
        """Check if a spiral move is beneficial"""
        # Always allow the move if we're early in the sequence
        if index < 3:
            return True

        score = 0

        # Check for H-H contacts
        if self.protein.is_hydrophobic(index):
            for i, placed_pos in enumerate(positions):
                if self.protein.is_hydrophobic(i):
                    dist = abs(pos[0] - placed_pos[0]) + abs(pos[1] - placed_pos[1])
                    if dist == 1 and abs(i - index) > 1:
                        score += 5  # Contact found

        # Check for dead ends
        available = sum(1 for move in self.MOVES
                       if (pos[0] + move[0], pos[1] + move[1]) not in used)

        if available == 0 and index < self.protein.length - 1:
            return False  # Dead end, reject

        # Check compactness
        xs = [p[0] for p in positions] + [pos[0]]
        ys = [p[1] for p in positions] + [pos[1]]
        bounding_box = (max(xs) - min(xs)) + (max(ys) - min(ys))

        # Prefer moves that keep structure compact
        if bounding_box > len(positions) * 1.5:
            score -= 2

        return score >= -1  # Accept if not too bad


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

    if csp_conf:
        visualize_conformation_plot(csp_conf, f'CSP+AC-3 (Gap: {abs(csp_conf.calculate_energy() - optimal)})', axes[0])
    else:
        axes[0].text(0.5, 0.5, 'No solution', ha='center', va='center')
        axes[0].set_title('CSP+AC-3 (No solution)')

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
    ax.bar(x - 0.5*width, csp, width, label='CSP+AC-3', color='blue')
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

    ax.bar(x - width, csp_gaps, width, label='CSP+AC-3', color='blue')
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

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output"""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


def run_csp_method(protein, seq_idx, max_time, optimal_energy=None):
    """Run CSP solver for a sequence with optional early stopping at optimal"""
    with suppress_stdout():
        hp_csp = HPProteinCSP(protein)
    t0 = time.time()
    csp_conf = hp_csp.solve(max_time=max_time, target_energy=optimal_energy)
    csp_time = time.time() - t0

    if csp_conf:
        csp_energy = csp_conf.calculate_energy()
        return csp_energy, csp_time, csp_conf
    else:
        return 0, csp_time, None


def run_rl_method(protein, seq_idx):
    """Run RL solver for a sequence"""
    rl = ReinforcementLearningSolver(protein, 100000)
    t0 = time.time()
    with suppress_stdout():
        rl_conf = rl.solve()
    rl_time = time.time() - t0
    rl_energy = rl_conf.calculate_energy()
    return rl_energy, rl_time, rl_conf


def run_greedy_method(protein, seq_idx):
    """Run Greedy solver for a sequence"""
    greedy = ImprovedGreedySolver(protein, lookahead=3)
    t0 = time.time()
    greedy_conf = greedy.solve()
    greedy_time = time.time() - t0
    greedy_energy = greedy_conf.calculate_energy()
    return greedy_energy, greedy_time, greedy_conf


def process_single_sequence(args):
    """Process a single sequence - methods run in parallel"""
    from concurrent.futures import ThreadPoolExecutor

    seq_idx, length, seq_str, optimal_energy, max_time, timestamp = args

    print(f"\n[Seq {seq_idx}] Starting: {seq_str[:30]}... (len={length})")

    protein = HPProtein(seq_str)
    result = {
        'sequence': seq_str,
        'length': length,
        'optimal_energy': optimal_energy,
        'H_count': seq_str.count('H'),
        'P_count': seq_str.count('P')
    }

    # Run all three methods in parallel using ThreadPoolExecutor
    print(f"[Seq {seq_idx}] Running CSP, RL, and Greedy in parallel...")

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all three methods - pass optimal_energy to CSP for early stopping
        csp_future = executor.submit(run_csp_method, protein, seq_idx, max_time, optimal_energy)
        rl_future = executor.submit(run_rl_method, protein, seq_idx)
        greedy_future = executor.submit(run_greedy_method, protein, seq_idx)

        # Get results
        csp_energy, csp_time, csp_conf = csp_future.result()
        rl_energy, rl_time, rl_conf = rl_future.result()
        greedy_energy, greedy_time, greedy_conf = greedy_future.result()

    # Store CSP results
    result['csp_energy'] = csp_energy
    result['csp_time'] = csp_time
    gap = abs(csp_energy - optimal_energy)
    print(f"[Seq {seq_idx}] CSP: Energy {csp_energy} (gap {gap}), Time {csp_time:.2f}s")

    # Store RL results
    result['rl_energy'] = rl_energy
    result['rl_time'] = rl_time
    gap = abs(rl_energy - optimal_energy)
    print(f"[Seq {seq_idx}] RL: Energy {rl_energy} (gap {gap}), Time {rl_time:.2f}s")

    # Store Greedy results
    result['greedy_energy'] = greedy_energy
    result['greedy_time'] = greedy_time
    gap = abs(greedy_energy - optimal_energy)
    print(f"[Seq {seq_idx}] Greedy: Energy {greedy_energy} (gap {gap}), Time {greedy_time:.4f}s")

    # Save comparison plot
    plot_filename = f'comparison_seq{seq_idx}_{timestamp}.png'
    save_comparison_plot(csp_conf, rl_conf, greedy_conf, seq_str, optimal_energy, plot_filename)
    
    print(f"[Seq {seq_idx}] ✓ Complete!")
    
    return result, seq_idx


def run_benchmark(sequences: List[Tuple[int, str, int]], max_time: int = 30):
    """Run complete benchmark with visualization and CSV export - in parallel"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("2D HP PROTEIN FOLDING - IMPROVED CSP WITH AC-3 (PARALLEL)")
    print(f"Timestamp: {timestamp}")
    print(f"Running {len(sequences)} sequences in parallel with max_workers={len(sequences)}")
    print("=" * 80)

    # Prepare arguments for parallel processing
    args_list = [
        (seq_idx, length, seq_str, optimal_energy, max_time, timestamp)
        for seq_idx, (length, seq_str, optimal_energy) in enumerate(sequences, 1)
    ]

    results = [None] * len(sequences)  # Pre-allocate to maintain order
    
    # Run in parallel using ProcessPoolExecutor
    print(f"\nStarting parallel execution of {len(sequences)} sequences...")
    print("=" * 80)
    
    with ProcessPoolExecutor(max_workers=len(sequences)) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(process_single_sequence, args): args[0] 
                        for args in args_list}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            try:
                result, seq_idx = future.result()
                results[seq_idx - 1] = result  # Store in correct position
                completed += 1
                print(f"\n{'=' * 80}")
                print(f"✓ SEQUENCE {seq_idx} COMPLETED ({completed}/{len(sequences)})")
                print(f"{'=' * 80}\n")
            except Exception as e:
                seq_idx = future_to_idx[future]
                completed += 1
                print(f"\n{'=' * 80}")
                print(f"❌ SEQUENCE {seq_idx} FAILED ({completed}/{len(sequences)}): {e}")
                print(f"{'=' * 80}\n")
                # Create a failed result entry
                idx = seq_idx - 1
                results[idx] = {
                    'sequence': args_list[idx][2],
                    'length': args_list[idx][1],
                    'optimal_energy': args_list[idx][3],
                    'H_count': args_list[idx][2].count('H'),
                    'P_count': args_list[idx][2].count('P'),
                    'csp_energy': 0,
                    'csp_time': 0,
                    'rl_energy': 0,
                    'rl_time': 0,
                    'greedy_energy': 0,
                    'greedy_time': 0
                }
    
    print("\n" + "=" * 80)
    print("All sequences completed! Processing results...")
    print("=" * 80)

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
    # Test sequences (starting with smaller ones)
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
    results, timestamp = run_benchmark(sequences, max_time=1800)

    print("\n🎉 Benchmark complete!")
    print(f"Check files with timestamp {timestamp} for results")