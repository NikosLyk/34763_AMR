import numpy as np
from scipy.optimize import linear_sum_assignment

class GNNDataAssociator:
    """
    Global Nearest Neighbor (GNN) Data Association using the Hungarian algorithm.
    """
    def __init__(self, gate_threshold: float = 9.21):
        self.gate_threshold = gate_threshold
        self.GATE_PENALTY = 1e5

    def compute_mahalanobis_distance(y: np.ndarray, S: np.ndarray) -> float:
        """
        Computes the squared Mahalanobis distance.

        Args:
            y: The innovation vector (z - h(x)), shape (N, 1)
            S: The innovation covariance matrix, shape (N, N)

        Returns:
            float: The squared Mahalanobis distance (d^2)
        """
        S_inv = np.linalg.inv(S)

        d_squared = y.T @ S_inv @ y

        return float(np.squeeze(d_squared))

    def _compute_cost_matrix(self, tracks: list, measurements: list, coord_manager) -> np.ndarray:
        """
        Computes the distance matrix between all tracks and measurements.
        Currently uses Euclidean distance as a placeholder.
        TODO Need data from the coordinate frame manager.
        """
        num_tracks = len(tracks)
        num_measurements = len(measurements)
        cost_matrix = np.zeros((num_tracks, num_measurements))

        for i, track in enumerate(tracks):
            for j, measurement in enumerate(measurements):
                z = measurement['z']
                sensor_id = measurement['sensor_id']
                h, H, R = coord_manager.get_model(track['ekf'].X, sensor_id)
                y, S = track.ekf.compute_innovation(z, h, H, R)
                #dx = track['x'] - measurement['x']
                #dy = track['y'] - measurement['y']
                #dist = np.sqrt(np.square(dx) + np.square(dy))
                dist = self.compute_mahalanobis_distance(y, S)
                cost_matrix[i, j] = dist

        return cost_matrix

    def associate(self, tracks: list, measurements: list):
        """
        Assigns measurements to tracks using GNN.

        Returns:
            matches (list of tuples): [(track_idx, meas_idx), ...]
            unmatched_tracks (list): [track_idx, ...]
            unmatched_measurements (list): [meas_idx, ...]
        """
        matches = []
        unmatched_tracks = []
        unmatched_measurements = []

        if len(measurements) == 0:
            return [], list(range(len(tracks))), []

        if len(tracks) == 0:
            return [], [], list(range(len(measurements)))

        cost_matrix = self._compute_cost_matrix(tracks, measurements)

        gated_cost_matrix = np.where(cost_matrix > self.gate_threshold, self.GATE_PENALTY, cost_matrix)

        row_idx, col_idx = linear_sum_assignment(gated_cost_matrix)

        for r, c in zip(row_idx, col_idx):
            if gated_cost_matrix[r, c] >= self.GATE_PENALTY:
                unmatched_tracks.append(int(r))
                unmatched_measurements.append(int(c))
            else:
                matches.append((int(r), int(c)))

        assigned_tracks = set(row_idx)
        assigned_measurements = set(col_idx)

        for i in range(len(tracks)):
            if i not in assigned_tracks:
                unmatched_tracks.append(i)

        for j in range(len(measurements)):
            if j not in assigned_measurements:
                unmatched_measurements.append(j)

        return matches, unmatched_tracks, unmatched_measurements