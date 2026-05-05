import numpy as np
from scipy.optimize import linear_sum_assignment

class GNNDataAssociator:
    """
    Global Nearest Neighbor (GNN) Data Association using the Hungarian algorithm.
    """
    def __init__(self, gate_threshold: float = 9.21):
        self.gate_threshold = gate_threshold
        self.GATE_PENALTY = 1e5
        self.sensor_status = {
            'radar': True,
            'camera': True,
            'ais': True,
            'gnss': True
        }

    def set_sensor_availability(self, sensor_id, is_available):
        if sensor_id in self.sensor_status:
            self.sensor_status[sensor_id] = is_available

    def compute_mahalanobis_distance(self, y: np.ndarray, S: np.ndarray) -> float:
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

    def _compute_cost_matrix(self, tracks: list, measurements: list, coord_managers: dict) -> np.ndarray:
        """
        Computes the distance matrix between all tracks and measurements.
        """
        num_tracks = len(tracks)
        num_measurements = len(measurements)
        cost_matrix = np.zeros((num_tracks, num_measurements))

        for i, track in enumerate(tracks):
            x_pred = np.squeeze(track['ekf'].X).flatten()
            for j, measurement in enumerate(measurements):
                z = measurement['z'].flatten()
                sensor_id = measurement['sensor_id']
                manager = coord_managers[sensor_id]
                h = manager.get_h(x_pred).flatten()
                H = manager.get_H(x_pred)
                R = manager.get_R()
                is_polar = sensor_id in ['radar', 'camera']
                y, S = track['ekf'].compute_innovation(z, h, H, R, is_polar=is_polar)
                #dx = track['x'] - measurement['x']
                #dy = track['y'] - measurement['y']
                #dist = np.sqrt(np.square(dx) + np.square(dy))
                dist = self.compute_mahalanobis_distance(y, S)
                cost_matrix[i, j] = dist

        return cost_matrix


    def associate(self, tracks, measurements, coord_managers):
        """
        Assigns measurements to tracks using GNN.
        """
        # Filter measurements based on sensor availability flag
        available_measurements = [
            m for m in measurements
            if self.sensor_status.get(m['sensor_id'], False)
        ]

        if not available_measurements:
            return [], list(range(len(tracks))), []
        if not tracks:
            return [], [], list(range(len(available_measurements)))

        cost_matrix = self._compute_cost_matrix(tracks, available_measurements, coord_managers)

        gated_cost_matrix = np.where(cost_matrix > self.gate_threshold, self.GATE_PENALTY, cost_matrix)

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(gated_cost_matrix)

        matches = []
        unmatched_tracks = []
        unmatched_measurements = []

        for r, c in zip(row_ind, col_ind):
            if gated_cost_matrix[r, c] >= self.GATE_PENALTY:
                unmatched_tracks.append(int(r))
                unmatched_measurements.append(int(c))
            else:
                matches.append((int(r), int(c)))

        assigned_tracks = set(row_ind)
        for i in range(len(tracks)):
            if i not in assigned_tracks:
                unmatched_tracks.append(i)

        assigned_meas = set(col_ind)
        for j in range(len(available_measurements)):
            if j not in assigned_meas:
                unmatched_measurements.append(j)

        return matches, unmatched_tracks, unmatched_measurements