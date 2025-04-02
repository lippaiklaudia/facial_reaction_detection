class StressEstimator:
    def estimate_stress(self, blink_rate, ear, pupil_ratio, blink_stddev, pupil_delta):
        score = 0

        if blink_rate > 25 or blink_rate < 5:
            score += 1

        if ear is not None and ear < 0.20:
            score += 1

        if pupil_ratio is not None and pupil_ratio > 0.6:
            score += 1

        if blink_stddev is not None and blink_stddev > 1.0:
            score += 1

        if pupil_delta is not None and pupil_delta > 0.15:
            score += 1

        return min(score, 5)
