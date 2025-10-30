import numpy as np
from collections import deque

def role_assignment(teammate_positions, formation_positions, ball_position=(0, 0), goal_position=(15, 0)):
    """
    Input:
      - teammate_positions: list of ndarrays [ [x,y], ... ] length N
      - formation_positions: list of ndarrays [ [x,y], ... ] length M
      - ball_position: ndarray([x, y]) position of the ball
      - goal_position: ndarray([x, y]) opponent’s goal (default = (15, 0))
    Output:
      - point_preferences: dict mapping unum (1..N) -> ndarray([x,y]) assigned position
    """

    N = len(teammate_positions)
    M = len(formation_positions)

    teammates = [np.asarray(p) for p in teammate_positions]
    formations = [np.asarray(p) for p in formation_positions]
    ball = np.asarray(ball_position)
    goal = np.asarray(goal_position)

    # === 1) Build player preference lists with tactical weighting ===
    player_prefs = []
    for p_idx, p_pos in enumerate(teammates):
        dists = []
        for r_idx, r_pos in enumerate(formations):
            # Base distance
            dist = np.linalg.norm(r_pos - p_pos)

            # Tactical factors
            # 1. Distance (primary)
            distance_score = dist

            # 2. Goal bias — prefer formation positions further toward opponent goal
            goal_bias = 15 - abs(r_pos[0])   # higher x closer to opponent goal

            # 3. Ball alignment — smaller angle between player->ball and player->formation is better
            vec_to_ball = ball - p_pos
            vec_to_role = r_pos - p_pos
            cos_angle = np.dot(vec_to_ball, vec_to_role) / (
                np.linalg.norm(vec_to_ball) * np.linalg.norm(vec_to_role) + 1e-6
            )
            ball_alignment = (1 - cos_angle)  # smaller = better

            # 4. Spacing bonus — prefer roles not too close to teammates
            avg_team_center = np.mean(teammates, axis=0)
            spacing_bonus = np.linalg.norm(r_pos - avg_team_center)

            # Combine with tuned weights (you can adjust these later)
            α, β, γ, δ = 0.6, -0.3, -0.2, 0.5
            weighted_score = (
                α * distance_score
                + β * goal_bias
                + γ * spacing_bonus
                + δ * ball_alignment
            )

            dists.append((weighted_score, r_idx))

        # Sort by weighted score (lower = better)
        dists.sort(key=lambda x: (x[0], x[1]))
        player_prefs.append([r for _, r in dists])

    # === 2) Build role preference ranking maps (unchanged except weights also applied) ===
    role_rank = []
    for r_idx, r_pos in enumerate(formations):
        dists = []
        for p_idx, p_pos in enumerate(teammates):
            # Now roles also rank players by tactical fit (not just distance)
            dist = np.linalg.norm(p_pos - r_pos)
            goal_proximity = abs(goal[0] - p_pos[0])
            spacing_from_ball = np.linalg.norm(p_pos - ball)

            # Combine into role preference (defenders prefer closer, forwards prefer higher x)
            w1, w2, w3 = 0.7, -0.3, 0.2
            role_score = w1 * dist + w2 * goal_proximity + w3 * spacing_from_ball

            dists.append((role_score, p_idx))
        dists.sort(key=lambda x: (x[0], x[1]))
        rank_map = {p: rank for rank, (_, p) in enumerate(dists)}
        role_rank.append(rank_map)

    # === 3) Gale-Shapley algorithm (same as before) ===
    current_matches = {r: None for r in range(M)}
    next_proposal = [0] * N
    unmatched = deque([p for p in range(N)])

    while unmatched:
        p = unmatched.popleft()
        if next_proposal[p] >= len(player_prefs[p]):
            continue

        r = player_prefs[p][next_proposal[p]]
        next_proposal[p] += 1

        current = current_matches[r]
        if current is None:
            current_matches[r] = p
        else:
            if role_rank[r].get(p, float('inf')) < role_rank[r].get(current, float('inf')):
                current_matches[r] = p
                if next_proposal[current] < len(player_prefs[current]):
                    unmatched.append(current)
            else:
                if next_proposal[p] < len(player_prefs[p]):
                    unmatched.append(p)

    # === 4) Build final mapping ===
    point_preferences = {}
    player_to_role = {p: None for p in range(N)}
    for r, p in current_matches.items():
        if p is not None:
            player_to_role[p] = r

    for p in range(N):
        unum = p + 1
        r = player_to_role[p]
        if r is None:
            point_preferences[unum] = np.array(teammates[p])
        else:
            point_preferences[unum] = np.array(formations[r])

    return point_preferences
