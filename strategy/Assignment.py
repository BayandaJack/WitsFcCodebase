import numpy as np
from collections import deque

def role_assignment(teammate_positions, formation_positions):
    """
    Input:
      - teammate_positions: list of ndarrays [ [x,y], ... ] length N
      - formation_positions: list of ndarrays [ [x,y], ... ] length M
    Output:
      - point_preferences: dict mapping unum (1..N) -> ndarray([x,y]) assigned position
        If a player is unmatched (only possible when M < N), assign their current position.
    """

    N = len(teammate_positions)
    M = len(formation_positions)

    # Quick conversion to numpy arrays for vectorized distance calculation if not already
    teammates = [np.asarray(p) for p in teammate_positions]
    formations = [np.asarray(p) for p in formation_positions]

    # 1) Build player preference lists (players -> list of role indices sorted by nearest)
    player_prefs = []
    for p_idx, p_pos in enumerate(teammates):
        # compute squared distances to all formation positions
        dists = []
        for r_idx, r_pos in enumerate(formations):
            dx = float(r_pos[0]) - float(p_pos[0])
            dy = float(r_pos[1]) - float(p_pos[1])
            sq = dx*dx + dy*dy
            dists.append((sq, r_idx))
        # sort by distance then by role index as deterministic tie-breaker
        dists.sort(key=lambda x: (x[0], x[1]))
        player_prefs.append([r for _, r in dists])

    # 2) Build role preference ranking maps (role -> player -> rank)
    role_rank = []
    for r_idx, r_pos in enumerate(formations):
        dists = []
        for p_idx, p_pos in enumerate(teammates):
            dx = float(p_pos[0]) - float(r_pos[0])
            dy = float(p_pos[1]) - float(r_pos[1])
            sq = dx*dx + dy*dy
            dists.append((sq, p_idx))
        # sort ascending (best players first)
        dists.sort(key=lambda x: (x[0], x[1]))
        # create rank map: player_idx -> rank (lower is better)
        rank_map = {p: rank for rank, (_, p) in enumerate(dists)}
        role_rank.append(rank_map)

    # 3) Gale-Shapley (players propose)
    # current_matches: role_index -> player_index or None
    current_matches = {r: None for r in range(M)}
    # next_proposal pointer for each player: index into player_prefs[p]
    next_proposal = [0] * N
    # queue of players who are unmatched but still have roles to propose to
    unmatched = deque([p for p in range(N)])

    while unmatched:
        p = unmatched.popleft()
        if next_proposal[p] >= len(player_prefs[p]):
            # this player has proposed to all roles and remains unmatched
            continue

        r = player_prefs[p][next_proposal[p]]
        next_proposal[p] += 1

        current = current_matches[r]
        if current is None:
            # role r accepts p
            current_matches[r] = p
        else:
            # role r prefers between current and p
            # lower rank number means more preferred
            if role_rank[r].get(p, float('inf')) < role_rank[r].get(current, float('inf')):
                # r prefers p: swap
                current_matches[r] = p
                # previous player becomes unmatched (if they still have proposals left)
                if next_proposal[current] < len(player_prefs[current]):
                    unmatched.append(current)
            else:
                # r rejects p; p remains unmatched if still has proposals
                if next_proposal[p] < len(player_prefs[p]):
                    unmatched.append(p)

    # 4) Build point_preferences mapping unum (1..N) -> assigned position (ndarray)
    point_preferences = {}
    # invert current_matches to get player -> role
    player_to_role = {p: None for p in range(N)}
    for r, p in current_matches.items():
        if p is not None:
            player_to_role[p] = r

    for p in range(N):
        unum = p + 1
        r = player_to_role[p]
        if r is None:
            # unmatched player: fallback policy — keep current position (or handle as you prefer)
            point_preferences[unum] = np.array(teammates[p])
        else:
            point_preferences[unum] = np.array(formations[r])

    return point_preferences
