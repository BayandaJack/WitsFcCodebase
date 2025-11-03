from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np
import time

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance





    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
        return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation (optional)
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()

    def think_and_send(self):
        
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass


        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""




    def select_skill(self, strategyData):
        drawer = self.world.draw

        #------------------------------------------------------
        # Role Assignment Phase
        formation_positions = GenerateBasicFormation(step_size=0.1)
        point_preferences = role_assignment(
            strategyData.teammate_positions,
            formation_positions,
            strategyData.ball_2d
        )

        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        strategyData.my_desried_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
            strategyData.my_desired_position
        )

        my_pos = np.array(strategyData.mypos, dtype=float)
        ball_pos = np.array(strategyData.ball_2d, dtype=float)
        goal_pos = np.array([15.0, 0.0], dtype=float)
        our_goal_pos = np.array([-15.0, 0.0], dtype=float)  # Our goal (left side)
        team_direction = 1  # attacking right

        #------------------------------------------------------
        # Maintain a sticky active player
        if not hasattr(self, "active_player_unum"):
            self.active_player_unum = None

        distances_to_ball = [np.linalg.norm(np.array(pos, dtype=float) - ball_pos)
                            for pos in strategyData.teammate_positions]
        closest_to_ball_unum = np.argmin(distances_to_ball) + 1
        closest_distance = distances_to_ball[closest_to_ball_unum - 1]

        # Sticky logic with improved possession detection
        if self.active_player_unum is None or closest_distance < 0.5:
            self.active_player_unum = closest_to_ball_unum
            self.dribble_counter = 0
        else:
            active_index = self.active_player_unum - 1
            if active_index < len(strategyData.teammate_positions):
                active_pos = np.array(strategyData.teammate_positions[active_index], dtype=float)
                dist_to_ball = np.linalg.norm(active_pos - ball_pos)
                if dist_to_ball > 2.0:
                    self.active_player_unum = closest_to_ball_unum
                    self.dribble_counter = 0

        strategyData.active_player_unum = self.active_player_unum

        #------------------------------------------------------
        # 🧤 Goalkeeper logic — stays near goal
        if strategyData.player_unum == 1:  # assuming player 1 is keeper
            goal_x, goal_y = -15.0, 0.0  # left-side goal
            distance_to_goal = np.linalg.norm(my_pos - np.array([goal_x, goal_y]))
            distance_to_ball = np.linalg.norm(my_pos - ball_pos)

            # Keep within small area near goal, chase only if close
            if distance_to_ball < 5.0:
                return self.move(ball_pos)
            elif distance_to_goal > 3.0:
                return self.move(np.array([goal_x, goal_y]))
            else:
                return self.move(my_pos)  # stay idle near goal

        #------------------------------------------------------
        # 🧍 Non-active players: move BEHIND the active player (3-4m support)
        if strategyData.player_unum != self.active_player_unum:
            active_pos = np.array(strategyData.teammate_positions[self.active_player_unum - 1], dtype=float)
            
            # NEW: Calculate support position 3-4m BEHIND the active player
            ball_to_active = active_pos - ball_pos
            ball_to_active_norm = np.linalg.norm(ball_to_active)
            
            if ball_to_active_norm > 0:
                # Direction from ball to active player
                ball_to_active_dir = ball_to_active / ball_to_active_norm
                
                # Support position is 3-4m behind active player (towards our goal)
                SUPPORT_DISTANCE = 3.5  # meters behind
                support_pos = active_pos - ball_to_active_dir * SUPPORT_DISTANCE
                
                # Add some lateral offset to avoid stacking
                perp_offset = np.array([-ball_to_active_dir[1], ball_to_active_dir[0]]) * 1.5
                # Alternate left/right based on player number
                if strategyData.player_unum % 2 == 0:
                    support_pos += perp_offset
                else:
                    support_pos -= perp_offset
            else:
                # Fallback: use formation position
                support_pos = np.array(strategyData.my_desired_position, dtype=float)

            # Clamp within pitch
            support_pos[0] = np.clip(support_pos[0], -15, 15)
            support_pos[1] = np.clip(support_pos[1], -10, 10)

            drawer.annotation(support_pos, f"P{strategyData.player_unum}", drawer.Color.cyan, "support")
            orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos)  # Face the ball
            return self.move(support_pos, orientation=orientation)

        #------------------------------------------------------
        # ⚽ Active player logic
        drawer.annotation((0, 10.5), f"🏃 Active: Player #{self.active_player_unum}", drawer.Color.yellow, "status")

        my_distance_to_goal = np.linalg.norm(my_pos - goal_pos)
        DRIBBLE_SPEED = 2.0
        BALL_CONTROL_DISTANCE = 0.15

        if not hasattr(self, "last_pass_time"):
            self.last_pass_time = 0
        PASS_COOLDOWN = 0.6
        now = time.time()
        can_pass = (now - self.last_pass_time) >= PASS_COOLDOWN

        if not hasattr(self, "dribble_counter"):
            self.dribble_counter = 0

        #------------------------------------------------------
        # 🆕 CRITICAL: Check if facing wrong direction and pass back
        def is_facing_wrong_direction(my_pos, ball_pos, our_goal_pos):
            """Check if player is facing towards our own goal (bad direction)"""
            # Vector from player to ball
            player_to_ball = ball_pos - my_pos
            
            # Vector from player to our goal
            player_to_our_goal = our_goal_pos - my_pos
            
            # Normalize vectors
            if np.linalg.norm(player_to_ball) > 0:
                player_to_ball = player_to_ball / np.linalg.norm(player_to_ball)
            if np.linalg.norm(player_to_our_goal) > 0:
                player_to_our_goal = player_to_our_goal / np.linalg.norm(player_to_our_goal)
            
            # Calculate dot product - if positive, we're facing towards our goal
            dot_product = np.dot(player_to_ball, player_to_our_goal)
            
            # If dot product > 0.7, we're facing significantly towards our goal
            return dot_product > 0.7

        def find_backward_pass_target(strategyData, my_pos, ball_pos):
            """Find best teammate for backward pass (behind the active player)"""
            best_teammate = None
            best_score = -float('inf')
            
            ball_to_me = my_pos - ball_pos
            if np.linalg.norm(ball_to_me) > 0:
                ball_to_me_dir = ball_to_me / np.linalg.norm(ball_to_me)
            else:
                ball_to_me_dir = np.array([1.0, 0.0])
            
            for i, teammate_pos in enumerate(strategyData.teammate_positions):
                teammate_pos = np.array(teammate_pos, dtype=float)
                teammate_unum = i + 1
                
                # Skip self and goalkeeper
                if teammate_unum == strategyData.player_unum or teammate_unum == 1:
                    continue
                
                # Vector from ball to teammate
                ball_to_teammate = teammate_pos - ball_pos
                
                # Check if teammate is behind me (relative to ball direction)
                if np.linalg.norm(ball_to_teammate) > 0:
                    ball_to_teammate_dir = ball_to_teammate / np.linalg.norm(ball_to_teammate)
                    behind_factor = np.dot(ball_to_me_dir, ball_to_teammate_dir)
                    
                    # Distance factors
                    dist_to_me = np.linalg.norm(teammate_pos - my_pos)
                    dist_score = 1.0 / (dist_to_me + 0.1)  # Prefer closer teammates
                    
                    # Angle score (how directly behind they are)
                    angle_score = max(0, behind_factor)
                    
                    # Total score
                    score = angle_score * 2.0 + dist_score
                    
                    if score > best_score and dist_to_me < 8.0:  # Reasonable pass distance
                        best_score = score
                        best_teammate = (teammate_unum, teammate_pos)
            
            return best_teammate

        # Check if we're facing wrong direction and should pass back
        facing_wrong_way = is_facing_wrong_direction(my_pos, ball_pos, our_goal_pos)
        if facing_wrong_way and can_pass:
            backward_target = find_backward_pass_target(strategyData, my_pos, ball_pos)
            if backward_target:
                drawer.annotation((0, 9.0), "🔄 FACING WRONG WAY → PASS BACK!", drawer.Color.red, "wrong_way_status")
                drawer.line(strategyData.mypos, backward_target[1], 2, drawer.Color.red, "backward_pass_line")
                self.last_pass_time = now
                self.dribble_counter = 0
                return self.kickTarget(strategyData, strategyData.mypos, backward_target[1])

        #------------------------------------------------------
        # 🧠 Mandatory close-range pass when near keeper
        CLOSE_RANGE = 10.0
        KEEPER_ALERT_DISTANCE = 5.0
        keeper_pos = np.array([-15.0, 0.0])  # assuming opposing keeper defends left goal
        keeper_to_ball_dist = np.linalg.norm(goal_pos - my_pos)

        if my_distance_to_goal <= CLOSE_RANGE and keeper_to_ball_dist <= KEEPER_ALERT_DISTANCE and can_pass:
            # find nearest teammate near goal
            best_teammate = None
            best_dist = float("inf")
            for i, pos in enumerate(strategyData.teammate_positions):
                pos = np.array(pos, dtype=float)
                if i + 1 == self.active_player_unum or i + 1 == 1:
                    continue
                dist_to_goal = np.linalg.norm(pos - goal_pos)
                dist_to_me = np.linalg.norm(pos - my_pos)
                if dist_to_goal < my_distance_to_goal and dist_to_me < 6.0:
                    if dist_to_goal < best_dist:
                        best_dist = dist_to_goal
                        best_teammate = (i + 1, pos)

            if best_teammate:
                drawer.annotation((0, 9.5), f"⚠️ Close-range pass → #{best_teammate[0]}", drawer.Color.cyan, "pass_status")
                drawer.line(strategyData.mypos, best_teammate[1], 2, drawer.Color.red, "pass_line")
                self.last_pass_time = now
                self.dribble_counter = 0
                return self.kickTarget(strategyData, strategyData.mypos, best_teammate[1])

        #------------------------------------------------------
        # --- SHOOT if in range
        SHOOT_RANGE = 4.5
        if my_distance_to_goal < SHOOT_RANGE:
            drawer.annotation((0, 9.5), "In range → SHOOT!", drawer.Color.green, "shoot_status")
            drawer.line(strategyData.mypos, goal_pos, 2, drawer.Color.red, "shot line")
            self.dribble_counter = 0
            return self.kickTarget(strategyData, strategyData.mypos, goal_pos)

        #------------------------------------------------------
        # 🎯 Pass logic: look for teammate closer to goal
        teammate_positions = strategyData.teammate_positions
        best_teammate = None
        best_dist_to_goal = float("inf")

        for i, pos in enumerate(teammate_positions):
            pos = np.array(pos, dtype=float)
            if i + 1 != self.active_player_unum and np.linalg.norm(pos - goal_pos) < my_distance_to_goal:
                dist_to_goal = np.linalg.norm(pos - goal_pos)
                if dist_to_goal < best_dist_to_goal:
                    best_dist_to_goal = dist_to_goal
                    best_teammate = (i + 1, pos)

        # Pass after 2–3 dribbles or if clear forward teammate
        if best_teammate and can_pass and (self.dribble_counter >= 2 or (my_distance_to_goal - best_dist_to_goal) > 1.0):
            drawer.annotation((0, 9.5), f"Pass → #{best_teammate[0]}", drawer.Color.cyan, "pass_status")
            drawer.line(strategyData.mypos, best_teammate[1], 2, drawer.Color.red, "pass_line")
            self.last_pass_time = now
            self.dribble_counter = 0
            return self.kickTarget(strategyData, strategyData.mypos, best_teammate[1])

        #------------------------------------------------------
        # Otherwise, DRIBBLE toward the goal — BUT first approach from behind when necessary
        dist_to_ball = np.linalg.norm(my_pos - ball_pos)

        # compute vectors
        to_goal_from_ball = goal_pos - ball_pos
        dist_goal_from_ball = np.linalg.norm(to_goal_from_ball)
        unit_goal_from_ball = to_goal_from_ball / dist_goal_from_ball if dist_goal_from_ball > 0 else np.array([1.0, 0.0])

        APPROACH_BACK_DIST = 0.6
        approach_point = ball_pos - unit_goal_from_ball * APPROACH_BACK_DIST

        vec_my_to_ball = my_pos - ball_pos
        proj = np.dot(vec_my_to_ball, unit_goal_from_ball)
        in_front_of_ball = proj > 0.15

        if not hasattr(self, "possession_lock"):
            self.possession_lock = 0

        if dist_to_ball <= BALL_CONTROL_DISTANCE:
            self.possession_lock = time.time() + 0.8

        if in_front_of_ball and dist_to_ball > BALL_CONTROL_DISTANCE:
            dribble_target = approach_point
            orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos)
            drawer.annotation(approach_point, "approach", drawer.Color.yellow, "approach")
            drawer.line(my_pos, approach_point, 1, drawer.Color.yellow, "approach_line")
            return self.move(dribble_target, orientation=orientation)

        if time.time() < self.possession_lock:
            dribble_target = ball_pos + unit_goal_from_ball * 0.4
            direction_to_ball = ball_pos - my_pos
            if np.linalg.norm(direction_to_ball) > 0:
                direction_to_ball /= np.linalg.norm(direction_to_ball)
            blended_dir = 0.75 * unit_goal_from_ball + 0.25 * direction_to_ball
            orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(my_pos + blended_dir)
            self.dribble_counter += 1
            drawer.annotation((0, 9.5), f"⚽ Possession lock dribble ({self.dribble_counter})", drawer.Color.orange, "dribble_status")
            drawer.line(strategyData.mypos, dribble_target, 2, drawer.Color.green, "dribble_line")
            return self.move(dribble_target, orientation=orientation)

        if dist_to_ball > BALL_CONTROL_DISTANCE:
            dribble_target = ball_pos
            orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos)
            self.dribble_counter += 1
            drawer.annotation((0, 9.5), f"⚽ Chasing ({self.dribble_counter})", drawer.Color.orange, "chase_status")
            drawer.line(strategyData.mypos, dribble_target, 2, drawer.Color.green, "chase_line")
            return self.move(dribble_target, orientation=orientation)

        direction_to_goal = unit_goal_from_ball
        dribble_target = my_pos + direction_to_goal * DRIBBLE_SPEED
        direction_to_ball = ball_pos - my_pos
        if np.linalg.norm(direction_to_ball) > 0:
            direction_to_ball /= np.linalg.norm(direction_to_ball)
        blended_dir = 0.7 * direction_to_goal + 0.3 * direction_to_ball
        orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(my_pos + blended_dir)

        self.dribble_counter += 1
        drawer.annotation((0, 9.5), f"⚽ Dribbling ({self.dribble_counter})", drawer.Color.orange, "dribble_status")
        drawer.line(strategyData.mypos, dribble_target, 2, drawer.Color.green, "dribble_line")

        return self.move(dribble_target, orientation=orientation)




    

    #--------------------------------------- Fat proxy auxiliary methods


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")