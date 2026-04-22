classdef CessnaMasterEnvv2 < rl.env.MATLABEnvironment
    % CessnaMasterEnvv2: 3-Axis Curriculum Learning 
    
    properties
        fdm                     % JSBSim interface
        TargetAltitude = 4000
        TargetSpeed = 100       % Target 100 kts
        
        % THE CURRICULUM SWITCH
        CurriculumPhase = 1;    
        
        % Internal Memory
        int_phi = 0;
    end
    
    properties(Access = protected)
        IsDone = false
        CurrentStep = 0
        EpisodeCount = 0
        PrevElev = 0            % For physical action smoothing
        PrevAgentElev = 0       % Tracking the raw agent output delta
    end
    
    methods
        function obj = CessnaMasterEnvv2()
            % Observation Space: [err_h, h_dot, theta, q, phi, p, err_v, elev_pos]
            obsInfo = rlNumericSpec([8 1]);
            obsInfo.Name = 'FlightStates';
            
            % Action Space: [Elevator, Aileron, Throttle] bounds [-1 to 1]
            actInfo = rlNumericSpec([3 1], 'LowerLimit', -1, 'UpperLimit', 1);
            actInfo.Name = 'FlightControls';
            
            obj = obj@rl.env.MATLABEnvironment(obsInfo, actInfo);
            
            % ==========================================================
            % Python JSBSim Initialization
            % ==========================================================
            try
                py.importlib.import_module('jsbsim');
            catch
                pyenv('Version', '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3');
                py.importlib.import_module('jsbsim');
            end
            
            root = '/Users/lee-michaelcookhorn/Documents/jsbsim/jsbsim-master';
            obj.fdm = py.jsbsim.FGFDMExec(root);
            obj.fdm.set_aircraft_path([root '/aircraft']);
            obj.fdm.set_engine_path([root '/engine']);
            obj.fdm.set_systems_path([root '/systems']);
            % NOTE: This must point to your new "dummy" barebones XML script!
            obj.fdm.load_script(py.str([root '/scripts/c172_cruise_test.xml']));
            obj.fdm.set_dt(0.02);
        end
        
        function [Observation, Reward, IsDone, LoggedSignals] = step(obj, Action)
            % Incremental step counter
            obj.CurrentStep = obj.CurrentStep + 1;
            
            % Read Agent Desired Actions
            agent_elev = double(Action(1));
            agent_ail  = double(Action(2));
            agent_thr  = (double(Action(3)) + 1) / 2; % Map [-1,1] to [0,1]
            
            % Track Agent Brain Smoothness
            delta_agent_elev = agent_elev - obj.PrevAgentElev;
            obj.PrevAgentElev = agent_elev;
            
            phi = double(obj.fdm.get_property_value('attitude/phi-rad'));
            p = double(obj.fdm.get_property_value('velocities/p-rad_sec'));
            v_kts = double(obj.fdm.get_property_value('velocities/vc-kts'));
            
            % 2. THE CURRICULUM OVERRIDE LOGIC & LOW-PASS ACTION FILTER
            % Alpha = 0.1 means the stick only moves 10% of the distance toward the 
            % agent's desired position per step. This physically absorbs high-frequency vibration.
            alpha_filter = 0.05;
            filtered_elev = (alpha_filter * agent_elev) + ((1 - alpha_filter) * obj.PrevElev);
            
            % Pre-calculate the PID Auto-Leveler output to be blended as needed
            err_phi = 0 - phi;
            obj.int_phi = max(min(obj.int_phi + (err_phi * 0.02), 0.5), -0.5);
            pid_ail = (1.5 * err_phi) + (0.1 * obj.int_phi) - (0.2 * p);

            if obj.CurriculumPhase == 1
                actual_elev = filtered_elev; 
                actual_ail = pid_ail;
                actual_thr = 0.65 + (0.02 * (obj.TargetSpeed - v_kts));
            elseif obj.CurriculumPhase == 2
                actual_elev = filtered_elev;
                actual_ail  = (0.1 * agent_ail) + (0.9 * pid_ail);
                actual_thr = 0.65 + (0.02 * (obj.TargetSpeed - v_kts));
            elseif obj.CurriculumPhase == 3
                actual_elev = filtered_elev;
                actual_ail  = (0.5 * agent_ail) + (0.5 * pid_ail);
                actual_thr = 0.65 + (0.02 * (obj.TargetSpeed - v_kts));
            elseif obj.CurriculumPhase >= 4
                actual_elev = filtered_elev;
                actual_ail  = agent_ail;
                actual_thr  = agent_thr;
            end
            
            % Enforce physical hard-stops
            actual_elev = max(min(actual_elev, 1), -1);
            actual_ail  = max(min(actual_ail, 1), -1);
            actual_thr  = max(min(actual_thr, 1), 0);
            
            % Smoothness Tracking (Delta is now naturally small and capped at 0.1)
            delta_elev = actual_elev - obj.PrevElev;
            obj.PrevElev = actual_elev;
            
            % 3. Send to JSBSim
            obj.fdm.set_property_value('fcs/elevator-cmd-norm', actual_elev);
            obj.fdm.set_property_value('fcs/aileron-cmd-norm', actual_ail);
            obj.fdm.set_property_value('fcs/throttle-cmd-norm', actual_thr);
            
            % Prevent JSBSim crash on extreme maneuvers
            try
                obj.fdm.run(); 
            catch
                Observation = zeros(8,1); Reward = -100; IsDone = true; LoggedSignals = []; return;
            end
            
            % 4. Gather New Observations
            h = double(obj.fdm.get_property_value('position/h-sl-ft'));
            h_dot = double(obj.fdm.get_property_value('velocities/h-dot-fps'));
            theta = double(obj.fdm.get_property_value('attitude/theta-rad'));
            q = double(obj.fdm.get_property_value('velocities/q-rad_sec'));
            phi_now = double(obj.fdm.get_property_value('attitude/phi-rad'));
            p_now = double(obj.fdm.get_property_value('velocities/p-rad_sec'));
            v_now = double(obj.fdm.get_property_value('velocities/vc-kts'));
            
            elev_pos = obj.PrevElev; 
            
            err_h = h - obj.TargetAltitude;
            err_v = v_now - obj.TargetSpeed;
            Observation = [err_h/1000; h_dot/100; theta; q; phi_now; p_now; err_v/10; elev_pos];
            
            % ==========================================================
            % 5. PROGRESSIVE DENSE REWARD FUNCTION (h_dot tracking)
            % ==========================================================
            
            % (A) Calculate Target Vertical Velocity (Outer Loop)
            target_h_dot = -(err_h * 0.05); 
            target_h_dot = max(min(target_h_dot, 10), -10); % Max 10 fps (600 ft/min)
            
            % (B) Survival Bonus
            Reward = 2.0; 
            
            % (C) Dense Goal Penalty: How far are we from the target climb rate?
            climb_penalty = abs(h_dot - target_h_dot) * 0.1;
            climb_penalty = min(climb_penalty, 1.5); 
            Reward = Reward - climb_penalty;
            
            % (C2) Direct altitude error — eliminates steady-state offset
            % Capped so severe turbulence doesn't overwhelm the reward signal
            Reward = Reward - min(abs(err_h)/600, 0.1);
            
            % (D) Bankrupting the Vibration Exploit (THE CHATTER CRUSHER)
            raw_smooth_penalty = (delta_agent_elev^2) * 10.0;
            
            % THE FIX: Cap the penalty at 3.0 to bound the math and prevent gradient starvation
            capped_smooth_penalty = min(raw_smooth_penalty, 3.0); 
            
            Reward = Reward - capped_smooth_penalty;
            
            % Penalize physical stick movement (stick pumping)
            % delta_elev was computed on line 113 BEFORE PrevElev was updated
            Reward = Reward - (abs(delta_elev) * 1.0); 
            
            Reward = Reward - (abs(actual_elev) * 0.05); % Gentle penalty for holding extreme un-trimmed stick

            if obj.CurriculumPhase >= 2
                Reward = Reward - abs(phi_now); 
            end
            if obj.CurriculumPhase >= 3
                Reward = Reward - abs(err_v)/20;  
            end
            
            % Speed Penalty Slope: Warn agent if speed < 75 kts
            %if v_now < 75
                %Reward = Reward - (75 - v_now); 
            %end
            
            % 6. TERMINATION & LOGGING LOGIC
            isCrash   = (h < 100);
            isSpin    = (abs(phi_now) > 1.2);
            isStall   = (v_now < 55);
            isPitchEx = (abs(theta) > 0.35); % +/- 20 degree pitch limits
            isTimeout = (obj.CurrentStep >= 3000); % Survived 60 seconds!
            
            IsDone = isCrash || isSpin || isStall || isPitchEx || isTimeout;
            
            % --- FIXED DEATH PENALTY ---
            if IsDone && ~isTimeout
                Reward = Reward - 100; 
            end
            
            obj.IsDone = IsDone;
            
            % Print to console if the episode just ended
            if IsDone
                if isTimeout
                    status = 'SUCCESS (Survived 60s)';
                elseif isCrash
                    status = 'CRASH (Hit Ground)';
                elseif isPitchEx
                    status = 'FAILURE (Extreme Pitch)';
                elseif isSpin
                    status = 'SPIN (Rolled Over)';
                elseif isStall
                    status = 'STALL (Airspeed < 55kts)';
                else
                    status = 'TERMINATED';
                end
                
                fprintf('Ep %4d | Phase %d | %s | Steps: %4d/3000 | Final Alt: %4.0f ft | Speed: %3.0f kts\n', ...
                    obj.EpisodeCount, obj.CurriculumPhase, status, obj.CurrentStep, h, v_now);
            end
            
            LoggedSignals = [];
        end
        
        function InitialObs = reset(obj)
            obj.EpisodeCount = obj.EpisodeCount + 1;
            obj.CurrentStep = 0;
            obj.PrevElev = 0; % Reset physical smoothness tracker
            obj.PrevAgentElev = 0; % Reset brain smoothness tracker
            
            % ==============================================
            % DOMAIN CONFIGURATION BASED ON CURRICULUM PHASE
            % ==============================================
            if obj.CurriculumPhase == 1
                % PHASE 1: DETERMINISTIC SPAWN. 
                random_alt = 4000;
                random_v   = 100;
                wind_sev   = 0;
            else
                % PHASE 2+: WIDER DOMAIN RANDOMIZATION (The PIO Fix)
                random_alt = obj.TargetAltitude + (rand() * 1200 - 600);
                
                % Spawn anywhere between 70 kts and 140 kts to force learning dynamic pressure sensitivity
                random_v = 80 + (rand() * 40); 
                
                wind_sev = 1 + (rand() * 3);
            end
            
            
            obj.fdm.set_property_value('ic/h-sl-ft', random_alt);
            obj.fdm.set_property_value('ic/vc-kts', random_v);
            obj.fdm.set_property_value('atmosphere/turbulence/milspec/wind-severity', wind_sev);
            
            obj.fdm.run_ic();
            
            % Start the engine
            obj.fdm.set_property_value('propulsion/engine[0]/set-running', 1);
            obj.fdm.set_property_value('propulsion/magneto_cmd', 3);
            obj.fdm.set_property_value('propulsion/starter_cmd', 1);
            
            % --- ZERO OUT FLIGHT CONTROLS ---
            % Fixes the bug where JSBSim retains elevator deflections from a
            % previous crash and holds them during the 0.5s engine spool-up
            obj.fdm.set_property_value('fcs/elevator-cmd-norm', 0.0);
            obj.fdm.set_property_value('fcs/aileron-cmd-norm', 0.0);
            obj.fdm.set_property_value('fcs/rudder-cmd-norm', 0.0);
            
            % --- ENGINE SPOOL UP ---
            obj.fdm.set_property_value('fcs/throttle-cmd-norm', 0.80);
            for settle_idx = 1:25
                obj.fdm.run();
            end
            
            % Refuel the aircraft every episode
            obj.fdm.set_property_value('propulsion/tank[0]/contents-lbs', 130); 
            obj.fdm.set_property_value('propulsion/tank[1]/contents-lbs', 130); 
            
            obj.int_phi = 0;
            
            % Grab the actual physics states after the spool up (Includes the h_dot bug fix!)
            h = double(obj.fdm.get_property_value('position/h-sl-ft'));
            h_dot = double(obj.fdm.get_property_value('velocities/h-dot-fps'));
            theta = double(obj.fdm.get_property_value('attitude/theta-rad'));
            q = double(obj.fdm.get_property_value('velocities/q-rad_sec'));
            phi_now = double(obj.fdm.get_property_value('attitude/phi-rad'));
            p_now = double(obj.fdm.get_property_value('velocities/p-rad_sec'));
            v_kts = double(obj.fdm.get_property_value('velocities/vc-kts'));
            
            elev_pos = obj.PrevElev; 
            
            err_h = h - obj.TargetAltitude;
            err_v = v_kts - obj.TargetSpeed;
            InitialObs = [err_h/1000; h_dot/100; theta; q; phi_now; p_now; err_v/10; elev_pos];
        end
    end
end