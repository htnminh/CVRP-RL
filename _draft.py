        self.observation_space['coord'] = self.other_data['stops_coords'].astype('int32')
        self.observation_space['demand'] = self.ortools_data['demands'].astype('int32')
        self.observation_space['current_load'] = self.ortools_data['vehicle_caps'][0]
        self.observation_space['visited'] = np.zeros(self.n_stops)
        self.observation_space['current_stop'] = 0


 # Gymnasium env
        self.action_space = spaces.Discrete(self.n_stops, seed=gym_seed)        
        self.observation_space = spaces.Dict({
            'coord': spaces.Box(
                low=0, high=self.max_env_size,
                shape=(self.n_stops, 2), dtype=np.float32,
                seed=gym_seed),
            'demand': spaces.Box(
                low=0, high=self.max_demand,
                shape=(self.n_stops, ), dtype=np.float32,
                seed=gym_seed),
            'visited': spaces.Box(
                low=0, high=1,
                shape=(self.n_stops, ), dtype=int,
                seed=gym_seed),
            'current_load': spaces.Box(
                low=0, high=self.max_vehicle_cap,
                shape=(1, ), dtype=np.float32,
                seed=gym_seed),
            'current_stop': spaces.Discrete(self.n_stops, seed=gym_seed)
        })  # visited of stop 0 should always be 0
        self.reward_range = (-np.inf, 0)

observation = dict(
            coord=self.other_data['stops_coords'].astype(int),
            demand=self.ortools_data['demands'].astype(int),
            visited=np.zeros(self.n_stops).astype(int),
            current_load=np.array([self.ortools_data['vehicle_caps'][0], ], dtype=int),
            current_stop=0,
            current_length=np.array([0, ], dtype=int)
        )