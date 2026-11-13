"""All tunable numbers for the project. Defaults are starting guesses."""

from dataclasses import dataclass


def _steps(duration: float, dt: float) -> int:
    """Convert a duration in time units to a whole number of steps."""
    return round(duration / dt)


@dataclass(frozen=True)
class Config:
    # --- World ---
    signal_zone: tuple[float, float] = (0.0, 1.0)
    target_zone: tuple[float, float] = (2.0, 3.0)
    min_landmark_spacing: float = 0.3
    signaller_start: float = 0.25
    receiver_start: float = 0.75
    n_colours: int = 2  # 2 = red, green; 3 adds blue

    # --- Sensors ---
    proximity_range: float = 1.0
    landmark_range: float = 0.2

    # --- Time (in time units; step counts are derived below) ---
    dt: float = 0.1
    trial_duration: float = 30.0
    score_duration: float = 2.0

    # --- Scoring ---
    score_range: float = 0.5
    success_radius: float = 0.1

    # --- CTRNN ---
    n_neurons: int = 6
    v_max: float = 1.0
    weight_range: tuple[float, float] = (-10.0, 10.0)
    bias_range: tuple[float, float] = (-10.0, 10.0)
    tau_range: tuple[float, float] = (0.5, 10.0)  # log-scaled mapping

    # --- Scripted components ---
    signal_start: float = 1.0
    pulse_on: float = 0.5
    pulse_period: float = 1.0
    cheat_stop_threshold: float = 0.5

    # --- Evolution ---
    algorithm: str = "ga"  # file in evolvers/: "ga" or "neat"
    pop_size: int = 400
    elite_fraction: float = 0.05
    mutation_std: float = 0.1
    n_generations: int = 1000

    # --- NEAT-style evolver only ---
    n_species: int = 10
    neat_survival: float = 0.2  # top fraction of each species that can be a parent
    neat_add_connection: float = 0.3  # chance per child of switching on one connection
    neat_remove_connection: float = 0.1  # chance per child of switching off one connection

    # --- Fitness and evaluation ---
    n_layouts: int = 4
    n_heldout_layouts: int = 100
    seed: int = 0
    heldout_seed: int = 12345

    # --- Derived values ---
    @property
    def n_steps(self) -> int:
        return _steps(self.trial_duration, self.dt)

    @property
    def score_steps(self) -> int:
        return _steps(self.score_duration, self.dt)

    @property
    def signal_start_step(self) -> int:
        return _steps(self.signal_start, self.dt)

    @property
    def pulse_on_steps(self) -> int:
        return _steps(self.pulse_on, self.dt)

    @property
    def pulse_period_steps(self) -> int:
        return _steps(self.pulse_period, self.dt)

    @property
    def n_inputs(self) -> int:
        return 1 + self.n_colours  # proximity + one per colour

    @property
    def motor_index(self) -> int:
        return self.n_neurons - 1

    @property
    def n_params_per_agent(self) -> int:
        n, k = self.n_neurons, self.n_inputs
        return n * n + n + n + k * n  # weights, biases, taus, input weights

    @property
    def genome_size(self) -> int:
        return 2 * self.n_params_per_agent
