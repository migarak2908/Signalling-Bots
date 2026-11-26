from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # World
    n_colours: int = 3
    signal_zone: tuple[float, float] = (0.0, 1.0)
    target_zone: tuple[float, float] = (2.0, 3.0)
    min_gap: float = 0.3
    signaller_start: float = 0.25
    receiver_start: float = 0.75
    landmark_range: float = 0.2

    # Time
    dt: float = 0.1
    trial_time: float = 30.0
    score_time: float = 2.0

    # Scoring
    score_range: float = 0.5
    success_radius: float = 0.1

    # Network
    n_extra: int = 0  # neurons with no sensor input, after the proximity neuron
    motor_neuron: int = -1  # last neuron: proximity if n_extra is 0, else the last extra neuron
    param_range: tuple[float, float] = (-10.0, 10.0)  # weights, biases and input weights
    tau_range: tuple[float, float] = (0.5, 10.0)
    plastic: bool = False  # within-trial weight changes: eta, A, B, C, D genes per agent

    # Evolution
    pop_size: int = 200
    elite_fraction: float = 0.05
    mutation_std: float = 0.1
    generations: int = 5000
    layouts_per_order: int = 10

    # Testing
    test_layouts: int = 100
    test_seed: int = 12345
    test_dt: float = 0.01
    solved_success: float = 0.9

    @property
    def n_steps(self) -> int:
        return round(self.trial_time / self.dt)

    @property
    def score_steps(self) -> int:
        return round(self.score_time / self.dt)

    @property
    def n_inputs(self) -> int:
        return self.n_colours + 1

    @property
    def n_neurons(self) -> int:
        return self.n_inputs + self.n_extra

    @property
    def params_per_agent(self) -> int:
        n = self.n_neurons
        return n * n + 2 * n + self.n_inputs + 5 * self.plastic

    @property
    def genome_size(self) -> int:
        return 2 * self.params_per_agent
