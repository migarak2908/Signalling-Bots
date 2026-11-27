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
    unroll: int = 10  # steps per loop iteration, only affects speed

    # Scoring
    score_range: float = 0.5
    success_radius: float = 0.1

    # Network
    n_extra: int = 0  # neurons without a sensor, after the proximity neuron
    motor: str = "single"  # "single": the last neuron sets the velocity. "pair": two added motor neurons, right - left
    motor_neuron: int = -1
    param_range: tuple[float, float] = (-10.0, 10.0)  # weights, biases and input weights
    tau_range: tuple[float, float] = (0.5, 10.0)
    full_input: bool = False  # every sensor feeds every neuron
    plastic: bool = False  # Hebbian weight changes within a trial, rule genes eta, A, B, C, D

    # Evolution
    pop_size: int = 200
    elite_fraction: float = 0.05
    mutation_std: float = 0.1
    generations: int = 5000
    layouts_per_order: int = 10
    init_range: float = 1.0  # starting weight, bias and input weight genes are drawn from [-init_range, init_range]
    centre: bool = False  # starting biases are centre-crossing: theta_i = -sum_j w_ji / 2

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
        return self.n_inputs + self.n_extra + self.n_motors

    @property
    def n_motors(self) -> int:
        return 2 if self.motor == "pair" else 0

    @property
    def params_per_agent(self) -> int:
        n = self.n_neurons
        return n * n + 2 * n + self.n_input_weights + 5 * self.plastic

    @property
    def n_input_weights(self) -> int:
        return self.n_inputs * self.n_neurons if self.full_input else self.n_inputs

    @property
    def genome_size(self) -> int:
        return 2 * self.params_per_agent
