# Signalling-Bots
Evolving simple neural agents to communicate about a target they can't see, without giving them a communication channel.

## Setup

Two agents on a line, each a CTRNN with one neuron per colour plus one for proximity. The signaller stays in 0-1 and is
told the target colour. The receiver starts at 0.75 (signaller at 0.25) and has to stop at the landmark of that colour.
Landmarks are placed at random in 2-3, at least 0.3 apart. Each agent senses how close the other one is; the receiver
also senses each landmark within 0.2. A trial lasts 30 time units (300 steps, dt = 0.1).

Each sensor feeds one neuron: neuron 0 red, 1 green, 2 blue (3 colours only), then proximity. `--extra N` adds N
neurons with no sensor input after those. The last neuron drives the motor, so with no extra neurons that is the
proximity neuron. A genome holds both networks (36 genes with 2 colours, 56 with 3, without extra neurons).

**Fitness.** Mean score over every colour order x 3 random layouts x every target colour (12 trials with 2 colours,
54 with 3). The score is 1 on the target, falls to 0 at a distance of 0.5, and is 0 whenever the receiver is nearer
another landmark. It is averaged over the last 2 time units. Layouts are redrawn each generation.

**Evolution.** Population 200, 5000 generations, the top 5% are kept and mutated (Gaussian noise 0.1), no crossover.

**Testing.** The best genome of each run is tested on 100 held-out layouts with each target colour. Success means
ending within 0.1 of the target. A run is solved if success is at least 0.9 for every colour, both at dt = 0.1 and
dt = 0.01. The signal check repeats the test with the signaller's colour inputs set to 0.

## Running

```
python -m bots.evolve --colours 2 --seeds 0 1 2 3 4 5 6 7 8 9      # Stage A
python -m bots.evolve --colours 3 --seeds $(seq 0 19)              # Stage B
python -m bots.summarize 3col_n4
python -m pytest tests
```

Runs are saved to `results/<colours>col_n<neurons>/s<seed>_g<generations>/`.

On Colab (GPU runtime), in a new notebook:

```
!git clone https://github.com/migarak2908/Signalling-Bots
%cd Signalling-Bots
%run bots/evolve.py --colours 2 --seeds 0 1 2 3 4 5 6 7 8 9
%run bots/summarize.py 2col_n3
```
