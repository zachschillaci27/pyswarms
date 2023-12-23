# Import modules
import numpy as np
from manim import GREEN, RED, UP, Dot, ManimColor, Scene

# Import from pyswarms
from pyswarms.single import GlobalBestPSO


def objective_function(x: float | np.ndarray) -> float | np.ndarray:
    return (
        1 / 2 * (x[:, 0] ** 2 + x[:, 1] ** 2)
        - np.cos(2 * np.pi * x[:, 0])
        - np.cos(2 * np.pi * x[:, 1])
        + 2
    )


def run_pso() -> GlobalBestPSO:
    pso = GlobalBestPSO(
        n_particles=100,
        dimensions=2,
        options={"c1": 0.5, "c2": 0.3, "w": 0.9},
        init_pos=np.random.uniform(-5.12, 5.12, (100, 2)),
    )
    cost, pos = pso.optimize(objective_function, iters=15)
    return pso


def score_to_color(
    score: float, min_score: float, max_score: float
) -> ManimColor:
    # Normalize score between 0 and 1
    normalized_score = (score - min_score) / (max_score - min_score)
    # Linear interpolation between RED and GREEN based on normalized score
    return RED.interpolate(GREEN, normalized_score)


class PSOAnimation(Scene):
    def construct(self):
        pso = run_pso()

        # Particles
        particles = [
            Dot().move_to([p[0], p[1], 0]) for p in pso.swarm.position
        ]
        self.add(*particles)

        # Update positions in each iteration
        print(f"Number of iterations: {len(pso.pos_history)}")
        print(f"Positions: {np.array(pso.pos_history).shape}")
        print(f"Costs: {np.array(pso.cost_history).shape}")
        min_cost, max_cost = np.min(pso.cost_history), np.max(pso.cost_history)
        for swarm_positions in pso.pos_history:
            new_positions = [[p[0], p[1], 0] for p in swarm_positions]
            new_costs = objective_function(swarm_positions)

            self.play(
                *[
                    particle.animate.move_to(new_pos).set_color(
                        score_to_color(new_cost, min_cost, max_cost)
                    )
                    for particle, new_pos, new_cost in zip(
                        particles, new_positions, new_costs
                    )
                ],
                run_time=1,
            )

        self.wait()


# Run the animation
scene = PSOAnimation()
scene.render()
