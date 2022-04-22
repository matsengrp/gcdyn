from dataclasses import dataclass


@dataclass
class Parameters:
    def __init__(self, θ, μ, m, ρ):
        self.θ = θ
        self.μ = μ
        self.m = m
        self.ρ = ρ
