from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from enum import Enum, auto

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import psutil
import pygame
import torch

from src.model import (
    ALLY_SLOTS,
    ENEMY_SLOTS,
    OBSERVATION_DIM,
    PER_AGENT_FEATURES,
    PolicyAction,
    RAY_COUNT,
    CombatPolicyNet,
)
from src.training import EvolutionTrainer

# =============================================================================
# Window & Arena
# =============================================================================
WIN_W, WIN_H = 1280, 900
WALL_T = 50
FPS_CAP = 60
PHYSICS_STEP = 1 / 120
PHYSICS_HZ = round(1 / PHYSICS_STEP)
MAX_FRAME_TIME = 0.25

# =============================================================================
# Agent body
# =============================================================================
AGENT_RADIUS = 26
ORBIT_RADIUS = 36
FIST_RADIUS = 9
FIST_ANGLE_NORMAL = math.radians(35)
FIST_ANGLE_PARRY = math.radians(18)
ORBIT_PARRY_SCALE = 0.95

# =============================================================================
# Physics
# =============================================================================
AGENT_DRAG = 8.0
AGENT_WALK_SPEED = 260.0
AGENT_STOP_SPEED = 2.0

# =============================================================================
# Combat
# =============================================================================
MAX_HP = 100
MELEE_DAMAGE = 10
DEATH_HEAL_FRACTION = 0.5
SWING_DURATION = 0.18
SWING_COOLDOWN = 0.3
SWING_EXTEND = 14          # extra px fist extends at max reach (visual only)
HITBOX_OFFSET = 20         # px past agent edge the hitbox center sits
HITBOX_RADIUS = 15
HIT_FLASH_DURATION = 0.12
HEAL_PARTICLE_TTL = 0.42
HEAL_PARTICLE_MIN_COUNT = 7
HEAL_PARTICLE_MAX_COUNT = 16
HEAL_PARTICLE_GRAVITY = -95.0
PARRY_WINDOW = 0.4
PARRY_WEAKNESS_DURATION = 0.8
PARRY_WEAKNESS_SPEED_SCALE = 0.25
PARRY_COOLDOWN = 4.0

# Stun (applied to attacker when their swing is parried)
STUN_DURATION = 0.6
STAR_COUNT = 3
STAR_ORBIT_RADIUS = 20
STAR_ORBIT_FLATTEN = 0.45      # vertical squish so it looks like a flat orbit above head
STAR_OUTER_R = 6
STAR_INNER_R = 2.6
STAR_ROTATION_SPEED = 4.5      # rad/s
STAR_HOVER_OFFSET = 16         # px above the agent center

# =============================================================================
# Neural team matches
# =============================================================================
MODE_HUMAN_VS_AI = "human_vs_ai"
MODE_AI_VS_AI = "ai_vs_ai"
AI_ROUND_SECONDS = 45.0
NO_COMBAT_TIMEOUT = 8.0
AGENT_COLLISION_ITERATIONS = 6
ENGAGEMENT_REWARD_RANGE = 250.0
ENGAGEMENT_DISTANCE_REWARD_RATE = 0.18
ENGAGEMENT_FACING_REWARD_RATE = 0.03
WALL_PROXIMITY_PENALTY_RANGE = 160.0
WALL_PROXIMITY_PENALTY_RATE = 0.045


@dataclass(frozen=True)
class TeamSetup:
    name: str
    team_sizes: tuple[int, ...]

    @property
    def team_count(self) -> int:
        return len(self.team_sizes)


DEFAULT_TEAM_SETUP = TeamSetup("3v3v3", (3, 3, 3))
AI_TEAM_SETUPS = (
    DEFAULT_TEAM_SETUP,
    TeamSetup("4v4", (4, 4)),
)

# =============================================================================
# Compute tuning
# =============================================================================
AI_DECISION_HZ = 30
AI_DECISION_STEPS = max(1, round(PHYSICS_HZ / AI_DECISION_HZ))
AI_ATTACK_THRESHOLD = 0.52
AI_PARRY_THRESHOLD = 0.68
FAST_FORWARD_STEPS_PER_FRAME = 300
FAST_FORWARD_FPS_CAP = 10
CPU_THROTTLE_LIMIT = 75       # percent; pause simulation above this
CPU_THROTTLE_SLEEP_MS = 100   # ms to wait each frame while over limit

# =============================================================================
# Colors
# =============================================================================
C_FLOOR = (26, 26, 46)
C_WALL = (58, 72, 110)
C_WALL_EDGE = (90, 110, 160)
C_PLAYER_BODY = (55, 135, 215)
C_PLAYER_EDGE = (30, 80, 155)
C_PLAYER_FIST = (95, 185, 255)
C_BOT_BODY = (195, 50, 50)
C_BOT_EDGE = (120, 25, 25)
C_BOT_FIST = (255, 125, 45)
C_SHIELD = (86, 210, 232)
C_SHIELD_EDGE = (18, 82, 132)
C_SHIELD_SHADE = (42, 145, 188)
C_SHIELD_HIGHLIGHT = (226, 252, 255)
C_STAR_FILL = (255, 225, 70)
C_HEAL_PARTICLE = (92, 255, 148)
C_HEAL_PARTICLE_CORE = (228, 255, 216)
C_HIT_FLASH = (255, 255, 255)
C_HP_BG = (35, 35, 35)
C_HP_FILL = (55, 195, 75)
C_HP_LOW = (195, 55, 55)
C_CD_BG = (28, 28, 28)
C_CD_SWING = (180, 140, 55)
C_CD_SWING_ACTIVE = (255, 200, 80)
C_CD_PARRY = (75, 155, 195)
C_CD_PARRY_ACTIVE = (120, 220, 255)
C_TEXT = (228, 234, 242)
C_TEXT_MUTED = (151, 163, 181)

TEAM_COLORS = {
    1: (55, 135, 215),
    2: (195, 50, 50),
    3: (70, 175, 105),
    4: (210, 160, 55),
    5: (168, 92, 220),
}
TEAM_EDGE_COLORS = {
    1: (30, 80, 155),
    2: (120, 25, 25),
    3: (32, 105, 62),
    4: (132, 92, 18),
    5: (92, 42, 145),
}
TEAM_FIST_COLORS = {
    1: (95, 185, 255),
    2: (255, 125, 45),
    3: (120, 235, 150),
    4: (255, 222, 92),
    5: (214, 150, 255),
}


# =============================================================================
# Shield template (centered at origin, rounded face toward +x).
# Sourced from shield-minimalistic-svgrepo-com.svg, rotated 90° CW so that
# Rotated by look_angle so the rounded face points toward the guard direction.
# =============================================================================
_SHIELD_OUTER = [
    (21.0, -8.0),
    (22.5, -4.0),
    (22.5, 4.0),
    (21.0, 8.0),
    (15.5, 14.5),
    (6.0, 18.5),
    (-4.5, 16.0),
    (-15.5, 7.0),
    (-19.0, 0.0),
    (-15.5, -7.0),
    (-4.5, -16.0),
    (6.0, -18.5),
    (15.5, -14.5),
]
# Inner highlight polygon (smaller, slightly offset toward forward face)
_SHIELD_INNER = [
    (18.0, -6.0),
    (19.2, -2.5),
    (19.2, 2.5),
    (18.0, 6.0),
    (12.5, 11.5),
    (5.0, 14.5),
    (-3.0, 12.5),
    (-11.5, 5.0),
    (-14.0, 0.0),
    (-11.5, -5.0),
    (-3.0, -12.5),
    (5.0, -14.5),
    (12.5, -11.5),
]
_SHIELD_SHADE_POLY = [
    (-2.0, -12.0),
    (5.0, -14.0),
    (13.0, -11.0),
    (17.5, -5.0),
    (12.0, 0.0),
    (-11.0, 0.0),
    (-13.0, -4.0),
]
_SHIELD_HIGHLIGHT_LINE = [
    (13.5, 7.0),
    (8.0, 11.0),
    (1.0, 12.2),
    (-6.5, 8.0),
]


# =============================================================================
# Helpers
# =============================================================================
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def unit(angle: float) -> pygame.Vector2:
    return pygame.Vector2(math.cos(angle), math.sin(angle))


def _star_points(cx: float, cy: float, outer_r: float, inner_r: float, rotation: float = 0.0):
    pts = []
    for i in range(10):
        angle = rotation + i * math.pi / 5 - math.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
    return pts


# =============================================================================
# Enums
# =============================================================================
class SwingState(Enum):
    IDLE = auto()
    SWINGING = auto()
    ON_COOLDOWN = auto()


class ParryState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    WEAKNESS = auto()
    ON_COOLDOWN = auto()


@dataclass
class HealParticle:
    pos: pygame.Vector2
    vel: pygame.Vector2
    age: float
    ttl: float
    radius: float


# =============================================================================
# Arena
# =============================================================================
class Arena:
    def __init__(self) -> None:
        self.floor_rect = pygame.Rect(WALL_T, WALL_T, WIN_W - 2 * WALL_T, WIN_H - 2 * WALL_T)
        self.wall_rects = [
            pygame.Rect(0, 0, WIN_W, WALL_T),
            pygame.Rect(0, WIN_H - WALL_T, WIN_W, WALL_T),
            pygame.Rect(0, 0, WALL_T, WIN_H),
            pygame.Rect(WIN_W - WALL_T, 0, WALL_T, WIN_H),
        ]

    def draw(self, surface: pygame.Surface) -> None:
        for r in self.wall_rects:
            pygame.draw.rect(surface, C_WALL, r)
            pygame.draw.rect(surface, C_WALL_EDGE, r, 3)

    def clamp_circle(self, pos: pygame.Vector2, vel: pygame.Vector2, radius: int) -> None:
        fr = self.floor_rect
        if pos.x - radius < fr.left:
            pos.x = fr.left + radius
            vel.x = max(0.0, vel.x)
        elif pos.x + radius > fr.right:
            pos.x = fr.right - radius
            vel.x = min(0.0, vel.x)
        if pos.y - radius < fr.top:
            pos.y = fr.top + radius
            vel.y = max(0.0, vel.y)
        elif pos.y + radius > fr.bottom:
            pos.y = fr.bottom - radius
            vel.y = min(0.0, vel.y)


# =============================================================================
# Agent base
# =============================================================================
class Agent:
    body_color: tuple = (200, 200, 200)
    edge_color: tuple = (100, 100, 100)
    fist_color: tuple = (220, 220, 220)

    def __init__(self, pos: pygame.Vector2, team_id: int | None = None) -> None:
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.look_angle: float = 0.0
        self.team_id = team_id
        self.label: str = "Agent"

        self.hp: int = MAX_HP

        self.swing_state = SwingState.IDLE
        self.swing_timer: float = 0.0
        self._fire_hitbox: bool = False
        self._active_swing_side: int = -1
        self._next_swing_side: int = -1

        self.parry_state = ParryState.IDLE
        self.parry_timer: float = 0.0

        self.stun_timer: float = 0.0
        self.stars_phase: float = 0.0

        self.hit_flash_timer: float = 0.0
        self.heal_particles: list[HealParticle] = []
        self.damage_dealt: float = 0.0
        self.damage_taken: float = 0.0
        self.kills: int = 0
        self.deaths: int = 0
        self.death_heal_received: float = 0.0
        self.damage_sources: dict[Agent, float] = {}
        self.hits_landed: int = 0
        self.swings_started: int = 0
        self.parries_started: int = 0
        self.parries_landed: int = 0
        self.times_parried: int = 0
        self.round_survival_time: float = 0.0
        self.engagement_reward: float = 0.0
        self.facing_reward: float = 0.0
        self.wall_proximity_penalty: float = 0.0

    # -------------------------------------------------------------------------
    # Combat actions
    # -------------------------------------------------------------------------
    def is_alive(self) -> bool:
        return self.hp > 0

    def is_stunned(self) -> bool:
        return self.stun_timer > 0.0

    def can_swing(self) -> bool:
        return (
            self.swing_state == SwingState.IDLE
            and not self.is_weak()
            and not self.is_stunned()
        )

    def can_parry(self) -> bool:
        return self.parry_state == ParryState.IDLE and not self.is_stunned()

    def is_parrying(self) -> bool:
        return self.parry_state == ParryState.ACTIVE

    def is_weak(self) -> bool:
        return self.parry_state in (ParryState.ACTIVE, ParryState.WEAKNESS)

    def start_swing(self) -> None:
        if self.can_swing():
            self.swing_state = SwingState.SWINGING
            self.swing_timer = 0.0
            self._fire_hitbox = True
            self._active_swing_side = self._next_swing_side
            self._next_swing_side *= -1
            self.swings_started += 1

    def start_parry(self) -> None:
        if self.can_parry():
            self.parry_state = ParryState.ACTIVE
            self.parry_timer = PARRY_WINDOW
            self.parries_started += 1

    def clear_parry_weakness(self) -> None:
        if self.parry_state in (ParryState.ACTIVE, ParryState.WEAKNESS):
            self.parry_state = ParryState.IDLE
            self.parry_timer = 0.0

    def apply_stun(self) -> None:
        """Called on the attacker when their swing is parried."""
        self.stun_timer = STUN_DURATION
        self.stars_phase = 0.0
        # Cancel any in-progress swing
        self.swing_state = SwingState.IDLE
        self.swing_timer = 0.0
        self._fire_hitbox = False
        self.clear_parry_weakness()
        self.vel.update(0, 0)

    def spawn_heal_effect(self, amount: float) -> None:
        if amount <= 0:
            return

        count = max(
            HEAL_PARTICLE_MIN_COUNT,
            min(HEAL_PARTICLE_MAX_COUNT, round(amount / 3)),
        )
        phase = (self.death_heal_received * 0.37) % (2 * math.pi)
        for i in range(count):
            t = i / max(1, count - 1)
            angle = phase + i * (2 * math.pi / count)
            direction = unit(angle)
            speed = lerp(70.0, 145.0, t)
            pos = self.pos + direction * (AGENT_RADIUS * 0.45)
            vel = direction * speed + pygame.Vector2(0, -35)
            self.heal_particles.append(
                HealParticle(
                    pos=pygame.Vector2(pos),
                    vel=vel,
                    age=0.0,
                    ttl=HEAL_PARTICLE_TTL * lerp(0.82, 1.12, 1.0 - abs(0.5 - t) * 2),
                    radius=lerp(2.4, 4.2, 1.0 - t),
                )
            )

    # -------------------------------------------------------------------------
    # Tick helpers
    # -------------------------------------------------------------------------
    def _tick_swing(self, dt: float) -> None:
        if self.swing_state == SwingState.SWINGING:
            self.swing_timer += dt
            if self.swing_timer >= SWING_DURATION:
                self.swing_state = SwingState.ON_COOLDOWN
                self.swing_timer = SWING_COOLDOWN
        elif self.swing_state == SwingState.ON_COOLDOWN:
            self.swing_timer -= dt
            if self.swing_timer <= 0.0:
                self.swing_state = SwingState.IDLE
                self.swing_timer = 0.0

    def _tick_parry(self, dt: float) -> None:
        if self.parry_state == ParryState.ACTIVE:
            self.parry_timer -= dt
            if self.parry_timer <= 0.0:
                weakness_time = max(0.0, PARRY_WEAKNESS_DURATION - PARRY_WINDOW)
                if weakness_time > 0.0:
                    self.parry_state = ParryState.WEAKNESS
                    self.parry_timer = weakness_time
                else:
                    self.parry_state = ParryState.ON_COOLDOWN
                    self.parry_timer = PARRY_COOLDOWN
        elif self.parry_state == ParryState.WEAKNESS:
            self.parry_timer -= dt
            if self.parry_timer <= 0.0:
                self.parry_state = ParryState.ON_COOLDOWN
                self.parry_timer = PARRY_COOLDOWN
        elif self.parry_state == ParryState.ON_COOLDOWN:
            self.parry_timer -= dt
            if self.parry_timer <= 0.0:
                self.parry_state = ParryState.IDLE
                self.parry_timer = 0.0

    def _tick_hit_flash(self, dt: float) -> None:
        if self.hit_flash_timer > 0.0:
            self.hit_flash_timer -= dt

    def _tick_heal_particles(self, dt: float) -> None:
        if not self.heal_particles:
            return

        survivors: list[HealParticle] = []
        drag = max(0.0, 1.0 - 5.5 * dt)
        for particle in self.heal_particles:
            particle.age += dt
            if particle.age >= particle.ttl:
                continue
            particle.vel.y += HEAL_PARTICLE_GRAVITY * dt
            particle.pos += particle.vel * dt
            particle.vel *= drag
            survivors.append(particle)
        self.heal_particles = survivors

    def _tick_stun(self, dt: float) -> None:
        if self.stun_timer > 0.0:
            self.stun_timer -= dt
            if self.stun_timer < 0.0:
                self.stun_timer = 0.0
            self.stars_phase = (self.stars_phase + STAR_ROTATION_SPEED * dt) % (2 * math.pi)

    # -------------------------------------------------------------------------
    # Physics
    # -------------------------------------------------------------------------
    def _apply_physics(
        self,
        direction: pygame.Vector2,
        terminal_speed: float,
        dt: float,
        arena: Arena,
    ) -> None:
        if self.is_stunned():
            self.vel.update(0, 0)
            arena.clamp_circle(self.pos, self.vel, AGENT_RADIUS)
            return

        if self.parry_state == ParryState.ACTIVE:
            self.vel.update(0, 0)
            arena.clamp_circle(self.pos, self.vel, AGENT_RADIUS)
            return
        if self.parry_state == ParryState.WEAKNESS:
            terminal_speed *= PARRY_WEAKNESS_SPEED_SCALE

        direction_len_sq = direction.length_squared()
        if direction_len_sq > 0:
            target_vel = direction.normalize() * terminal_speed
        else:
            target_vel = pygame.Vector2(0, 0)

        if AGENT_DRAG > 0.0:
            old_vel = pygame.Vector2(self.vel)
            decay = math.exp(-AGENT_DRAG * dt)
            self.vel = target_vel + (old_vel - target_vel) * decay
            self.pos += target_vel * dt + (old_vel - target_vel) * ((1.0 - decay) / AGENT_DRAG)
        else:
            self.vel = pygame.Vector2(target_vel)
            self.pos += self.vel * dt

        if direction_len_sq == 0 and self.vel.length_squared() < AGENT_STOP_SPEED ** 2:
            self.vel.update(0, 0)

        arena.clamp_circle(self.pos, self.vel, AGENT_RADIUS)

    # -------------------------------------------------------------------------
    # Fist positions
    # -------------------------------------------------------------------------
    def fist_positions(self) -> tuple[pygame.Vector2, pygame.Vector2]:
        if self.parry_state == ParryState.ACTIVE:
            r = ORBIT_RADIUS * ORBIT_PARRY_SCALE
            lf = self.pos + unit(self.look_angle + FIST_ANGLE_PARRY) * r
            rf = self.pos + unit(self.look_angle - FIST_ANGLE_PARRY) * r
            return lf, rf

        lf = self.pos + unit(self.look_angle + FIST_ANGLE_NORMAL) * ORBIT_RADIUS
        rf = self.pos + unit(self.look_angle - FIST_ANGLE_NORMAL) * ORBIT_RADIUS

        if self.swing_state == SwingState.SWINGING:
            t = self.swing_timer / SWING_DURATION
            t_half = 0.5
            side = self._active_swing_side
            if t <= t_half:
                phase = t / t_half
                angle_off = lerp(side * FIST_ANGLE_NORMAL, 0.0, phase)
                r = lerp(ORBIT_RADIUS, ORBIT_RADIUS + SWING_EXTEND, phase)
            else:
                phase = (t - t_half) / (1.0 - t_half)
                angle_off = lerp(0.0, side * FIST_ANGLE_NORMAL, phase)
                r = lerp(ORBIT_RADIUS + SWING_EXTEND, ORBIT_RADIUS, phase)
            fist_pos = self.pos + unit(self.look_angle + angle_off) * r
            if side > 0:
                lf = fist_pos
            else:
                rf = fist_pos

        return lf, rf

    def hitbox_position(self) -> pygame.Vector2:
        # Swing hit detection is fixed to the agent, not to either animated fist.
        return self.pos + unit(self.look_angle) * (AGENT_RADIUS + HITBOX_OFFSET)

    # -------------------------------------------------------------------------
    # Draw
    # -------------------------------------------------------------------------
    def draw(self, surface: pygame.Surface) -> None:
        if not self.is_alive():
            return

        ix, iy = round(self.pos.x), round(self.pos.y)
        flash = self.hit_flash_timer > 0.0
        body_color = C_HIT_FLASH if flash else self.body_color
        edge_color = C_HIT_FLASH if flash else self.edge_color
        fist_color = C_HIT_FLASH if flash else self.fist_color

        # Body
        pygame.draw.circle(surface, body_color, (ix, iy), AGENT_RADIUS)
        pygame.draw.circle(surface, edge_color, (ix, iy), AGENT_RADIUS, 2)

        # Fists (drawn before shield so the shield can sit cleanly on top)
        lf, rf = self.fist_positions()
        for fp in (lf, rf):
            pygame.draw.circle(surface, fist_color, (round(fp.x), round(fp.y)), FIST_RADIUS)
            pygame.draw.circle(surface, (0, 0, 0), (round(fp.x), round(fp.y)), FIST_RADIUS, 1)

        # Shield: offset forward and rotated to look direction
        if self.parry_state == ParryState.ACTIVE:
            self._draw_shield(surface)

    def _draw_shield(self, surface: pygame.Surface) -> None:
        cos_a = math.cos(self.look_angle)
        sin_a = math.sin(self.look_angle)
        offset = unit(self.look_angle) * (AGENT_RADIUS * 0.5)
        cx, cy = self.pos.x + offset.x, self.pos.y + offset.y

        def transform(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
            return [
                (cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a)
                for x, y in points
            ]

        outer = transform(_SHIELD_OUTER)
        inner = transform(_SHIELD_INNER)
        shade = transform(_SHIELD_SHADE_POLY)
        highlight = transform(_SHIELD_HIGHLIGHT_LINE)

        pygame.draw.polygon(surface, C_SHIELD_EDGE, outer)
        pygame.draw.polygon(surface, C_SHIELD, inner)
        pygame.draw.polygon(surface, C_SHIELD_SHADE, shade)
        if len(highlight) > 1:
            pygame.draw.lines(surface, C_SHIELD_HIGHLIGHT, False, highlight, 2)
        pygame.draw.polygon(surface, C_SHIELD_EDGE, outer, 2)

    def draw_stun_stars(self, surface: pygame.Surface) -> None:
        if not self.is_stunned():
            return
        cx = self.pos.x
        cy = self.pos.y - AGENT_RADIUS - STAR_HOVER_OFFSET
        for i in range(STAR_COUNT):
            a = self.stars_phase + i * (2 * math.pi / STAR_COUNT)
            sx = cx + math.cos(a) * STAR_ORBIT_RADIUS
            sy = cy + math.sin(a) * STAR_ORBIT_RADIUS * STAR_ORBIT_FLATTEN
            verts = _star_points(sx, sy, STAR_OUTER_R, STAR_INNER_R, rotation=a * 0.5)
            pygame.draw.polygon(surface, C_STAR_FILL, verts)

    def draw_heal_particles(self, surface: pygame.Surface) -> None:
        for particle in self.heal_particles:
            life = 1.0 - particle.age / particle.ttl
            if life <= 0.0:
                continue

            radius = max(1, round(particle.radius * life))
            size = radius * 4
            center = size // 2
            alpha = round(210 * life)
            glow = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(
                glow,
                (*C_HEAL_PARTICLE, max(0, alpha // 3)),
                (center, center),
                radius * 2,
            )
            pygame.draw.circle(
                glow,
                (*C_HEAL_PARTICLE, alpha),
                (center, center),
                radius,
            )
            pygame.draw.circle(
                glow,
                (*C_HEAL_PARTICLE_CORE, min(255, alpha + 30)),
                (center, center),
                max(1, radius // 2),
            )
            surface.blit(glow, (round(particle.pos.x) - center, round(particle.pos.y) - center))


# =============================================================================
# PlayerAgent
# =============================================================================
class PlayerAgent(Agent):
    body_color = C_PLAYER_BODY
    edge_color = C_PLAYER_EDGE
    fist_color = C_PLAYER_FIST

    def __init__(self, pos: pygame.Vector2, team_id: int = 1) -> None:
        super().__init__(pos, team_id)
        self.label = "Player"
        self._swing_held = False
        self._parry_requested = False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self._swing_held = True
            elif event.button == 3:
                self._parry_requested = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self._swing_held = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                self._parry_requested = True

    def update(self, dt: float, arena: Arena) -> None:
        # Held attack input keeps retrying, so attacks fire immediately off cooldown.
        if self._swing_held:
            self.start_swing()
        if self._parry_requested:
            self.start_parry()
            self._parry_requested = False

        # Look direction from mouse
        mx, my = pygame.mouse.get_pos()
        self.look_angle = math.atan2(my - self.pos.y, mx - self.pos.x)

        # Movement
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        dy = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - (keys[pygame.K_w] or keys[pygame.K_UP])
        direction = pygame.Vector2(dx, dy)
        self._apply_physics(direction, AGENT_WALK_SPEED, dt, arena)

        self._tick_swing(dt)
        self._tick_parry(dt)
        self._tick_stun(dt)
        self._tick_hit_flash(dt)
        self._tick_heal_particles(dt)


# =============================================================================
# Neural observations and agents
# =============================================================================
def _team_color(team_id: int | None) -> tuple[int, int, int]:
    return TEAM_COLORS.get(team_id or 0, C_BOT_BODY)


def _team_edge_color(team_id: int | None) -> tuple[int, int, int]:
    return TEAM_EDGE_COLORS.get(team_id or 0, C_BOT_EDGE)


def _team_fist_color(team_id: int | None) -> tuple[int, int, int]:
    return TEAM_FIST_COLORS.get(team_id or 0, C_BOT_FIST)


def _rotate(vec: pygame.Vector2, angle: float) -> pygame.Vector2:
    c = math.cos(angle)
    s = math.sin(angle)
    return pygame.Vector2(vec.x * c - vec.y * s, vec.x * s + vec.y * c)


def _world_to_local(vec: pygame.Vector2, look_angle: float) -> pygame.Vector2:
    return _rotate(vec, -look_angle)


def _local_to_world(vec: pygame.Vector2, look_angle: float) -> pygame.Vector2:
    return _rotate(vec, look_angle)


def _norm_timer(timer: float, full: float) -> float:
    if full <= 0.0:
        return 0.0
    return max(0.0, min(1.0, timer / full))


def _nearest_enemy(agent: Agent, agents: list[Agent]) -> Agent | None:
    enemies = [
        other for other in agents
        if other is not agent
        and other.is_alive()
        and agent.team_id is not None
        and other.team_id != agent.team_id
    ]
    if not enemies:
        return None
    return min(enemies, key=lambda other: (other.pos - agent.pos).length_squared())


def _tick_engagement_rewards(agent: Agent, agents: list[Agent], dt: float) -> None:
    if not agent.is_alive() or agent.team_id is None:
        return

    nearby_facing = 0.0
    nearest_dist: float | None = None
    facing = unit(agent.look_angle)
    for other in agents:
        if other is agent or not other.is_alive() or other.team_id == agent.team_id:
            continue

        diff = other.pos - agent.pos
        dist = diff.length()
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist
        if 1e-6 < dist <= ENGAGEMENT_REWARD_RANGE:
            nearby_facing = max(nearby_facing, facing.dot(diff / dist))

    if nearest_dist is not None and nearest_dist <= ENGAGEMENT_REWARD_RANGE:
        closeness = 1.0 - nearest_dist / ENGAGEMENT_REWARD_RANGE
        agent.engagement_reward += closeness * ENGAGEMENT_DISTANCE_REWARD_RATE * dt

    if nearby_facing > 0.0:
        agent.facing_reward += nearby_facing * ENGAGEMENT_FACING_REWARD_RATE * dt


def _tick_wall_proximity_penalty(agent: Agent, arena: Arena, dt: float) -> None:
    if not agent.is_alive():
        return

    fr = arena.floor_rect
    nearest_wall = min(
        agent.pos.x - AGENT_RADIUS - fr.left,
        fr.right - (agent.pos.x + AGENT_RADIUS),
        agent.pos.y - AGENT_RADIUS - fr.top,
        fr.bottom - (agent.pos.y + AGENT_RADIUS),
    )
    nearest_wall = max(0.0, nearest_wall)
    if nearest_wall >= WALL_PROXIMITY_PENALTY_RANGE:
        return

    closeness = 1.0 - nearest_wall / WALL_PROXIMITY_PENALTY_RANGE
    agent.wall_proximity_penalty += closeness * WALL_PROXIMITY_PENALTY_RATE * dt


def _ray_distance_to_arena(pos: pygame.Vector2, angle: float, arena: Arena, max_dist: float = 300.0) -> float:
    dx = math.cos(angle)
    dy = math.sin(angle)
    candidates: list[float] = []
    fr = arena.floor_rect

    if dx > 1e-6:
        candidates.append((fr.right - pos.x) / dx)
    elif dx < -1e-6:
        candidates.append((fr.left - pos.x) / dx)

    if dy > 1e-6:
        candidates.append((fr.bottom - pos.y) / dy)
    elif dy < -1e-6:
        candidates.append((fr.top - pos.y) / dy)

    positive = [value for value in candidates if value >= 0.0]
    if not positive:
        return 1.0
    return max(0.0, min(min(positive), max_dist)) / max_dist


def _arena_rays(agent: Agent, arena: Arena) -> list[float]:
    return [
        _ray_distance_to_arena(agent.pos, agent.look_angle + i * (2 * math.pi / RAY_COUNT), arena)
        for i in range(RAY_COUNT)
    ]


def _agent_features(subject: Agent, other: Agent) -> list[float]:
    rel = other.pos - subject.pos
    rel_local = _world_to_local(rel, subject.look_angle)
    rel_vel_local = _world_to_local(other.vel - subject.vel, subject.look_angle)
    dist = rel.length()
    if dist > 1e-6:
        rel_cos = rel_local.x / dist
        rel_sin = rel_local.y / dist
    else:
        rel_cos = 1.0
        rel_sin = 0.0

    direction_to_subject = subject.pos - other.pos
    if direction_to_subject.length_squared() > 0:
        facing_subject = unit(other.look_angle).dot(direction_to_subject.normalize())
    else:
        facing_subject = 0.0

    state_pressure = 1.0 if other.is_weak() or other.is_stunned() else 0.0
    return [
        1.0,
        max(-1.0, min(1.0, rel_cos)),
        max(-1.0, min(1.0, rel_sin)),
        max(0.0, min(1.0, dist / math.hypot(WIN_W, WIN_H))),
        max(-1.0, min(1.0, rel_vel_local.x / AGENT_WALK_SPEED)),
        max(-1.0, min(1.0, rel_vel_local.y / AGENT_WALK_SPEED)),
        max(0.0, min(1.0, other.hp / MAX_HP)),
        1.0 if other.swing_state == SwingState.SWINGING else 0.0,
        1.0 if other.can_swing() else 0.0,
        1.0 if other.is_parrying() else 0.0,
        state_pressure,
        max(-1.0, min(1.0, facing_subject)),
    ]


def _empty_agent_features() -> list[float]:
    return [0.0] * PER_AGENT_FEATURES


def build_observation(agent: Agent, agents: list[Agent], arena: Arena) -> list[float]:
    vel_local = _world_to_local(agent.vel, agent.look_angle)
    enemies = [
        other for other in agents
        if other is not agent
        and other.is_alive()
        and agent.team_id is not None
        and other.team_id != agent.team_id
    ]
    allies = [
        other for other in agents
        if other is not agent
        and other.is_alive()
        and agent.team_id is not None
        and other.team_id == agent.team_id
    ]
    enemies.sort(key=lambda other: (other.pos - agent.pos).length_squared())
    allies.sort(key=lambda other: (other.pos - agent.pos).length_squared())

    enemy_relation = 0.0
    if len(enemies) >= 2:
        enemy_relation = 1.0 if enemies[0].team_id == enemies[1].team_id else -1.0

    self_features = [
        max(0.0, min(1.0, agent.hp / MAX_HP)),
        max(-1.0, min(1.0, vel_local.x / AGENT_WALK_SPEED)),
        max(-1.0, min(1.0, vel_local.y / AGENT_WALK_SPEED)),
        math.cos(agent.look_angle),
        math.sin(agent.look_angle),
        1.0 if agent.can_swing() else 0.0,
        1.0 if agent.swing_state == SwingState.SWINGING else 0.0,
        _norm_timer(agent.swing_timer, SWING_COOLDOWN),
        1.0 if agent.can_parry() else 0.0,
        1.0 if agent.parry_state == ParryState.ACTIVE else 0.0,
        1.0 if agent.parry_state == ParryState.WEAKNESS else 0.0,
        _norm_timer(agent.parry_timer, PARRY_COOLDOWN),
        _norm_timer(agent.stun_timer, STUN_DURATION),
        enemy_relation,
        1.0,
    ]

    features = self_features + _arena_rays(agent, arena)
    for slot in range(ENEMY_SLOTS):
        features.extend(_agent_features(agent, enemies[slot]) if slot < len(enemies) else _empty_agent_features())
    for slot in range(ALLY_SLOTS):
        features.extend(_agent_features(agent, allies[slot]) if slot < len(allies) else _empty_agent_features())

    if len(features) < OBSERVATION_DIM:
        features.extend([0.0] * (OBSERVATION_DIM - len(features)))
    return features[:OBSERVATION_DIM]


NEUTRAL_POLICY_ACTION = PolicyAction(
    move_x=0.0,
    move_y=0.0,
    aim_x=0.0,
    aim_y=0.0,
    attack=False,
    parry=False,
    attack_score=0.0,
    parry_score=0.0,
)


class NeuralAgent(Agent):
    def __init__(
        self,
        pos: pygame.Vector2,
        team_id: int,
        policy: CombatPolicyNet,
        *,
        genome_id: int | None = None,
        label: str = "AI",
    ) -> None:
        super().__init__(pos, team_id)
        self.policy = policy
        self.genome_id = genome_id
        self.label = label
        self.body_color = _team_color(team_id)
        self.edge_color = _team_edge_color(team_id)
        self.fist_color = _team_fist_color(team_id)
        self.policy_action = NEUTRAL_POLICY_ACTION

    def set_policy_action(self, action: PolicyAction) -> None:
        self.policy_action = action

    def update(self, dt: float, arena: Arena, agents: list[Agent]) -> None:
        if not self.is_alive():
            return

        action = self.policy_action
        nearest_enemy = _nearest_enemy(self, agents)
        decision_angle = self.look_angle

        aim_local = pygame.Vector2(action.aim_x, action.aim_y)
        if aim_local.length_squared() > 0.01:
            aim_world = _local_to_world(aim_local, decision_angle)
        elif nearest_enemy is not None:
            aim_world = nearest_enemy.pos - self.pos
        else:
            aim_world = unit(decision_angle)

        if aim_world.length_squared() > 1e-6:
            self.look_angle = math.atan2(aim_world.y, aim_world.x)

        if action.parry and self.can_parry():
            self.start_parry()
        elif action.attack:
            self.start_swing()

        move_local = pygame.Vector2(action.move_x, action.move_y)
        if move_local.length_squared() > 0.09:
            move_world = _local_to_world(move_local, decision_angle)
        elif nearest_enemy is not None:
            move_world = nearest_enemy.pos - self.pos
        else:
            move_world = pygame.Vector2(0, 0)

        self._apply_physics(move_world, AGENT_WALK_SPEED, dt, arena)
        self._tick_swing(dt)
        self._tick_parry(dt)
        self._tick_stun(dt)
        self._tick_hit_flash(dt)
        self._tick_heal_particles(dt)


def _actions_from_policy_output(output: torch.Tensor) -> list[PolicyAction]:
    if output.ndim == 1:
        output = output.unsqueeze(0)

    move = torch.tanh(output[:, 0:2]).cpu()
    aim = torch.tanh(output[:, 2:4]).cpu()
    attack_scores = torch.sigmoid(output[:, 4]).cpu()
    parry_scores = torch.sigmoid(output[:, 5]).cpu()

    actions: list[PolicyAction] = []
    for index in range(output.shape[0]):
        attack_score = float(attack_scores[index].item())
        parry_score = float(parry_scores[index].item())
        actions.append(
            PolicyAction(
                move_x=float(move[index, 0].item()),
                move_y=float(move[index, 1].item()),
                aim_x=float(aim[index, 0].item()),
                aim_y=float(aim[index, 1].item()),
                attack=attack_score >= AI_ATTACK_THRESHOLD,
                parry=parry_score >= AI_PARRY_THRESHOLD,
                attack_score=attack_score,
                parry_score=parry_score,
            )
        )
    return actions


def _refresh_neural_policy_actions(agents: list[Agent], arena: Arena) -> tuple[int, int]:
    policy_groups: dict[int, tuple[CombatPolicyNet, list[NeuralAgent], list[list[float]]]] = {}

    for agent in agents:
        if not isinstance(agent, NeuralAgent) or not agent.is_alive():
            continue

        key = id(agent.policy)
        if key not in policy_groups:
            policy_groups[key] = (agent.policy, [], [])
        _, group_agents, observations = policy_groups[key]
        group_agents.append(agent)
        observations.append(build_observation(agent, agents, arena))

    decision_count = 0
    for policy, group_agents, observations in policy_groups.values():
        if not observations:
            continue

        obs_tensor = torch.tensor(observations, dtype=torch.float32)
        with torch.inference_mode():
            actions = _actions_from_policy_output(policy(obs_tensor))

        for agent, action in zip(group_agents, actions):
            agent.set_policy_action(action)
        decision_count += len(group_agents)

    return decision_count, len(policy_groups)


# =============================================================================
# Combat resolution
# =============================================================================
def _distribute_death_heal(defender: Agent) -> None:
    total_damage = sum(max(0, damage) for damage in defender.damage_sources.values())
    if total_damage <= 0:
        return

    heal_pool = MAX_HP * DEATH_HEAL_FRACTION
    for source, damage in defender.damage_sources.items():
        if damage <= 0 or not source.is_alive():
            continue
        heal = int(round(heal_pool * (damage / total_damage)))
        if heal <= 0:
            continue
        before = source.hp
        source.hp = min(MAX_HP, source.hp + heal)
        actual_heal = source.hp - before
        source.death_heal_received += actual_heal
        source.spawn_heal_effect(actual_heal)


@dataclass(frozen=True)
class AttackPlan:
    attacker: Agent
    parried_by: Agent | None
    defenders: tuple[Agent, ...]


def _can_attack_target(attacker: Agent, defender: Agent) -> bool:
    return (
        defender is not attacker
        and defender.is_alive()
        and not (
            attacker.team_id is not None
            and attacker.team_id == defender.team_id
        )
    )


def _build_attack_plan(attacker: Agent, agents: list[Agent]) -> AttackPlan:
    hitbox_pos = attacker.hitbox_position()
    hit_range_sq = (HITBOX_RADIUS + AGENT_RADIUS) ** 2
    parry_targets: list[Agent] = []
    hit_targets: list[Agent] = []

    for defender in agents:
        if not _can_attack_target(attacker, defender):
            continue
        if (defender.pos - hitbox_pos).length_squared() > hit_range_sq:
            continue
        if defender.is_parrying():
            parry_targets.append(defender)
        else:
            hit_targets.append(defender)

    if parry_targets:
        parried_by = min(
            parry_targets,
            key=lambda defender: (defender.pos - hitbox_pos).length_squared(),
        )
        return AttackPlan(attacker, parried_by, ())

    return AttackPlan(attacker, None, tuple(hit_targets))


def _kill_credit_source(
    defender: Agent,
    attackers: list[Agent],
    agent_order: dict[Agent, int],
) -> Agent:
    return max(
        attackers,
        key=lambda attacker: (
            defender.damage_sources.get(attacker, 0.0),
            -agent_order.get(attacker, 0),
        ),
    )


def _apply_attack_plans(plans: list[AttackPlan], agents: list[Agent]) -> bool:
    agent_order = {agent: index for index, agent in enumerate(agents)}
    hits_by_defender: dict[Agent, list[Agent]] = {}

    for plan in plans:
        if plan.parried_by is None:
            for defender in plan.defenders:
                hits_by_defender.setdefault(defender, []).append(plan.attacker)
            continue

        plan.attacker.apply_stun()
        plan.parried_by.clear_parry_weakness()
        plan.parried_by.parries_landed += 1
        plan.attacker.times_parried += 1

    any_damage = False
    defeated_defenders: list[tuple[Agent, list[Agent]]] = []
    for defender, attackers in hits_by_defender.items():
        if not defender.is_alive() or not attackers:
            continue

        hp_before = defender.hp
        nominal_damage = MELEE_DAMAGE * len(attackers)
        actual_damage = min(hp_before, nominal_damage)
        if actual_damage <= 0:
            continue

        credit_per_hit = actual_damage / len(attackers)
        defender.hp = max(0, hp_before - nominal_damage)
        defender.hit_flash_timer = HIT_FLASH_DURATION
        defender.damage_taken += actual_damage
        for attacker in attackers:
            attacker.damage_dealt += credit_per_hit
            attacker.hits_landed += 1
            defender.damage_sources[attacker] = (
                defender.damage_sources.get(attacker, 0.0) + credit_per_hit
            )

        any_damage = True
        if hp_before > 0 and defender.hp <= 0:
            defeated_defenders.append((defender, attackers))

    for defender, attackers in defeated_defenders:
        killer = _kill_credit_source(defender, attackers, agent_order)
        killer.kills += 1
        defender.deaths += 1
        _distribute_death_heal(defender)

    return any_damage


def _resolve_pending_attacks(agents: list[Agent]) -> bool:
    attackers = [
        agent for agent in agents
        if agent.is_alive() and agent._fire_hitbox
    ]
    plans = [_build_attack_plan(attacker, agents) for attacker in attackers]

    for agent in agents:
        agent._fire_hitbox = False

    return _apply_attack_plans(plans, agents)


def _collision_candidate_pairs(agents: list[Agent], cell_size: float) -> list[tuple[int, int]]:
    grid: dict[tuple[int, int], list[int]] = {}
    for index, agent in enumerate(agents):
        if not agent.is_alive():
            continue
        cell = (
            math.floor(agent.pos.x / cell_size),
            math.floor(agent.pos.y / cell_size),
        )
        grid.setdefault(cell, []).append(index)

    pairs: set[tuple[int, int]] = set()
    for cell in sorted(grid):
        cx, cy = cell
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                neighbor = (cx + ox, cy + oy)
                if neighbor not in grid:
                    continue
                for i in grid[cell]:
                    for j in grid[neighbor]:
                        if i < j:
                            pairs.add((i, j))

    return sorted(pairs)


def _resolve_agent_collisions(arena: Arena, agents: list[Agent]) -> None:
    min_dist = AGENT_RADIUS * 2.0
    min_dist_sq = min_dist * min_dist

    for iteration in range(AGENT_COLLISION_ITERATIONS):
        pairs = _collision_candidate_pairs(agents, min_dist)
        if not pairs:
            break
        if iteration % 2 == 1:
            pairs.reverse()

        moved = False
        for i, j in pairs:
            a = agents[i]
            b = agents[j]
            if not a.is_alive() or not b.is_alive():
                continue

            delta = b.pos - a.pos
            dist_sq = delta.length_squared()
            if dist_sq >= min_dist_sq:
                continue

            if dist_sq > 1e-8:
                dist = math.sqrt(dist_sq)
                normal = delta / dist
            else:
                dist = 0.0
                normal = pygame.Vector2(1, 0)

            overlap = min_dist - dist
            correction = normal * (overlap * 0.5)
            a.pos -= correction
            b.pos += correction

            a_toward_b = a.vel.dot(normal)
            if a_toward_b > 0.0:
                a.vel -= normal * a_toward_b

            b_toward_a = b.vel.dot(normal)
            if b_toward_a < 0.0:
                b.vel -= normal * b_toward_a

            arena.clamp_circle(a.pos, a.vel, AGENT_RADIUS)
            arena.clamp_circle(b.pos, b.vel, AGENT_RADIUS)

            moved = True

        if not moved:
            break


# =============================================================================
# HUD drawing
# =============================================================================
def _draw_cooldown_bar(
    surf: pygame.Surface,
    font: pygame.font.Font,
    x: int, y: int, w: int, h: int,
    label: str,
    state: SwingState | ParryState,
    timer: float,
    full_cd: float,
    color_ready: tuple,
    color_active: tuple,
) -> None:
    pygame.draw.rect(surf, C_CD_BG, (x, y, w, h))

    if state in (SwingState.IDLE, ParryState.IDLE):
        pygame.draw.rect(surf, color_ready, (x, y, w, h))
        status = "READY"
    elif state in (SwingState.SWINGING, ParryState.ACTIVE):
        pygame.draw.rect(surf, color_active, (x, y, w, h))
        status = "ACTIVE"
    elif state == ParryState.WEAKNESS:
        pygame.draw.rect(surf, color_active, (x, y, w, h))
        status = f"WEAK {timer:.1f}s"
    elif state in (SwingState.ON_COOLDOWN, ParryState.ON_COOLDOWN):
        fraction = 1.0 - (timer / full_cd)
        fw = round(w * fraction)
        if fw > 0:
            pygame.draw.rect(surf, color_ready, (x, y, fw, h))
        status = f"{timer:.1f}s"
    else:
        status = ""

    pygame.draw.rect(surf, C_TEXT_MUTED, (x, y, w, h), 1)
    lbl = font.render(f"{label} {status}", True, C_TEXT)
    surf.blit(lbl, (x + 3, y + (h - lbl.get_height()) // 2))


def _draw_agent_hud(
    surf: pygame.Surface,
    font_md: pygame.font.Font,
    font_sm: pygame.font.Font,
    x: int, y: int,
    agent: Agent,
    label: str,
) -> None:
    bar_w, bar_h = 200, 18
    cd_w, cd_h = 200, 14
    pad = 5

    panel_h = 26 + bar_h + pad + cd_h + pad + cd_h + 8
    bg = pygame.Surface((bar_w + 24, panel_h), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    surf.blit(bg, (x - 8, y - 4))

    # Label + HP
    header = font_md.render(f"{label}  {agent.hp} / {MAX_HP}", True, C_TEXT)
    surf.blit(header, (x, y))
    y += 26

    # HP bar
    pygame.draw.rect(surf, C_HP_BG, (x, y, bar_w, bar_h))
    hp_w = round(bar_w * max(0, agent.hp) / MAX_HP)
    hp_color = C_HP_FILL if agent.hp > 30 else C_HP_LOW
    if hp_w > 0:
        pygame.draw.rect(surf, hp_color, (x, y, hp_w, bar_h))
    pygame.draw.rect(surf, C_TEXT_MUTED, (x, y, bar_w, bar_h), 1)
    y += bar_h + pad

    # Swing cooldown
    _draw_cooldown_bar(
        surf, font_sm, x, y, cd_w, cd_h,
        "ATK", agent.swing_state, agent.swing_timer,
        SWING_COOLDOWN, C_CD_SWING, C_CD_SWING_ACTIVE,
    )
    y += cd_h + pad

    # Parry cooldown
    _draw_cooldown_bar(
        surf, font_sm, x, y, cd_w, cd_h,
        "PAR", agent.parry_state, agent.parry_timer,
        PARRY_COOLDOWN, C_CD_PARRY, C_CD_PARRY_ACTIVE,
    )


def _draw_frame(
    screen: pygame.Surface,
    hud: pygame.Surface,
    font_md: pygame.font.Font,
    font_sm: pygame.font.Font,
    font_lg: pygame.font.Font,
    arena: Arena,
    agents: list[Agent],
    player: PlayerAgent | None,
    mode: str,
    team_setup: TeamSetup,
    fast_forward: bool,
    round_elapsed: float,
    trainer: EvolutionTrainer,
    game_state: str,
    last_round_summary: str,
) -> None:
    screen.fill(C_FLOOR)
    arena.draw(screen)

    for agent in agents:
        agent.draw(screen)
        _draw_overhead_hp(screen, font_sm, agent)

    for agent in agents:
        agent.draw_stun_stars(screen)

    for agent in agents:
        agent.draw_heal_particles(screen)

    # HUD overlay
    hud.fill((0, 0, 0, 0))
    _draw_match_hud(
        hud,
        font_md,
        font_sm,
        agents,
        player,
        mode,
        team_setup,
        fast_forward,
        round_elapsed,
        trainer,
        last_round_summary,
    )

    screen.blit(hud, (0, 0))

    # Game-over overlay
    if game_state == "game_over":
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))

        live_teams = _live_team_ids(agents)
        if player is not None and not player.is_alive():
            msg = "YOU WERE ELIMINATED"
        elif len(live_teams) == 1:
            msg = f"TEAM {next(iter(live_teams))} WINS"
        else:
            msg = "ROUND OVER"

        txt = font_lg.render(msg, True, C_TEXT)
        screen.blit(txt, (WIN_W // 2 - txt.get_width() // 2, WIN_H // 2 - 60))
        sub = font_md.render("Press R for a new round", True, C_TEXT_MUTED)
        screen.blit(sub, (WIN_W // 2 - sub.get_width() // 2, WIN_H // 2 + 10))


def _draw_overhead_hp(surface: pygame.Surface, font: pygame.font.Font, agent: Agent) -> None:
    if not agent.is_alive():
        return

    w, h = 46, 5
    x = round(agent.pos.x - w / 2)
    y = round(agent.pos.y - AGENT_RADIUS - 14)
    pygame.draw.rect(surface, C_HP_BG, (x, y, w, h))
    fill_w = round(w * max(0, agent.hp) / MAX_HP)
    if fill_w > 0:
        pygame.draw.rect(surface, _team_color(agent.team_id), (x, y, fill_w, h))
    pygame.draw.rect(surface, C_TEXT_MUTED, (x, y, w, h), 1)

    if isinstance(agent, NeuralAgent) and agent.genome_id is not None:
        label = font.render(str(agent.genome_id), True, C_TEXT_MUTED)
        surface.blit(label, (round(agent.pos.x - label.get_width() / 2), y - label.get_height() - 1))


def _draw_match_hud(
    hud: pygame.Surface,
    font_md: pygame.font.Font,
    font_sm: pygame.font.Font,
    agents: list[Agent],
    player: PlayerAgent | None,
    mode: str,
    team_setup: TeamSetup,
    fast_forward: bool,
    round_elapsed: float,
    trainer: EvolutionTrainer,
    last_round_summary: str,
) -> None:
    panel = pygame.Surface((430, 154), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 165))
    hud.blit(panel, (16, 14))

    mode_label = "Player Team" if mode == MODE_HUMAN_VS_AI else "AI Self-Play"
    mode_label += f" {team_setup.name}"
    if fast_forward:
        mode_label += f" FAST {FAST_FORWARD_STEPS_PER_FRAME}@{FAST_FORWARD_FPS_CAP}"
    timer_text = ""
    if mode == MODE_AI_VS_AI:
        timer_text = f"   Time {max(0.0, AI_ROUND_SECONDS - round_elapsed):04.1f}s"
    title = font_md.render(f"{mode_label}{timer_text}", True, C_TEXT)
    hud.blit(title, (28, 24))

    evo = font_sm.render(
        f"Generation {trainer.generation}   evaluated {trainer.evaluated_this_generation}/{trainer.population_size}   best {trainer.best_score:.1f}",
        True,
        C_TEXT_MUTED,
    )
    hud.blit(evo, (28, 51))

    y = 76
    for team_id in range(1, team_setup.team_count + 1):
        team_agents = [agent for agent in agents if agent.team_id == team_id]
        alive = sum(1 for agent in team_agents if agent.is_alive())
        hp = sum(max(0, agent.hp) for agent in team_agents)
        pygame.draw.rect(hud, _team_color(team_id), (28, y + 3, 12, 12))
        text = font_sm.render(f"Team {team_id}: {alive}/{len(team_agents)} alive   HP {hp}", True, C_TEXT)
        hud.blit(text, (48, y))
        y += 20

    if player is not None:
        player_text = font_sm.render(f"Player HP {max(0, player.hp)}/{MAX_HP}", True, C_TEXT_MUTED)
        hud.blit(player_text, (240, 76))

    if last_round_summary:
        summary = font_sm.render(last_round_summary[:54], True, C_TEXT_MUTED)
        hud.blit(summary, (28, 138))

    legend_bg = pygame.Surface((WIN_W, 34), pygame.SRCALPHA)
    legend_bg.fill((0, 0, 0, 160))
    hud.blit(legend_bg, (0, WIN_H - 34))
    legend = font_sm.render(
        "1: Toggle Player/AI self-play   2: Fast training   R: New round   WASD/Mouse/LClick/RClick/F: Player controls   Esc: Quit",
        True,
        C_TEXT_MUTED,
    )
    hud.blit(legend, (WIN_W // 2 - legend.get_width() // 2, WIN_H - 27))


# =============================================================================
# Game factory
# =============================================================================
def _choose_round_setup(mode: str) -> TeamSetup:
    if mode == MODE_AI_VS_AI:
        return random.choice(AI_TEAM_SETUPS)
    return DEFAULT_TEAM_SETUP


def _vertical_spawn_column(x_fraction: float, count: int) -> list[pygame.Vector2]:
    usable_top = WALL_T + AGENT_RADIUS + 30
    usable_bottom = WIN_H - WALL_T - AGENT_RADIUS - 30
    if count <= 1:
        ys = [(usable_top + usable_bottom) / 2]
    else:
        step = (usable_bottom - usable_top) / (count - 1)
        ys = [usable_top + step * i for i in range(count)]
    return [pygame.Vector2(WIN_W * x_fraction, y) for y in ys]


def _team_spawn_points(setup: TeamSetup) -> dict[int, list[pygame.Vector2]]:
    if setup.name == "3v3v3":
        return {
            1: [
                pygame.Vector2(WIN_W * 0.20, WIN_H * 0.30),
                pygame.Vector2(WIN_W * 0.20, WIN_H * 0.50),
                pygame.Vector2(WIN_W * 0.20, WIN_H * 0.70),
            ],
            2: [
                pygame.Vector2(WIN_W * 0.80, WIN_H * 0.30),
                pygame.Vector2(WIN_W * 0.80, WIN_H * 0.50),
                pygame.Vector2(WIN_W * 0.80, WIN_H * 0.70),
            ],
            3: [
                pygame.Vector2(WIN_W * 0.48, WIN_H * 0.22),
                pygame.Vector2(WIN_W * 0.58, WIN_H * 0.50),
                pygame.Vector2(WIN_W * 0.48, WIN_H * 0.78),
            ],
        }

    if setup.name == "4v4":
        return {
            1: _vertical_spawn_column(0.20, setup.team_sizes[0]),
            2: _vertical_spawn_column(0.80, setup.team_sizes[1]),
        }

    raise ValueError(f"Unsupported team setup: {setup.name}")


def _set_initial_look(agent: Agent, agents: list[Agent]) -> None:
    target = _nearest_enemy(agent, agents)
    if target is None:
        agent.look_angle = 0.0
        return
    diff = target.pos - agent.pos
    if diff.length_squared() > 0:
        agent.look_angle = math.atan2(diff.y, diff.x)


def _make_round_agents(
    mode: str,
    trainer: EvolutionTrainer,
    setup: TeamSetup,
) -> tuple[PlayerAgent | None, list[Agent]]:
    spawns = _team_spawn_points(setup)
    player: PlayerAgent | None = None
    agents: list[Agent] = []
    team_genomes = dict(zip(range(1, setup.team_count + 1), trainer.select_round_genomes(setup.team_count)))
    team_models = {
        team_id: trainer.build_model(genome)
        for team_id, genome in team_genomes.items()
    }

    for team_id, team_size in enumerate(setup.team_sizes, start=1):
        if len(spawns[team_id]) != team_size:
            raise ValueError(f"Setup {setup.name} has mismatched spawn count for team {team_id}.")
        for index, pos in enumerate(spawns[team_id]):
            if mode == MODE_HUMAN_VS_AI and team_id == 1 and index == 0:
                player = PlayerAgent(pos, team_id=1)
                agents.append(player)
            else:
                genome = team_genomes[team_id]
                agent = NeuralAgent(
                    pos,
                    team_id,
                    team_models[team_id],
                    genome_id=genome.genome_id,
                    label=f"T{team_id}-{index + 1}",
                )
                agents.append(agent)

    for agent in agents:
        _set_initial_look(agent, agents)

    return player, agents


def _live_team_ids(agents: list[Agent]) -> set[int]:
    return {
        agent.team_id for agent in agents
        if agent.team_id is not None and agent.is_alive()
    }


def _round_end_reason(
    mode: str,
    agents: list[Agent],
    player: PlayerAgent | None,
    round_elapsed: float,
    last_hit_elapsed: float,
) -> str | None:
    live_teams = _live_team_ids(agents)
    if len(live_teams) <= 1:
        return "team_eliminated"
    if mode == MODE_AI_VS_AI and round_elapsed >= AI_ROUND_SECONDS:
        return "timeout"
    if mode == MODE_AI_VS_AI and (round_elapsed - last_hit_elapsed) >= NO_COMBAT_TIMEOUT:
        return "no_combat"
    if mode == MODE_HUMAN_VS_AI and player is not None and not player.is_alive():
        return "player_eliminated"
    return None


def _team_ids(agents: list[Agent]) -> list[int]:
    return sorted({
        agent.team_id for agent in agents
        if agent.team_id is not None
    })


def _team_hp_totals(agents: list[Agent]) -> dict[int, int]:
    return {
        team_id: sum(max(0, agent.hp) for agent in agents if agent.team_id == team_id)
        for team_id in _team_ids(agents)
    }


def _team_bonuses(agents: list[Agent], reason: str) -> dict[int, float]:
    live_teams = _live_team_ids(agents)
    bonuses = {team_id: -4.0 for team_id in _team_ids(agents)}
    if len(live_teams) == 1:
        winner = next(iter(live_teams))
        bonuses[winner] = 16.0
        return bonuses

    if reason in ("timeout", "no_combat"):
        return bonuses

    hp_totals = _team_hp_totals(agents)
    ranked = sorted(hp_totals, key=lambda team_id: hp_totals[team_id], reverse=True)
    rank_values = [7.0, 2.0, -3.0]
    for team_id, value in zip(ranked, rank_values):
        bonuses[team_id] = value
    if reason == "player_eliminated":
        bonuses[1] -= 6.0
    return bonuses


def _team_genome_fitness(team_agents: list[NeuralAgent], elapsed: float, team_bonus: float) -> float:
    if not team_agents:
        return 0.0

    damage_dealt = sum(agent.damage_dealt for agent in team_agents)
    damage_taken = sum(agent.damage_taken for agent in team_agents)
    kills = sum(agent.kills for agent in team_agents)
    deaths = sum(agent.deaths for agent in team_agents)
    hits_landed = sum(agent.hits_landed for agent in team_agents)
    swings_started = sum(agent.swings_started for agent in team_agents)
    parries_landed = sum(agent.parries_landed for agent in team_agents)
    times_parried = sum(agent.times_parried for agent in team_agents)
    death_heal_received = sum(agent.death_heal_received for agent in team_agents)
    survival_time = sum(min(agent.round_survival_time, elapsed) for agent in team_agents)
    engagement_reward = sum(agent.engagement_reward for agent in team_agents)
    facing_reward = sum(agent.facing_reward for agent in team_agents)
    wall_proximity_penalty = sum(agent.wall_proximity_penalty for agent in team_agents)
    remaining_hp = sum(max(0, agent.hp) for agent in team_agents)
    alive_count = sum(1 for agent in team_agents if agent.is_alive())
    misses = max(0, swings_started - hits_landed - times_parried)

    return (
        damage_dealt * 0.12
        - damage_taken * 0.045
        + kills * 14.0
        - deaths * 7.0
        + hits_landed * 0.25
        + parries_landed * 3.0
        - times_parried * 1.5
        - misses * 0.025
        + death_heal_received * 0.035
        + survival_time * 0.015
        + engagement_reward
        + facing_reward
        - wall_proximity_penalty
        + remaining_hp / MAX_HP * 0.55
        + alive_count * 0.75
        + team_bonus
    )


def _finish_round(
    trainer: EvolutionTrainer,
    agents: list[Agent],
    elapsed: float,
    reason: str,
    *,
    train: bool = True,
) -> str:
    bonuses = _team_bonuses(agents, reason)
    team_agents_by_genome: dict[int, list[NeuralAgent]] = {}
    team_id_by_genome: dict[int, int] = {}
    for agent in agents:
        if isinstance(agent, NeuralAgent) and agent.genome_id is not None:
            team_agents_by_genome.setdefault(agent.genome_id, []).append(agent)
            team_id_by_genome[agent.genome_id] = agent.team_id or 0

    fitness_by_genome = {
        genome_id: _team_genome_fitness(
            team_agents,
            elapsed,
            bonuses.get(team_id_by_genome.get(genome_id, 0), 0.0),
        )
        for genome_id, team_agents in team_agents_by_genome.items()
        if team_agents
    }

    evolved = trainer.record_round(fitness_by_genome) if train and fitness_by_genome else False

    if len(_live_team_ids(agents)) == 1:
        summary = f"Team {next(iter(_live_team_ids(agents)))} won"
    elif reason == "timeout":
        fitness_by_team = {
            team_id_by_genome[genome_id]: fitness
            for genome_id, fitness in fitness_by_genome.items()
            if genome_id in team_id_by_genome
        }
        if fitness_by_team:
            leader = max(fitness_by_team, key=lambda team_id: fitness_by_team[team_id])
            summary = f"Timeout; Team {leader} led on fitness {fitness_by_team[leader]:.1f}"
        else:
            summary = "Timeout"
    elif reason == "no_combat":
        summary = "No combat; all penalized"
    elif reason == "player_eliminated":
        summary = "Player eliminated"
    else:
        summary = "Round ended"
    if not train:
        summary += "; no training"
    if evolved:
        summary += f"; evolved to gen {trainer.generation}"
    return summary


@dataclass
class SimulationStepResult:
    player: PlayerAgent | None
    agents: list[Agent]
    team_setup: TeamSetup
    round_elapsed: float
    last_hit_elapsed: float
    physics_tick: int
    game_state: str
    summary: str | None = None
    round_finished: bool = False
    ai_decisions: int = 0
    ai_batches: int = 0


def _simulate_physics_step(
    arena: Arena,
    trainer: EvolutionTrainer,
    mode: str,
    player: PlayerAgent | None,
    agents: list[Agent],
    team_setup: TeamSetup,
    round_elapsed: float,
    last_hit_elapsed: float,
    physics_tick: int,
) -> SimulationStepResult:
    ai_decisions = 0
    ai_batches = 0
    if physics_tick % AI_DECISION_STEPS == 0:
        ai_decisions, ai_batches = _refresh_neural_policy_actions(agents, arena)

    for agent in agents:
        if not agent.is_alive():
            continue
        if isinstance(agent, PlayerAgent):
            agent.update(PHYSICS_STEP, arena)
        elif isinstance(agent, NeuralAgent):
            agent.update(PHYSICS_STEP, arena, agents)

    _resolve_agent_collisions(arena, agents)

    if _resolve_pending_attacks(agents):
        last_hit_elapsed = round_elapsed

    for agent in agents:
        if agent.is_alive():
            agent.round_survival_time += PHYSICS_STEP
            _tick_engagement_rewards(agent, agents, PHYSICS_STEP)
            _tick_wall_proximity_penalty(agent, arena, PHYSICS_STEP)

    round_elapsed += PHYSICS_STEP
    physics_tick += 1
    reason = _round_end_reason(mode, agents, player, round_elapsed, last_hit_elapsed)
    if reason is None:
        return SimulationStepResult(
            player,
            agents,
            team_setup,
            round_elapsed,
            last_hit_elapsed,
            physics_tick,
            "playing",
            ai_decisions=ai_decisions,
            ai_batches=ai_batches,
        )

    summary = _finish_round(
        trainer,
        agents,
        round_elapsed,
        reason,
        train=mode == MODE_AI_VS_AI,
    )
    if mode == MODE_AI_VS_AI:
        next_setup = _choose_round_setup(mode)
        next_player, next_agents = _make_round_agents(mode, trainer, next_setup)
        return SimulationStepResult(
            next_player,
            next_agents,
            next_setup,
            0.0,
            0.0,
            0,
            "playing",
            summary,
            True,
            ai_decisions,
            ai_batches,
        )

    return SimulationStepResult(
        player,
        agents,
        team_setup,
        round_elapsed,
        last_hit_elapsed,
        physics_tick,
        "game_over",
        summary,
        True,
        ai_decisions,
        ai_batches,
    )


# =============================================================================
# Main
# =============================================================================
def _run_visual() -> int:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Arena Combat")
    clock = pygame.time.Clock()

    font_lg = pygame.font.SysFont("consolas", 64, bold=True)
    font_md = pygame.font.SysFont("consolas", 20, bold=True)
    font_sm = pygame.font.SysFont("consolas", 14)

    hud = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

    arena = Arena()
    trainer = EvolutionTrainer()
    mode = MODE_HUMAN_VS_AI
    team_setup = _choose_round_setup(mode)
    player, agents = _make_round_agents(mode, trainer, team_setup)
    accumulator = 0.0
    round_elapsed = 0.0
    last_hit_elapsed = 0.0
    physics_tick = 0
    game_state = "playing"
    last_round_summary = ""
    fast_forward = False

    def start_round(next_mode: str) -> None:
        nonlocal mode, team_setup, player, agents, accumulator, round_elapsed, last_hit_elapsed, physics_tick, game_state, last_round_summary
        mode = next_mode
        team_setup = _choose_round_setup(mode)
        player, agents = _make_round_agents(mode, trainer, team_setup)
        accumulator = 0.0
        round_elapsed = 0.0
        last_hit_elapsed = 0.0
        physics_tick = 0
        game_state = "playing"
        last_round_summary = ""

    def shutdown() -> int:
        trainer.save()
        pygame.quit()
        return 0

    while True:
        fps_cap = FAST_FORWARD_FPS_CAP if fast_forward else FPS_CAP
        frame_time = min(clock.tick(fps_cap) / 1000.0, MAX_FRAME_TIME)

        if psutil.cpu_percent() > CPU_THROTTLE_LIMIT:
            pygame.time.wait(CPU_THROTTLE_SLEEP_MS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return shutdown()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return shutdown()
                if event.key == pygame.K_1:
                    fast_forward = False
                    next_mode = MODE_AI_VS_AI if mode == MODE_HUMAN_VS_AI else MODE_HUMAN_VS_AI
                    start_round(next_mode)
                if event.key == pygame.K_2:
                    fast_forward = not fast_forward
                    if fast_forward and mode != MODE_AI_VS_AI:
                        start_round(MODE_AI_VS_AI)
                if event.key == pygame.K_r:
                    start_round(mode)
            if game_state == "playing" and player is not None:
                player.handle_event(event)

        if game_state == "playing":
            if fast_forward and mode == MODE_AI_VS_AI:
                accumulator = 0.0
                for _ in range(FAST_FORWARD_STEPS_PER_FRAME):
                    result = _simulate_physics_step(
                        arena,
                        trainer,
                        mode,
                        player,
                        agents,
                        team_setup,
                        round_elapsed,
                        last_hit_elapsed,
                        physics_tick,
                    )
                    player = result.player
                    agents = result.agents
                    team_setup = result.team_setup
                    round_elapsed = result.round_elapsed
                    last_hit_elapsed = result.last_hit_elapsed
                    physics_tick = result.physics_tick
                    game_state = result.game_state
                    if result.summary:
                        last_round_summary = result.summary
                    if game_state != "playing":
                        fast_forward = False
                        break
            else:
                accumulator += frame_time
                while accumulator >= PHYSICS_STEP:
                    result = _simulate_physics_step(
                        arena,
                        trainer,
                        mode,
                        player,
                        agents,
                        team_setup,
                        round_elapsed,
                        last_hit_elapsed,
                        physics_tick,
                    )
                    player = result.player
                    agents = result.agents
                    team_setup = result.team_setup
                    round_elapsed = result.round_elapsed
                    last_hit_elapsed = result.last_hit_elapsed
                    physics_tick = result.physics_tick
                    game_state = result.game_state
                    if result.summary:
                        last_round_summary = result.summary
                    if result.round_finished:
                        accumulator = 0.0
                        break
                    accumulator -= PHYSICS_STEP

                    if game_state != "playing":
                        break

        _draw_frame(
            screen,
            hud,
            font_md,
            font_sm,
            font_lg,
            arena,
            agents,
            player,
            mode,
            team_setup,
            fast_forward,
            round_elapsed,
            trainer,
            game_state,
            last_round_summary,
        )
        pygame.display.flip()

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arena Combat")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run AI-vs-AI training without opening a window or rendering frames",
    )
    parser.add_argument(
        "--headless-rounds",
        type=int,
        default=0,
        help="number of headless rounds to run; 0 means run until interrupted",
    )
    parser.add_argument(
        "--headless-log-every",
        type=int,
        default=10,
        help="log headless training progress every N finished rounds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional random seed for reproducible training runs",
    )
    return parser.parse_args(argv)


def _run_headless(args: argparse.Namespace) -> int:
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    arena = Arena()
    trainer = EvolutionTrainer()
    mode = MODE_AI_VS_AI
    team_setup = _choose_round_setup(mode)
    player, agents = _make_round_agents(mode, trainer, team_setup)
    round_elapsed = 0.0
    last_hit_elapsed = 0.0
    physics_tick = 0

    rounds_run = 0
    physics_steps = 0
    ai_decisions = 0
    ai_batches = 0
    started_at = time.perf_counter()
    last_logged_at = started_at
    last_logged_steps = 0
    last_logged_rounds = 0

    max_rounds = max(0, args.headless_rounds)
    log_every = max(1, args.headless_log_every)
    print(
        f"[headless] physics={PHYSICS_HZ}Hz ai={AI_DECISION_HZ}Hz "
        f"decision_steps={AI_DECISION_STEPS} rounds={'forever' if max_rounds == 0 else max_rounds}",
        flush=True,
    )

    try:
        while max_rounds == 0 or rounds_run < max_rounds:
            result = _simulate_physics_step(
                arena,
                trainer,
                mode,
                player,
                agents,
                team_setup,
                round_elapsed,
                last_hit_elapsed,
                physics_tick,
            )
            player = result.player
            agents = result.agents
            team_setup = result.team_setup
            round_elapsed = result.round_elapsed
            last_hit_elapsed = result.last_hit_elapsed
            physics_tick = result.physics_tick
            physics_steps += 1
            ai_decisions += result.ai_decisions
            ai_batches += result.ai_batches

            if not result.round_finished:
                continue

            rounds_run += 1
            should_log = (
                rounds_run % log_every == 0
                or rounds_run == 1
                or (result.summary is not None and "evolved" in result.summary)
                or (max_rounds > 0 and rounds_run >= max_rounds)
            )
            if not should_log:
                continue

            now = time.perf_counter()
            elapsed = max(1e-6, now - last_logged_at)
            total_elapsed = max(1e-6, now - started_at)
            window_steps = physics_steps - last_logged_steps
            window_rounds = rounds_run - last_logged_rounds
            steps_per_second = window_steps / elapsed
            rounds_per_second = window_rounds / elapsed
            sim_speed = steps_per_second * PHYSICS_STEP
            avg_batch = ai_decisions / ai_batches if ai_batches else 0.0
            print(
                f"[headless] rounds={rounds_run} gen={trainer.generation} "
                f"eval={trainer.evaluated_this_generation}/{trainer.population_size} "
                f"best={trainer.best_score:.2f} steps/s={steps_per_second:.0f} "
                f"sim={sim_speed:.1f}x rounds/s={rounds_per_second:.2f} "
                f"ai={ai_decisions} decisions/{ai_batches} batches avg_batch={avg_batch:.2f} "
                f"elapsed={total_elapsed:.1f}s last='{result.summary or ''}'",
                flush=True,
            )
            last_logged_at = now
            last_logged_steps = physics_steps
            last_logged_rounds = rounds_run
    except KeyboardInterrupt:
        print("[headless] interrupted; saving checkpoint", flush=True)
    finally:
        trainer.save()

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.headless:
        return _run_headless(args)
    return _run_visual()


if __name__ == "__main__":
    raise SystemExit(main())
