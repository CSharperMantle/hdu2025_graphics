# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Rong Bao <webmaster@csmantle.top>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing as ty

EasingFunc = ty.Callable[[int, int], float]
AnimationActor = ty.Callable[[float], bool]


def easing_linear(elapsed_us: int, duration_us: int) -> float:
    if duration_us <= 0:
        return 1.0
    return min(1.0, max(0.0, elapsed_us / duration_us))


def easing_in_out_quad(elapsed_us: int, duration_us: int) -> float:
    if duration_us <= 0:
        return 1.0
    t = elapsed_us / duration_us
    if t < 0.5:
        return 2 * t * t
    else:
        return -1 + (4 - 2 * t) * t


class Animation:
    _duration_us: int
    _elapsed_us: int
    _finished: bool
    _easing_func: EasingFunc
    _actor: AnimationActor

    @property
    def duration_us(self) -> int:
        return self._duration_us

    @property
    def elapsed_us(self) -> int:
        return self._elapsed_us

    @property
    def timed_out(self) -> bool:
        return self._elapsed_us >= self._duration_us

    @property
    def done(self) -> bool:
        return self._finished or self.timed_out

    def __init__(
        self,
        duration_us: int,
        actor: AnimationActor,
        easing_func: EasingFunc = easing_linear,
    ):
        self._duration_us = duration_us
        self._elapsed_us = 0
        self._finished = False
        self._easing_func = easing_func
        self._actor = actor

    def tick(self, elapsed_us: int):
        if self.done:
            return
        self._elapsed_us += elapsed_us
        if not self._finished and self.timed_out:
            self._actor(1.0)
            self._finished = True
        else:
            progress = self._easing_func(self._elapsed_us, self._duration_us)
            cont = self._actor(progress)
            if not cont:
                self._finished = True


class AnimationEngine:
    _queue: list[list[Animation]]

    def __init__(self):
        self._queue = []

    def split_keyframe(self):
        self._queue.append([])

    def fire(self, animation: Animation):
        if len(self._queue) == 0:
            self.split_keyframe()
        self._queue[-1].append(animation)

    def tick(self, elapsed_us: int):
        if len(self._queue) == 0:
            return
        for animation in self._queue[0]:
            animation.tick(elapsed_us)
        if all(animation.done for animation in self._queue[0]):
            self._queue.pop(0)
