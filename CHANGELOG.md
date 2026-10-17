# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Fix the action space bounds: Playground actions are normalized to `[-1, 1]` (dm_control_suite envs send them to actuators with ctrlrange `[-1, 1]`, locomotion / manipulation envs scale them by `action_scale`), but the space was declared unbounded, so `sample_action()` produced out-of-range actions.
- Support playground 0.2: `MadronaWrapper` is imported only when available (vision envs now render with MJX-Warp under the plain vmap wrapper), and masked resets skip state leaves that are not batched per environment (e.g. the empty placeholder arrays of `mjx.Data`), which made `reset(mask=...)` fail.

## 0.0.1a2 (2026-02-10)

- Make a separate repository for mujoco playground adaptor.

## 0.0.1a1 (2025-11-13)

- Initial release with mujoco playground environment adaptor