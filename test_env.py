from __future__ import annotations

import unittest

from env import FireDroneSwarmEnv


class FireDroneSwarmEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = FireDroneSwarmEnv(num_drones=3)
        self.env.reset()

    def test_reset_returns_expected_observation_shape(self) -> None:
        observation = self.env.reset()

        self.assertIn("city_grid", observation)
        self.assertIn("drones", observation)
        self.assertIn("sensor_alert", observation)
        self.assertEqual(len(observation["city_grid"]), 15)
        self.assertEqual(len(observation["city_grid"][0]), 15)
        self.assertEqual(len(observation["drones"]), 3)
        self.assertEqual(tuple(observation["sensor_alert"]), (9, 8))

    def test_invalid_commands_default_to_noop(self) -> None:
        before = {
            drone["id"]: (drone["x"], drone["y"], drone["battery_level"])
            for drone in self.env.drones.values()
        }

        observation, reward, done, info = self.env.step(
            {"commands": ["move(ghost_drone, right)", "bad command"]}
        )

        after = {
            drone["id"]: (drone["x"], drone["y"], drone["battery_level"])
            for drone in self.env.drones.values()
        }

        self.assertEqual(before, after)
        self.assertEqual(reward, -1.0)
        self.assertFalse(done)
        self.assertEqual(info["events"], [])
        self.assertEqual(len(observation["drones"]), 3)

    def test_scan_can_discover_civilian_and_complete_task1(self) -> None:
        self.env.drones["drone_1"]["x"] = 10
        self.env.drones["drone_1"]["y"] = 7

        _, reward, _, info = self.env.step({"commands": ["scan(drone_1, 2)"]})

        self.assertGreaterEqual(reward, 9.0)
        self.assertTrue(self.env.civilian_discovered)
        self.assertEqual(info["graders"]["task1_scout_map"], 1.0)

    def test_spray_extinguishes_single_low_intensity_tile(self) -> None:
        self.env.fire_intensity[(9, 8)] = 1
        self.env.drones["drone_0"]["x"] = 8
        self.env.drones["drone_0"]["y"] = 8

        _, reward, _, info = self.env.step({"commands": ["spray(drone_0, right)"]})

        self.assertNotIn((9, 8), self.env.fire_intensity)
        self.assertEqual(reward, 14.0)
        self.assertIn("drone_0 spray_1", info["events"])

    def test_drop_ball_can_clear_cluster_and_finish_task2(self) -> None:
        self.env.drones["drone_2"]["x"] = 9
        self.env.drones["drone_2"]["y"] = 8
        for key in list(self.env.fire_intensity):
            self.env.fire_intensity[key] = 1

        _, reward, _, info = self.env.step({"commands": ["drop_ball(drone_2)"]})

        self.assertEqual(len(self.env.fire_intensity), 0)
        self.assertEqual(info["graders"]["task2_containment"], 1.0)
        self.assertGreater(reward, 50.0)

    def test_projected_path_moves_civilian_and_rewards_follow_step(self) -> None:
        self.env.fire_intensity = {}
        self.env.drones["drone_1"]["x"] = 10
        self.env.drones["drone_1"]["y"] = 8

        _, reward, _, info = self.env.step({"commands": ["project_path(drone_1, up)"]})

        self.assertEqual(self.env.civilian_pos, (10, 7))
        self.assertEqual(reward, 49.0)
        self.assertIn("civilian_followed_projected_path", info["events"])

    def test_full_coordination_sequence_completes_task3(self) -> None:
        self.env.drones["drone_0"]["x"] = 8
        self.env.drones["drone_0"]["y"] = 8
        self.env.drones["drone_1"]["x"] = 10
        self.env.drones["drone_1"]["y"] = 8
        self.env.drones["drone_2"]["x"] = 9
        self.env.drones["drone_2"]["y"] = 8

        for key in list(self.env.fire_intensity):
            self.env.fire_intensity[key] = 1

        scripted_actions = [
            {"commands": ["spray(drone_0, right)", "scan(drone_1, 2)", "drop_ball(drone_2)"]},
            {"commands": ["project_path(drone_1, up)"]},
            {"commands": ["move(drone_1, up)"]},
            {"commands": ["project_path(drone_1, left)"]},
            {"commands": []},
            {"commands": []},
            {"commands": []},
            {"commands": []},
        ]

        info = {}
        done = False
        for action in scripted_actions:
            _, _, done, info = self.env.step(action)
            if done:
                break

        self.assertTrue(done)
        self.assertTrue(self.env.civilian_reached_exit)
        self.assertEqual(info["graders"]["task3_coordinated_rescue"], 1.0)
        self.assertEqual(info["coordination"]["suppression_drones"], ["drone_0", "drone_2"])
        self.assertEqual(info["coordination"]["path_drones"], ["drone_1"])


if __name__ == "__main__":
    unittest.main()
