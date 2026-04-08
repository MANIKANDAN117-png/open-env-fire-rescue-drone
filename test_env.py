from __future__ import annotations

import unittest

from app import list_tasks, reset_environment, step_environment
from dashboard_backend import MissionCommander
from env import FireDroneSwarmEnv


class FireDroneSwarmEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = FireDroneSwarmEnv(num_drones=6)
        self.env.reset("Medium")

    def test_reset_returns_expected_observation_shape(self) -> None:
        observation = self.env.reset("Medium")

        self.assertIn("city_grid", observation)
        self.assertIn("drones", observation)
        self.assertIn("sensor_alert", observation)
        self.assertEqual(len(observation["city_grid"]), 15)
        self.assertEqual(len(observation["city_grid"][0]), 15)
        self.assertEqual(len(observation["drones"]), 6)
        self.assertEqual(tuple(observation["sensor_alert"]), (9, 8))
        self.assertEqual(observation["scenario"], "Medium")

    def test_invalid_commands_fall_back_to_idle_behavior(self) -> None:
        before = {
            drone["id"]: (drone["x"], drone["y"], drone["battery_level"])
            for drone in self.env.drones.values()
        }

        _, reward, done, info = self.env.step({"commands": ["move(ghost_drone, right)", "bad command"]})

        after = {
            drone["id"]: (drone["x"], drone["y"], drone["battery_level"])
            for drone in self.env.drones.values()
        }

        self.assertFalse(done)
        self.assertEqual(info["events"], [])
        self.assertEqual(reward, -1.0)
        for drone_id, (x, y, battery) in before.items():
            self.assertEqual((x, y), after[drone_id][:2])
            self.assertAlmostEqual(after[drone_id][2], battery - 0.2, places=3)

    def test_spray_extinguishes_fire_and_updates_containment_score(self) -> None:
        self.env.fire_intensity = {(9, 8): 1}
        self.env.city_grid_true[8][9] = self.env.FIRE
        self.env.drones["drone_0"]["x"] = 8
        self.env.drones["drone_0"]["y"] = 8

        _, reward, _, info = self.env.step({"commands": ["spray(drone_0, right)"]})

        self.assertNotIn((9, 8), self.env.fire_intensity)
        self.assertEqual(reward, 12.0)
        self.assertGreater(info["graders"]["task2_containment"], 0.0)
        self.assertIn("drone_0", info["coordination"]["suppression_drones"])

    def test_civilian_waits_until_path_is_projected(self) -> None:
        self.env.fire_intensity = {}
        self.env.drones["drone_1"]["x"] = 10
        self.env.drones["drone_1"]["y"] = 7
        start_position = self.env.civilian_pos

        _, reward, _, info = self.env.step({"commands": ["scan(drone_1, 1)"]})

        self.assertEqual(self.env.civilian_pos, start_position)
        self.assertEqual(info["coordination"]["path_drones"], [])
        self.assertLessEqual(reward, 10.0)

    def test_project_path_moves_civilian_and_records_coordination(self) -> None:
        self.env.fire_intensity = {}
        self.env.drones["drone_3"]["x"] = 10
        self.env.drones["drone_3"]["y"] = 7

        _, reward, _, info = self.env.step({"commands": ["project_path(drone_3, left)"]})

        self.assertEqual(self.env.civilian_pos, (9, 8))
        self.assertGreater(reward, 10.0)
        self.assertEqual(info["coordination"]["path_drones"], ["drone_3"])

    def test_commander_completes_all_three_tasks(self) -> None:
        env = FireDroneSwarmEnv(num_drones=6)
        observation = env.reset("Medium")
        commander = MissionCommander()
        done = False
        info = {}

        for _ in range(80):
            plan = commander.act(env, observation)
            observation, _, done, info = env.step({"commands": plan["commands"]})
            if done:
                break

        self.assertTrue(done)
        self.assertTrue(env.civilian_reached_exit)
        self.assertEqual(len(env.fire_intensity), 0)
        self.assertEqual(info["graders"]["task1_scout_map"], 0.99)
        self.assertEqual(info["graders"]["task2_containment"], 0.99)
        self.assertEqual(info["graders"]["task3_coordinated_rescue"], 0.99)

    def test_app_helpers_accept_default_bodies_and_publish_tasks(self) -> None:
        reset_response = reset_environment()
        step_response = step_environment()
        task_response = list_tasks()

        self.assertEqual(reset_response.scenario, "Medium")
        self.assertEqual(reset_response.difficulty, "Medium")
        self.assertEqual(step_response.observation.scenario, "Medium")
        self.assertEqual(task_response.count, 3)
        self.assertEqual(task_response.tasks[0].score_range.min, 0.01)
        self.assertEqual(task_response.tasks[0].score_range.max, 0.99)


if __name__ == "__main__":
    unittest.main()
