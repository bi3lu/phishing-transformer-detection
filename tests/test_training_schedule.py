import unittest

from src.models.training_schedule import calculate_warmup_steps, resolve_training_schedule


class TrainingScheduleTests(unittest.TestCase):
    def test_memory_safe_schedule_preserves_effective_batch(self) -> None:
        schedule = resolve_training_schedule(
            {
                "batch_size": 1,
                "eval_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "gradient_checkpointing": True,
            }
        )
        self.assertEqual(schedule, (1, 1, 16, True))

    def test_invalid_effective_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Effective batch size"):
            resolve_training_schedule({"batch_size": 2, "gradient_accumulation_steps": 4})

    def test_warmup_steps_match_optimizer_updates(self) -> None:
        self.assertEqual(calculate_warmup_steps(1394, 4, 4, 10), 88)
        self.assertEqual(calculate_warmup_steps(1394, 1, 16, 10), 88)


if __name__ == "__main__":
    unittest.main()
