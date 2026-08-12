from __future__ import annotations

import torch
import unittest

from src.rl.root_pivot import (
    labels_from_boundaries,
    labels_from_data,
    route_logistic_loss,
    task_only_data,
)


class RootPivotTest(unittest.TestCase):
    def test_route_loss_has_opposite_margin_gradients(self):
        logits = torch.zeros(2, 2, requires_grad=True)
        labels = labels_from_boundaries(["NeedSearch", "NoSearch"])
        loss, margin = route_logistic_loss(logits, labels)
        grad = torch.autograd.grad(loss, margin)[0]
        self.assertLess(grad[0], 0)  # descent increases NeedSearch margin
        self.assertGreater(grad[1], 0)  # descent decreases NoSearch margin

    def test_task_credit_excludes_first_response_token(self):
        advantages = torch.tensor([[3.0, 2.0, 1.0], [-4.0, 5.0, 6.0]])
        data = __import__("tensordict").TensorDict({"advantages": advantages}, batch_size=[2])
        result = task_only_data(data)
        self.assertEqual(result["advantages"].tolist(), [[0.0, 2.0, 1.0], [0.0, 5.0, 6.0]])
        self.assertEqual(data["advantages"][0, 0].item(), 3.0)

    def test_undetermined_is_rejected(self):
        with self.assertRaises(ValueError):
            labels_from_boundaries(["Undetermined"])

    def test_labels_come_from_dispatched_extra_fields(self):
        from verl.utils.tensordict_utils import assign_non_tensor_stack

        data = __import__("tensordict").TensorDict({}, batch_size=[2])
        assign_non_tensor_stack(
            data,
            "extra_fields",
            [
                {"reward_extra_info": {"boundary": "NeedSearch"}},
                {"reward_extra_info": {"boundary": "NoSearch"}},
            ],
        )
        self.assertEqual(labels_from_data(data).tolist(), [1, -1])


if __name__ == "__main__":
    unittest.main()
