import torch
import unittest
from mor_model import MoRModel, MoRConfig

class TestMoRModel(unittest.TestCase):

    def test_model_initialization(self):
        config = MoRConfig("mor-135m")
        model = MoRModel(config)
        self.assertIsInstance(model, MoRModel)
        
        # Test parameter count
        params = model.count_parameters()
        print(f"Model parameters: {params}")
        # Should be much less than vanilla 135M due to sharing
        self.assertLess(params['total'], 135_000_000)

    def test_forward_pass_token_choice(self):
        config = MoRConfig("mor-135m", router_type='token_choice')
        model = MoRModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        
        logits, loss = model(input_ids)
        self.assertEqual(logits.shape, (2, 10, config.vocab_size))
        self.assertIsNone(loss)

    def test_forward_pass_expert_choice(self):
        config = MoRConfig("mor-135m", router_type='expert_choice')
        model = MoRModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        
        logits, loss = model(input_ids)
        self.assertEqual(logits.shape, (2, 10, config.vocab_size))
        self.assertIsNone(loss)

    def test_forward_pass_with_labels(self):
        config = MoRConfig("mor-135m")
        model = MoRModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        labels = torch.randint(0, config.vocab_size, (2, 10))
        
        logits, loss = model(input_ids, labels=labels)
        self.assertEqual(logits.shape, (2, 10, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.is_tensor(loss))

    def test_parameter_efficiency(self):
        """Test that MoR uses fewer parameters than vanilla."""
        # MoR model
        config_mor = MoRConfig("mor-360m", num_recursion_steps=3)
        model_mor = MoRModel(config_mor)
        mor_params = model_mor.count_parameters()['total']
        
        print(f"MoR-360M parameters: {mor_params:,}")
        print(f"Expected ~118M (paper): theoretical should be ~360M/3 = 120M")
        
        # Should be roughly 1/3 of vanilla due to 3-way sharing
        self.assertLess(mor_params, 200_000_000)

    def test_different_scales(self):
        """Test all model scales from paper."""
        scales = ["mor-135m", "mor-360m", "mor-730m", "mor-1.7b"]
        
        for scale in scales:
            with self.subTest(scale=scale):
                config = MoRConfig(scale, num_recursion_steps=3)
                model = MoRModel(config)
                params = model.count_parameters()['total']
                print(f"{scale}: {params:,} parameters")
                
                # Test forward pass
                input_ids = torch.randint(0, config.vocab_size, (1, 5))
                logits, _ = model(input_ids)
                self.assertEqual(logits.shape, (1, 5, config.vocab_size))

if __name__ == '__main__':
    unittest.main()