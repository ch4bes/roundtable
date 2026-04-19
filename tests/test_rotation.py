from core.config import Config, ModelConfig, DiscussionConfig


class TestRotationConfig:
    def test_rotation_sequential_config(self):
        config = Config(discussion=DiscussionConfig(rotation_order="sequential"))
        assert config.discussion.rotation_order == "sequential"

    def test_rotation_random_config(self):
        config = Config(discussion=DiscussionConfig(rotation_order="random"))
        assert config.discussion.rotation_order == "random"


class TestRotationLogic:
    def test_sequential_rotation_2_models(self):
        models = ["model1", "model2"]
        
        def rotate(round_num):
            rotation = (round_num - 1) % len(models)
            return models[rotation:] + models[:rotation]
        
        assert rotate(1) == ["model1", "model2"]
        assert rotate(2) == ["model2", "model1"]

    def test_single_model_no_rotation(self):
        models = ["only"]
        order = models[0:] + models[:0]
        assert order == ["only"]


class TestRotationWithModelConfigs:
    def test_rotation_with_config_objects(self):
        config = Config(
            models=[
                ModelConfig(name="x"),
                ModelConfig(name="y"),
            ],
            discussion=DiscussionConfig(
                rotation_order="sequential",
                max_rounds=5,
            ),
        )

        assert config.discussion.rotation_order == "sequential"
        assert config.discussion.max_rounds == 5