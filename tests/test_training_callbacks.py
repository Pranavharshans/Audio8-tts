from pathlib import Path
from types import SimpleNamespace

from audio8_tts_training_callbacks import (
    PermanentEpochCheckpointCallback,
    load_sample_prompts,
)


def test_load_sample_prompts_accepts_seeded_variants(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"id":"first","text":"ഒന്ന്","seed_offset":0}\n'
        '{"id":"second","text":"രണ്ട്","seed_offset":10}\n',
        encoding="utf-8",
    )

    values = load_sample_prompts(prompts)

    assert [value.sample_id for value in values] == ["first", "second"]
    assert [value.seed_offset for value in values] == [0, 10]


def test_epoch_checkpoint_is_preserved_outside_rotation(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    source = output_dir / "checkpoint-100"
    source.mkdir(parents=True)
    checkpoint_file = source / "model.safetensors"
    checkpoint_file.write_bytes(b"weights")

    class Processor:
        def save_pretrained(self, destination: Path) -> None:
            (destination / "processor.json").write_text("{}\n", encoding="utf-8")

    callback = PermanentEpochCheckpointCallback(Processor(), epochs=(1, 2, 3))
    args = SimpleNamespace(output_dir=str(output_dir))
    state = SimpleNamespace(epoch=1.0, global_step=100, is_world_process_zero=True)
    control = SimpleNamespace(should_save=False)

    callback.on_epoch_end(args, state, control)
    assert control.should_save is True
    callback.on_save(args, state, control)

    destination = output_dir / "epoch_checkpoints" / "epoch-1"
    assert (destination / "model.safetensors").read_bytes() == b"weights"
    assert (destination / "processor.json").is_file()
    assert (destination / "epoch_metadata.json").is_file()
    assert checkpoint_file.stat().st_ino == (destination / "model.safetensors").stat().st_ino
