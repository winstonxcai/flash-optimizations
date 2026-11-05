"""Pinned model metadata for serving and kernel preparation. Author: Winston Cai."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    repository: str
    revision: str
    config_sha256: str
    directory: str
    volume: str
    shards: int
    checkpoint_bytes: int | None
    metadata_files: tuple[str, ...]
    integrations: tuple[str, ...]

    @property
    def config_url(self) -> str:
        return f"https://huggingface.co/{self.repository}/resolve/{self.revision}/config.json"


_PROFILES = {
    "dsv4": ModelProfile(
        repository="deepseek-ai/DeepSeek-V4-Flash-0731",
        revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        config_sha256="6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
        directory="DeepSeek-V4-Flash-0731",
        volume="deepseek-v4-flash-0731",
        shards=48,
        checkpoint_bytes=None,
        metadata_files=("config.json", "model.safetensors.index.json"),
        integrations=("v0.5.18", "v0.5.22"),
    ),
    "glm53": ModelProfile(
        repository="zai-org/GLM-5.3-Flash",
        revision="46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
        config_sha256="bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f",
        directory="GLM-5.3-Flash",
        volume="glm53-flash",
        shards=62,
        checkpoint_bytes=328_337_455_672,
        metadata_files=(
            "config.json", "model.safetensors.index.json", "tokenizer.json",
            "tokenizer_config.json", "chat_template.jinja",
        ),
        integrations=("v0.5.22",),
    ),
}


def model_profile(family: str, integration: str) -> ModelProfile:
    """Resolve pinned assets without importing a runtime or accessing a service.

    V4.1 model-free tests retain their existing dsv4 configuration fixture.
    """
    if family not in _PROFILES:
        raise ValueError(f"Unsupported model family: {family!r}")
    profile = _PROFILES[family]
    if integration not in profile.integrations:
        raise ValueError(f"{family} requires an integration in {profile.integrations}")
    return profile
