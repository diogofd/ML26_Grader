from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..constants import PUBLIC_DATASET_COLUMNS


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: Path
    expected_columns: tuple[str, ...] = Field(default=PUBLIC_DATASET_COLUMNS)
    row_count_hint: int | None = Field(default=None, ge=0)
    includes_label: bool = True


PUBLIC_DATASETS: tuple[DatasetManifest, ...] = (
    DatasetManifest(
        name="complaints_training",
        path=Path("data/complaints_training.csv"),
        row_count_hint=374148,
        includes_label=True,
    ),
    DatasetManifest(
        name="complaints_test",
        path=Path("data/complaints_test.csv"),
        row_count_hint=268073,
        includes_label=True,
    ),
    DatasetManifest(
        name="complaints_modeltesting100",
        path=Path("data/complaints_modeltesting100.csv"),
        row_count_hint=143,
        includes_label=True,
    ),
)


def dataset_manifest_map(base_dir: Path = Path(".")) -> dict[str, DatasetManifest]:
    return {
        manifest.name: manifest.model_copy(update={"path": base_dir / manifest.path})
        for manifest in PUBLIC_DATASETS
    }
